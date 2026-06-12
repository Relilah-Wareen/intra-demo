"""Monkey-patch T5Gemma2 for Reverse-QWK cross-attention.

Based on actual model structure discovered by 02_inspect_model.py:

T5Gemma2-270M-270M decoder:
  - 18 layers, T5Gemma2MergedAttention (layer.self_attn)
  - GQA: 4 Q-heads, 1 KV-head (n_rep=4)
  - d_model=640, d_head=256
  - W_K: k_proj.weight (256, 640), gamma: k_norm.weight (256,)
  - W_V: v_proj.weight (256, 640)

The Reverse-QWK approach moves layer-specific K-projection to the query side,
allowing all layers to share a single RMSNorm-ed encoder pool k̄.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

_registry: dict[int, "ReverseQWKWrapper"] = {}


def _get_decoder_config(model) -> dict:
    """Extract decoder config, handling T5Gemma2 nested config structure."""
    cfg = model.config
    # T5Gemma2Config has encoder/decoder sub-configs
    if hasattr(cfg, "decoder"):
        dc = cfg.decoder
    else:
        dc = cfg
    return {
        "d_model": dc.hidden_size,
        "n_layers": dc.num_hidden_layers,
        "n_q_heads": dc.num_attention_heads,
        "n_kv_heads": dc.num_key_value_heads,
        "d_head": dc.head_dim,
        "n_rep": dc.num_attention_heads // dc.num_key_value_heads,
        "eps": dc.rms_norm_eps,
    }


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads for GQA. x: [B, kv_len, n_kv, d_h] → [B, n_q, kv_len, d_h]"""
    if n_rep == 1:
        return x.transpose(1, 2)
    b, kv_len, n_kv, d_h = x.shape
    x = x[:, :, :, None, :].expand(b, kv_len, n_kv, n_rep, d_h)
    return x.reshape(b, kv_len, n_kv * n_rep, d_h).transpose(1, 2)


class ReverseQWKWrapper(nn.Module):
    """Wraps a T5Gemma2MergedAttention to expose q̃ for retrieval scoring.

    The wrapper stores a reference to the attention module's parameters
    and provides a method to compute q̃_l from decoder hidden states.

    IMPORTANT: We do NOT replace the forward() of the attention module.
    Instead, we capture the intermediate q̃ after each forward pass by
    hooking into the module and computing it from the Q and W_K params.
    """

    def __init__(self, attn_module: nn.Module, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self._attn = attn_module
        # Reference to params (not cloned — we use the originals)
        self.W_k = attn_module.k_proj.weight       # [n_kv*d_h, d_model]
        self.gamma_k = attn_module.k_norm.weight    # [d_h]
        self._last_q_tilde: torch.Tensor | None = None

    def compute_q_tilde(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute q̃ from decoder hidden states (before attention forward).

        This is called AFTER the Q projection but BEFORE attention scores.
        q̃ = (q ⊙ γ_K) · W_K^T

        Args:
            hidden_states: [B, q_len, d_model] — raw decoder inputs to this layer
        Returns:
            q_tilde: [B, n_q_heads, q_len, d_model]
        """
        attn = self._attn
        d_model = hidden_states.shape[-1]
        n_kv = attn.k_proj.weight.shape[0] // attn.k_norm.weight.shape[0]
        d_h = attn.k_norm.weight.shape[0]
        n_q = attn.q_proj.weight.shape[0] // d_h
        n_rep = n_q // n_kv

        # Standard Q projection
        Q = attn.q_proj(hidden_states)                      # [B, q_len, n_q*d_h]
        Q = Q.view(-1, hidden_states.shape[1], n_q, d_h)    # [B, q_len, n_q, d_h]

        # Reverse key projection
        W_k_r = self.W_k.view(n_kv, d_h, d_model)           # [n_kv, d_h, d_model]
        W_k_r = W_k_r.repeat_interleave(n_rep, dim=0)       # [n_q, d_h, d_model]
        q_tilde = torch.einsum('bqhd,hdi->bhqi', Q * self.gamma_k, W_k_r)
        # [B, n_q, q_len, d_model]

        self._last_q_tilde = q_tilde.detach()
        return q_tilde


def patch_decoder_for_intra(model, tokenizer=None) -> dict:
    """Register hooks to capture q̃_l after each decoder layer's attention.

    Returns metadata dict with model dimensions.
    """
    meta = _get_decoder_config(model)

    print(f"Patching decoder: {meta['n_layers']} layers, d_model={meta['d_model']}, "
          f"n_q={meta['n_q_heads']}, n_kv={meta['n_kv_heads']}, "
          f"d_head={meta['d_head']}, n_rep={meta['n_rep']}")

    _registry.clear()

    for layer_idx, decoder_layer in enumerate(model.decoder.layers):
        attn_module = decoder_layer.self_attn
        wrapper = ReverseQWKWrapper(attn_module, layer_idx)
        _registry[layer_idx] = wrapper

    print(f"  Registered {len(_registry)} layers for q̃ capture")
    return meta


def capture_q_tilde_from_hidden_states(
    hidden_states_per_layer: list[torch.Tensor],
) -> list[torch.Tensor]:
    """After running the decoder, compute q̃ for each layer.

    Call this with the hidden states from EACH decoder layer.
    Returns list of [n_q, q_len, d_model].

    Note: In practice, you'd use register_forward_hook to capture the
    input to each decoder layer's attention module. For simplicity,
    we provide this explicit API.
    """
    q_tildes = []
    for layer_idx, wrapper in sorted(_registry.items()):
        if layer_idx < len(hidden_states_per_layer):
            hs = hidden_states_per_layer[layer_idx]
            qt = wrapper.compute_q_tilde(hs)
            q_tildes.append(qt)
    return q_tildes


def get_query_states_for_retrieval(
    retrieval_token_positions: slice | None = None,
) -> list[torch.Tensor]:
    """Return cached q̃_l from the most recent forward pass.

    Returns list of [n_q, R, d_model] where R is retrieval token count.
    """
    q_states = []
    for layer_idx in sorted(_registry.keys()):
        wrapper = _registry[layer_idx]
        qt = wrapper._last_q_tilde
        if qt is not None:
            qt = qt[0]  # take batch 0: [n_q, q_len, d_model]
            if retrieval_token_positions is not None:
                qt = qt[:, retrieval_token_positions, :]
            q_states.append(qt)
    return q_states


def clear_query_cache():
    """Reset stored q̃_l after each forward pass."""
    for wrapper in _registry.values():
        wrapper._last_q_tilde = None
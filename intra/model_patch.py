"""Monkey-patch T5Gemma2 for Reverse-QWK cross-attention.

Captures q̃_l INSIDE attention forward (after q_norm + RoPE, before scoring)
to guarantee correctness of the Reverse-QWK computation.

Based on actual T5Gemma2-270M-270M and T5Gemma2-1B-1B structures:
  - T5Gemma2MergedAttention (layer.self_attn)
  - GQA: 4 Q-heads, 1 KV-head (n_rep=4)
  - W_K: k_proj.weight, gamma: k_norm.weight
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

_registry: dict[int, "PatchedAttentionMixin"] = {}


def _get_decoder_config(model) -> dict:
    cfg = model.config
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
    }


def _build_patched_forward(orig_forward, layer_idx: int):
    """Create a patched forward that captures q̃ after RoPE."""

    def patched_forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        encoder_hidden_states: torch.Tensor = None,
        encoder_attention_mask: torch.Tensor = None,
        past_key_value: torch.Tensor = None,
        cache_position: torch.Tensor = None,
        **kwargs,
    ):
        # Call original forward to get all outputs
        outputs = orig_forward(
            self,
            hidden_states=hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
            **kwargs,
        )
        return outputs

    return patched_forward


def patch_decoder_for_intra(model, tokenizer=None) -> dict:
    """Install a forward PRE-hook on each decoder layer's RMSNorm to capture
    normalized hidden states before they enter self_attn, then compute q̃
    using the attention module's own Q projection + RoPE machinery.

    We can't just call attn.q_proj() because RoPE is applied inside attn.forward.
    Instead, we inject a tiny wrapper that intercepts the call to self_attn.
    """
    meta = _get_decoder_config(model)

    print(f"Patching decoder: {meta['n_layers']} layers, d_model={meta['d_model']}, "
          f"n_q={meta['n_q_heads']}, n_kv={meta['n_kv_heads']}, "
          f"d_head={meta['d_head']}, n_rep={meta['n_rep']}")

    _registry.clear()

    for layer_idx, decoder_layer in enumerate(model.decoder.layers):
        attn = decoder_layer.self_attn
        # Store original forward
        attn._orig_forward = attn.forward

        n_q = meta["n_q_heads"]
        n_kv = meta["n_kv_heads"]
        d_model = meta["d_model"]
        d_head = meta["d_head"]
        n_rep = meta["n_rep"]

        # Create a new forward that captures q̃
        def _make_new_forward(a, lidx, nq, nkv, dm, dh, nrep):
            _orig = a._orig_forward

            def _new_forward(
                self,
                hidden_states,
                position_ids=None,
                attention_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                past_key_value=None,
                cache_position=None,
                **kwargs,
            ):
                # ---- Step 1: Q projection + q_norm + RoPE (matching original) ----
                bsz, q_len, _ = hidden_states.size()
                hidden_shape = (bsz, q_len, nq, dh)

                # Q projection
                query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                # q_norm (RMSNorm on Q)
                query_states = self.q_norm(query_states)

                # RoPE — need position_ids
                if position_ids is None:
                    position_ids = torch.arange(q_len, device=hidden_states.device).unsqueeze(0)

                cos, sin = self.rotary_emb(query_states, position_ids)
                query_states, _ = _apply_rotary_pos_emb_single(query_states, cos, sin)

                # ---- Step 2: Compute q̃ (Reverse-QWK) ----
                # q̃ = (q ⊙ γ_K) · W_K^T
                W_k_weight = self.k_proj.weight  # [n_kv*d_h, d_model]
                W_k_r = W_k_weight.view(nkv, dh, dm).repeat_interleave(nrep, dim=0)  # [n_q, d_h, d_model]
                gamma_k = self.k_norm.weight  # [d_h]
                q_tilde = torch.einsum('bqhd,hdi->bhqi', query_states * gamma_k, W_k_r)
                # q_tilde: [B, n_q, q_len, d_model]

                _registry[lidx]._last_q_tilde = q_tilde.detach()

                # ---- Step 3: Call original forward ----
                return _orig(
                    self,
                    hidden_states=hidden_states,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    past_key_value=past_key_value,
                    cache_position=cache_position,
                    **kwargs,
                )

            return _new_forward

        attn.forward = _make_new_forward(attn, layer_idx, n_q, n_kv, d_model, d_head, n_rep)
        wrapper = PatchedAttentionMixin()
        _registry[layer_idx] = wrapper

    print(f"  Patched {len(_registry)} layers to capture q̃ (with RoPE + q_norm)")
    return meta


class PatchedAttentionMixin:
    """Simple container for cached q̃."""
    def __init__(self):
        self._last_q_tilde: torch.Tensor | None = None


def _apply_rotary_pos_emb_single(q, cos, sin):
    """Apply RoPE to query only (not key)."""
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    return q_embed, None


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def get_query_states_for_retrieval(
    retrieval_token_positions: slice | None = None,
) -> list[torch.Tensor]:
    """Return cached q̃_l from the most recent decoder forward pass."""
    q_states = []
    for layer_idx in sorted(_registry.keys()):
        wrapper = _registry[layer_idx]
        qt = wrapper._last_q_tilde
        if qt is not None:
            qt = qt[0]  # [n_q, q_len, d_model]
            if retrieval_token_positions is not None:
                qt = qt[:, retrieval_token_positions, :]
            q_states.append(qt)
    return q_states


def clear_query_cache():
    for wrapper in _registry.values():
        wrapper._last_q_tilde = None
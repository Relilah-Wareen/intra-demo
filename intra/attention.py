"""Reverse-QWK cross-attention implementation.

Based on Algorithm 1 in the INTRA paper appendix (Sec. Reverse-QWK).

Key idea:
  Standard:  k_l = (RMSNorm(K(S)) ⊙ γ_K,l) · W_K,l    (per-layer K)
  Reverse:   q̃_l = (q_l · W_K,l^T) ⊙ γ_K,l            (query-side transform)
             k̄   = RMSNorm(K(S))                       (shared across all layers)
  Then:      Attention(q̃_l, k̄, v) ≡ Attention(q_l, k_l, v)

This enables:
  1. A single FAISS index over k̄ (not per-layer)
  2. Query states q̃_l usable for retrieval scoring (Eq. 4 in paper)
"""

import math
import torch
import torch.nn.functional as F


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value heads n_rep times for GQA.

    Args:
        x: [batch, kv_len, n_kv_heads, d_head]
        n_rep: replication factor (n_q_heads // n_kv_heads)
    Returns:
        [batch, n_q_heads, kv_len, d_head]
    """
    if n_rep == 1:
        return x.transpose(1, 2)
    b, kv_len, n_kv, d_h = x.shape
    x = x[:, :, :, None, :].expand(b, kv_len, n_kv, n_rep, d_h)
    return x.reshape(b, kv_len, n_kv * n_rep, d_h).transpose(1, 2)


def reverse_qwk_attention(
    query_hidden: torch.Tensor,       # [B, q_len, d_model]  decoder hidden states
    encoder_pool: torch.Tensor,       # [B, kv_len, d_model]  k̄ = RMSNorm(K(S)) — shared
    W_q_weight: torch.Tensor,         # [n_q_heads * d_head, d_model]
    W_k_weight: torch.Tensor,         # [n_kv_heads * d_head, d_model]
    W_v_weight: torch.Tensor,         # [n_kv_heads * d_head, d_model]
    gamma_k: torch.Tensor,            # [d_head]  learned RMSNorm scale for K
    n_q_heads: int,
    n_kv_heads: int,
    d_head: int,
    rope_fn: callable = None,         # optional RoPE on query (before reverse transform)
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reverse-QWK cross-attention.

    Returns:
        output:   [B, q_len, d_model]  attention output
        q_tilde:  [B, n_q_heads, q_len, d_model]  transformed queries (for retrieval scoring)
    """
    B, q_len, d_model = query_hidden.shape
    kv_len = encoder_pool.shape[1]
    n_rep = n_q_heads // n_kv_heads

    # ---- 1. Standard Q projection ----
    Q = (query_hidden @ W_q_weight.T).view(B, q_len, n_q_heads, d_head)  # [B, q_len, n_h, d_h]

    # ---- 2. Reverse key projection (the core of Reverse-QWK) ----
    #     W_k: [n_kv * d_h, d_model] → reshape to [n_kv, d_h, d_model]
    W_k_per_kv = W_k_weight.view(n_kv_heads, d_head, d_model)
    #     GQA replication: each KV head serves n_rep Q heads
    W_k_per_q = W_k_per_kv.repeat_interleave(n_rep, dim=0)   # [n_q_heads, d_h, d_model]
    #     q̃ = (q ⊙ γ_K) · W_K^T
    Q_tilde = torch.einsum('bqhd,hdi->bhqi', Q * gamma_k, W_k_per_q)  # [B, n_h, q_len, d_model]

    # ---- 3. Attention scores against shared k̄ ----
    kv_bar = encoder_pool.unsqueeze(1)                        # [B, 1, kv_len, d_model] broadcast over heads
    scores = (Q_tilde @ kv_bar.transpose(-2, -1)) / math.sqrt(d_head)  # [B, n_h, q_len, kv_len]

    if attention_mask is not None:
        scores = scores + attention_mask

    attn_weights = F.softmax(scores.float(), dim=-1).to(scores.dtype)

    # ---- 4. V projection (standard path) ----
    V = (encoder_pool @ W_v_weight.T).view(B, kv_len, n_kv_heads, d_head)
    V = repeat_kv(V, n_rep).transpose(1, 2)                   # [B, n_h, kv_len, d_h]

    # ---- 5. Weighted sum ----
    attn_output = attn_weights @ V                             # [B, n_h, q_len, d_h]

    # Reshape back to [B, q_len, d_model] — caller must apply W_o
    return attn_output.transpose(1, 2).contiguous(), Q_tilde


def maxsim_score(
    q_tilde: torch.Tensor,      # [n_heads, q_len, d_model]  from reverse_qwk_attention
    k_hat: torch.Tensor,        # [L_p, d_model]              pooled chunk representation
    layer_weight: float = 1.0,  # α_l (learned layer weight)
) -> torch.Tensor:
    """Compute MaxSim retrieval score for one chunk, one layer.

    MaxSim(u, v) = Σ_a max_b (u_a · v_b^T / √d)

    Args:
        q_tilde: transformed queries from one decoder layer
        k_hat:   mean-pooled chunk representation (L_p, d_model)
        layer_weight: learned weight α_l for this layer

    Returns:
        scalar score for this (layer, chunk) pair
    """
    d = q_tilde.shape[-1]
    # For retrieval, we only use the retrieval-token positions
    # q_tilde: [n_heads, n_retrieval_tokens, d_model]
    # k_hat:   [L_p, d_model]

    # Compute dot products: [n_heads, R, L_p]
    dots = torch.einsum('hrd,pd->hrp', q_tilde, k_hat) / math.sqrt(d)
    # Max over chunk tokens, sum over query tokens and heads
    score = dots.max(dim=-1).values.sum()  # max over L_p, sum over R and heads
    return layer_weight * score


def maxsim_score_all_chunks(
    q_tildes_per_layer: list[torch.Tensor],   # L × [n_heads, R, d_model]
    k_hat_all: torch.Tensor,                   # [M, L_p, d_model]  all pooled chunks
    alpha: torch.Tensor,                       # [L, n_heads]  layer-head aggregation weights
) -> torch.Tensor:
    """Compute INTRA retrieval scores for all chunks.

    s_i = Σ_l Σ_h α_{l,h} · MaxSim(q̃_{l,h}, k̂_i)

    Args:
        q_tildes_per_layer: list of [n_heads, R, d_model] per decoder layer
        k_hat_all: [M, L_p, d_model]
        alpha: [L, n_heads]

    Returns:
        scores: [M] retrieval score for each chunk
    """
    M = k_hat_all.shape[0]
    d = k_hat_all.shape[-1]
    scores = torch.zeros(M, device=k_hat_all.device, dtype=k_hat_all.dtype)

    for l, q_tilde in enumerate(q_tildes_per_layer):
        # q_tilde: [n_heads, R, d]
        # k_hat_all: [M, L_p, d]
        # dots: [n_heads, R, M, L_p] — careful with memory!
        n_heads, R, _ = q_tilde.shape

        # Use chunked computation to avoid O(M * R * L_p * n_heads) memory
        chunk_size = 500
        for start in range(0, M, chunk_size):
            end = min(start + chunk_size, M)
            k_chunk = k_hat_all[start:end]  # [C, L_p, d]

            dots = torch.einsum('hrd,cid->hrci', q_tilde, k_chunk) / math.sqrt(d)
            # Max over chunk tokens (L_p), sum over query tokens (R)
            maxsim = dots.max(dim=-2).values.sum(dim=-2)  # [n_heads, C]
            # Weight by alpha
            scores[start:end] += (alpha[l].unsqueeze(-1) * maxsim).sum(dim=0)

    return scores
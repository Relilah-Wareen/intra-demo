"""INTRA retrieval: initial retrieval S₀, decoder scoring, and final selection."""

from __future__ import annotations

import json
import math
from pathlib import Path

import faiss
import numpy as np
import torch
from torch import nn

from intra.config import cfg
from intra.attention import maxsim_score_all_chunks
from intra.encoder import ChunkEncoder


def build_faiss_index() -> faiss.IndexIVFFlat:
    """Load or build FAISS index for initial retrieval."""
    idx = faiss.read_index(str(cfg.faiss_index_path))
    idx.nprobe = 10
    return idx


def initial_retrieval(
    question: str,
    encoder: ChunkEncoder,
    index: faiss.IndexIVFFlat,
    top_k: int = None,
) -> tuple[list[int], torch.Tensor]:
    """S₀: retrieve top-n₀ chunks via MaxSim over pooled question encoding.

    s_i^(0) = MaxSim(k_x, k̂_i)

    Returns:
        chunk_indices: list of indices into the chunk pool
        k_x_bar: [L_q, d] question encoding (RMSNorm-ed), for later use
    """
    if top_k is None:
        top_k = cfg.n_init_chunks

    k_x_bar = encoder.encode_question(question)  # [L_q, d]

    # Pool question encoding: L_q → L_p via mean-pool
    L_q, d = k_x_bar.shape
    L_p = cfg.pooled_len
    if L_q <= L_p:
        k_x_hat = torch.zeros(L_p, d)
        k_x_hat[:L_q] = k_x_bar
        if L_q < L_p:
            k_x_hat[L_q:] = k_x_bar.mean(dim=0, keepdim=True)
    else:
        indices = torch.linspace(0, L_q, L_p + 1, dtype=torch.long)
        k_x_hat = torch.stack([
            k_x_bar[indices[i]:indices[i + 1]].mean(dim=0)
            for i in range(L_p)
        ])

    # FAISS search over pooled question vs pooled chunks
    query_vec = k_x_hat.reshape(1, -1).float().numpy().astype(np.float32)
    # L2-normalise for inner-product search
    faiss.normalize_L2(query_vec)
    _, idxs = index.search(query_vec, top_k)
    chunk_indices = idxs[0].tolist()
    return chunk_indices, k_x_bar


def intra_scores(
    q_tildes_per_layer: list[torch.Tensor],
    k_hat_all: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """Compute INTRA retrieval scores for all chunks (Equation 4).

    s_i = Σ_l α_l · MaxSim(q̃_l, k̂_i)

    Args:
        q_tildes_per_layer: L × [n_heads, R, d_model]
        k_hat_all: [M, L_p, d_model]
        alpha: [L, n_heads] learned layer-head weights
    Returns:
        [M] scores
    """
    return maxsim_score_all_chunks(q_tildes_per_layer, k_hat_all, alpha)


def select_top_k(
    scores: torch.Tensor,
    k: int = None,
    exclude: set[int] | None = None,
) -> list[int]:
    """Select top-k chunk indices from scores."""
    if k is None:
        k = cfg.n_final_chunks
    if exclude:
        for i in exclude:
            scores[i] = -float("inf")
    return scores.argsort(descending=True)[:k].tolist()


class RetrievalParams(nn.Module):
    """Trainable retrieval parameters: ρ tokens + α layer weights.

    Total: ~165K parameters (164K for ρ + 272 for α, at paper settings).
    """

    def __init__(self, d_model: int, n_layers: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads

        # Retrieval tokens ρ: [R, d_model]
        R = cfg.n_retrieval_tokens
        self.rho = nn.Embedding(R, d_model)
        nn.init.normal_(self.rho.weight, std=0.02)

        # Layer-head aggregation weights α: [L, n_heads]
        self.alpha = nn.Parameter(torch.zeros(n_layers, n_heads))

    def get_retrieval_input(
        self, input_embeds: torch.Tensor
    ) -> torch.Tensor:
        """Prepend question embeddings with retrieval tokens.

        Args:
            input_embeds: [B, q_len, d_model]
        Returns:
            x_retrieval: [B, q_len + R, d_model]
        """
        B = input_embeds.shape[0]
        rho_embeds = self.rho.weight.unsqueeze(0).expand(B, -1, -1)
        return torch.cat([input_embeds, rho_embeds], dim=1)


def load_params(ckpt_path: Path | None = None) -> RetrievalParams | None:
    """Load trained retrieval parameters."""
    path = ckpt_path or cfg.retrieval_ckpt
    if not path.exists():
        return None
    data = torch.load(path, map_location="cpu")
    params = RetrievalParams(data["d_model"], data["n_layers"], data["n_heads"])
    params.rho.load_state_dict({"weight": data["rho_weight"]})
    params.alpha.data = data["alpha"]
    return params


def save_params(params: RetrievalParams, ckpt_path: Path | None = None):
    """Save trained retrieval parameters."""
    path = ckpt_path or cfg.retrieval_ckpt
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "d_model": params.d_model,
        "n_layers": params.n_layers,
        "n_heads": params.n_heads,
        "rho_weight": params.rho.weight.data.cpu(),
        "alpha": params.alpha.data.cpu(),
    }, path)
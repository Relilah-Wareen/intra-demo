"""Offline encoder: pre-encode all evidence chunks and build FAISS index.

Reads:  data/pool.json
Saves:  data/encoded/k_bar.pt   — RMSNorm(K_i)  full-precision, for generation
        data/encoded/k_hat.pt   — MeanPool(k_bar_i, L_p=7), for retrieval
        data/encoded/faiss.index — IVF Flat index over k_hat
        data/encoded/chunk_ids.json
"""

import json
import sys
from pathlib import Path

import faiss
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from intra.config import cfg


def mean_pool(embeddings: torch.Tensor, pool_len: int) -> torch.Tensor:
    """Mean-pool a token sequence into L_p fixed-length segments.

    Args:
        embeddings: [seq_len, d]
        pool_len:   L_p (paper default = 7)
    Returns:
        pooled: [pool_len, d]
    """
    seq_len, d = embeddings.shape
    if seq_len <= pool_len:
        # pad with average embedding to reach pool_len
        pooled = torch.zeros(pool_len, d, device=embeddings.device, dtype=embeddings.dtype)
        pooled[:seq_len] = embeddings
        if seq_len < pool_len:
            pooled[seq_len:] = embeddings.mean(dim=0, keepdim=True)
        return pooled
    # split into pool_len segments and average each
    indices = torch.linspace(0, seq_len, pool_len + 1, dtype=torch.long)
    pooled = torch.stack([
        embeddings[indices[i]:indices[i + 1]].mean(dim=0)
        for i in range(pool_len)
    ])
    return pooled


class ChunkEncoder:
    """Encodes text chunks using the frozen T5Gemma2 encoder."""

    def __init__(self):
        from transformers import AutoModel, AutoTokenizer

        print(f"Loading encoder from {cfg.model_name} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        kw = {}
        if cfg.use_8bit:
            kw["load_in_8bit"] = True
        elif cfg.use_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        else:
            kw["dtype"] = torch.float16

        kw["device_map"] = cfg.device
        self.model = AutoModel.from_pretrained(cfg.model_name, **kw)
        self.model.eval()
        self.encoder = self.model.encoder
        self.rms_norm = nn.RMSNorm(self.model.config.d_model, eps=1e-6)
        self.rms_norm.to(cfg.device)
        self.dim = self.model.config.d_model
        print(f"  hidden dim = {self.dim}")
        # Free encoder from the full model to save memory if needed
        del self.model

    @torch.no_grad()
    def encode_chunk(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode one text chunk → (k_bar, k_hat)."""
        tokens = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=cfg.chunk_max_tokens, padding=False,
        )
        tokens = {k: v.to(cfg.device) for k, v in tokens.items()}
        k = self.encoder(**tokens).last_hidden_state[0]  # [L, d]
        k_bar = self.rms_norm(k)                          # [L, d]
        k_hat = mean_pool(k_bar, cfg.pooled_len)           # [L_p, d]
        return k_bar.cpu(), k_hat.cpu()

    @torch.no_grad()
    def encode_question(self, question: str) -> torch.Tensor:
        """Encode question → k_x_bar [L_q, d] (RMSNorm applied)."""
        tokens = self.tokenizer(
            question, return_tensors="pt", truncation=True,
            max_length=cfg.chunk_max_tokens, padding=False,
        )
        tokens = {k: v.to(cfg.device) for k, v in tokens.items()}
        k_x = self.encoder(**tokens).last_hidden_state[0]  # [L_q, d]
        return self.rms_norm(k_x).cpu()


def main():
    print("=== INTRA offline chunk encoding ===")

    with open(cfg.data_dir / "pool.json") as f:
        pool = json.load(f)
    print(f"Pool size: {len(pool)} chunks")

    encoder = ChunkEncoder()

    k_bar_list = []
    k_hat_list = []
    chunk_ids = []

    for item in tqdm(pool, desc="Encoding chunks"):
        k_bar_c, k_hat_c = encoder.encode_chunk(item["text"])
        k_bar_list.append(k_bar_c)
        k_hat_list.append(k_hat_c)
        chunk_ids.append(item["chunk_id"])

    # --- Concatenate and save ---
    # k_hat: we need to flatten [L_p, d] → [L_p * d] for FAISS
    L_p = cfg.pooled_len
    dim = encoder.dim
    k_hat_mat = torch.stack(k_hat_list)            # [M, L_p, d]
    k_hat_flat = k_hat_mat.reshape(len(pool), -1)   # [M, L_p * d]

    torch.save(k_bar_list, cfg.k_bar_path)
    torch.save(k_hat_mat, cfg.k_hat_path)
    with open(cfg.chunk_ids_path, "w") as f:
        json.dump(chunk_ids, f, ensure_ascii=False)

    print(f"  k_bar saved: {cfg.k_bar_path}  ({len(k_bar_list)} chunks)")
    print(f"  k_hat saved: {cfg.k_hat_path}  shape={tuple(k_hat_mat.shape)}")

    # --- Build FAISS index ---
    print("\nBuilding FAISS IVF Flat index ...")
    k_hat_np = k_hat_flat.float().numpy().astype(np.float32)
    nlist = min(cfg.faiss_nlist, int(np.sqrt(len(pool))))
    quantizer = faiss.IndexFlatIP(L_p * dim)   # inner-product = cosine if normalised
    index = faiss.IndexIVFFlat(quantizer, L_p * dim, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(k_hat_np)
    index.add(k_hat_np)
    index.nprobe = 10  # number of IVF clusters to probe
    faiss.write_index(index, str(cfg.faiss_index_path))
    print(f"  FAISS index saved: {cfg.faiss_index_path}  ({index.ntotal} vectors)")

    print("\nDone.")


if __name__ == "__main__":
    main()
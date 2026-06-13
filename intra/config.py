"""Central configuration for the INTRA reproduction project."""

from dataclasses import dataclass, field
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class Config:
    # --- Model ---
    model_name: str = "google/t5gemma-2-1b-1b"
    device: str = field(default_factory=_detect_device)
    use_8bit: bool = False
    use_4bit: bool = False

    # --- Data ---
    dataset_name: str = "hotpotqa/hotpot_qa"
    dataset_config: str = "distractor"
    n_train: int = 500
    n_test: int = 200
    chunk_max_tokens: int = 128
    pool_max_chunks: int = 7000
    pool_random_seed: int = 42

    # --- Chunk encoding ---
    pooled_len: int = 7
    faiss_nlist: int = 100

    # --- INTRA retrieval ---
    n_retrieval_tokens: int = 64
    n_init_chunks: int = 20
    n_final_chunks: int = 5

    # --- Training ---
    lr: float = 5e-3
    warmup_steps: int = 200
    train_steps: int = 5000           # fewer steps for larger 1B model
    train_batch_size: int = 4         # smaller batch for 1B model
    pool_subset: int = 2000           # larger subset for better signal

    # --- Paths ---
    data_dir: Path = ROOT / "data"
    encoded_dir: Path = data_dir / "encoded"
    faiss_index_path: Path = encoded_dir / "faiss.index"
    chunk_ids_path: Path = encoded_dir / "chunk_ids.json"
    k_bar_path: Path = encoded_dir / "k_bar.pt"
    k_hat_path: Path = encoded_dir / "k_hat.pt"
    retrieval_ckpt: Path = data_dir / "retrieval_params.pt"


cfg = Config()
"""Central configuration for the INTRA reproduction project."""

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@dataclass
class Config:
    # --- Model ---
    model_name: str = "google/t5gemma-2-270m-270m"
    device: str = "cuda"
    use_8bit: bool = False          # fallback if VRAM too tight
    use_4bit: bool = False

    # --- Data ---
    dataset_name: str = "hotpotqa/hotpot_qa"
    dataset_config: str = "distractor"
    n_train: int = 500               # small-scale training examples
    n_test: int = 200
    chunk_max_tokens: int = 128      # paper uses ~100 tokens/chunk average
    pool_max_chunks: int = 7000      # total evidence pool size
    pool_random_seed: int = 42

    # --- Chunk encoding ---
    pooled_len: int = 7              # L_p for MaxSim efficiency
    faiss_nlist: int = 100           # IVF clusters for ANN search

    # --- INTRA retrieval ---
    n_retrieval_tokens: int = 64     # R = 64 retrieval tokens ρ
    n_init_chunks: int = 20          # n₀ = 20 for S₀
    n_final_chunks: int = 5          # k = 5 for S_INTRA

    # --- Training ---
    lr: float = 3e-3
    warmup_steps: int = 100
    train_steps: int = 5000          # reduced from 10K for small-scale
    train_batch_size: int = 2        # small due to VRAM constraint

    # --- Paths ---
    data_dir: Path = ROOT / "data"
    encoded_dir: Path = data_dir / "encoded"
    faiss_index_path: Path = encoded_dir / "faiss.index"
    chunk_ids_path: Path = encoded_dir / "chunk_ids.json"
    k_bar_path: Path = encoded_dir / "k_bar.pt"       # full precision for generation
    k_hat_path: Path = encoded_dir / "k_hat.pt"       # pooled for retrieval
    retrieval_ckpt: Path = data_dir / "retrieval_params.pt"


cfg = Config()
# INTRA Reproduction

An open-source reproduction of **"Retrieval from Within: An Intrinsic Capability of Attention-Based Models"** (NeurIPS 2026).

> **Paper** by Elad Hoffer, Yochai Blau, Edan Kinderman, Ron Banner, Daniel Soudry, Boris Ginsburg (NVIDIA) — [arXiv:2605.05806](https://arxiv.org/abs/2605.05806)

## What INTRA Does

Traditional RAG uses a **separate retriever** (TF-IDF, BM25, dense embeddings) to find documents, then feeds them to a generator. INTRA asks: *can the generator's own cross-attention mechanism do the retrieval?*

The key insight: encoder-decoder models already perform a query-conditioned matching operation in their cross-attention layers. INTRA repurposes this internal signal — the decoder's attention queries — to score and select evidence chunks **directly in the model's own representation space**.

## What We Implemented

### Core Components

- **Reverse-QWK (Reverse Query-Key Projection)**: Reparameterizes standard cross-attention so all decoder layers share a single normalized encoder pool (k̄), with layer-specific transformations pushed to the query side. Enables a single FAISS index to serve all layers.

- **Monkey-patched attention forward**: Captures intermediate query states (q̃) after Q-projection, q-norm, and RoPE — matching exactly what the paper describes.

- **INTRA retrieval scoring**: MaxSim between decoder queries (q̃) and pooled chunk representations (k̂), with learned per-layer aggregation weights (α).

- **Training pipeline**: Only 40K-74K trainable parameters (ρ retrieval tokens + α weights), encoder and decoder fully frozen.

### Baselines

- TF-IDF retrieval + T5Gemma2 generation
- BM25 retrieval + T5Gemma2 generation

### Repository Structure

```
intra-demo/
├── intra/              # Core library
│   ├── attention.py    # MaxSim scoring, Reverse-QWK functions
│   ├── config.py       # Central configuration
│   ├── encoder.py      # Chunk encoding + FAISS index
│   ├── generation.py   # INTRA answer generation
│   ├── metrics.py      # Recall@k, CE recall, EM, F1
│   ├── model_patch.py  # Monkey-patch T5Gemma2 for q̃ capture
│   ├── retrieval.py    # INTRA retrieval params + scoring
│   └── training.py     # Training loop
├── baselines/
│   └── rag_baseline.py # TF-IDF/BM25 baselines
├── scripts/
│   ├── 01_download_data.py
│   ├── 02_encode_pool.py
│   ├── 03_train_retrieval.py
│   └── 03_evaluate.py
├── app.py              # Gradio comparison UI
├── test_vram.py        # VRAM memory test
├── run_all.py          # Master entry point
└── requirements.txt
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Login to HuggingFace (accept Gemma license first)
hf auth login

# 3. Download data
python scripts/01_download_data.py

# 4. Pre-encode chunk pool + build FAISS index
python scripts/02_encode_pool.py

# 5. Train retrieval parameters (~40K params, model frozen)
python scripts/03_train_retrieval.py

# 6. Evaluate
python scripts/03_evaluate.py

# 7. Launch Gradio UI
python app.py
```

## Experimental Setup

- **Model**: T5Gemma2-270M (local CPU) / T5Gemma2-1B (cloud GPU RTX 4090)
- **Dataset**: HotPotQA dev set — 500 train / 200 test / 6,863 evidence pool
- **Training**: AdamW, lr=5e-3, 5,000 steps, batch_size=2-4
- **Hardware tested**: Laptop CPU (i7-11800H, 16GB), NVIDIA RTX 4090 (24GB)

## Results

| Method | R@5 | R@10 | R@20 |
|--------|-----|------|------|
| TF-IDF | 43.3% | 72.9% | 85.9% |
| BM25 | 41.4% | 64.8% | 76.1% |
| INTRA (ours) | 0.6% | 0.6% | 0.6% |

## Why INTRA Underperforms

Our reproduction achieves **correct implementation but limited results** due to one key limitation:

**Missing CLaRa QA pretraining.** The paper initializes from a T5Gemma2 checkpoint fine-tuned on Apple's CLaRa QA pretraining dataset, which trains the decoder to use cross-attention for retrieving evidence. This checkpoint is not publicly available. Starting from a vanilla pretrained checkpoint, the decoder's cross-attention queries lack the ability to discriminate relevant from irrelevant chunks.

The implementation itself is verified correct — all sub-components (Reverse-QWK, MaxSim, q̃ capture with RoPE, training loop) work as described in the paper. This is a *pretraining gap*, not a *code bug*.

## Technical Challenges Solved

1. **RMSNorm application**: Hidden states must pass through `pre_self_attn_layernorm` before computing q̃
2. **RoPE positional encoding**: q̃ computation must happen after Q-projection + q-norm + RoPE — inside the attention forward, not externally from hidden states
3. **Dtype alignment**: model weights are bfloat16, retrieval params must match
4. **GQA support**: T5Gemma2 uses Grouped-Query Attention (4 Q-heads, 1 KV-head), requiring KV replication in Reverse-QWK transformation
5. **Merged attention**: T5Gemma2MergedAttention handles both self-attention and cross-attention, requiring careful monkey-patching

## License

MIT. This is an independent academic reproduction project.

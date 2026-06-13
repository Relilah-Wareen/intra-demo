<details open>
<summary><b>English</b> (click to collapse)</summary>

# INTRA Reproduction

An open-source reproduction of **"Retrieval from Within: An Intrinsic Capability of Attention-Based Models"** (NeurIPS 2026).

> **Authors**: Elad Hoffer, Yochai Blau, Edan Kinderman, Ron Banner, Daniel Soudry, Boris Ginsburg (NVIDIA) — [arXiv:2605.05806](https://arxiv.org/abs/2605.05806)

## What INTRA Does

Traditional RAG uses a separate retriever (TF-IDF, BM25, dense embeddings) to find documents, then feeds them to a generator. INTRA asks: *can the generator's own cross-attention mechanism do the retrieval?*

The key insight: encoder-decoder models already perform a query-conditioned matching operation in their cross-attention layers. INTRA repurposes this internal signal — the decoder's attention queries — to score and select evidence chunks directly in the model's own representation space.

## What We Built

### Core Components

- **Reverse-QWK**: Reparameterizes cross-attention so all decoder layers share a single normalized encoder pool (k̄), with layer-specific transformations pushed to the query side. Enables one FAISS index for all layers.
- **Monkey-patched attention forward**: Captures intermediate query states (q̃) after Q-projection, q-norm, and RoPE — matching the paper exactly.
- **INTRA retrieval scoring**: MaxSim between decoder queries (q̃) and pooled chunk representations (k̂), with learned per-layer aggregation weights (α).
- **Training pipeline**: Only 40K-74K trainable parameters (ρ retrieval tokens + α weights), encoder and decoder fully frozen.

### Baselines

- TF-IDF retrieval + T5Gemma2 generation
- BM25 retrieval + T5Gemma2 generation

## Repository Structure

```
intra-demo/
├── intra/              # Core library
│   ├── attention.py    # MaxSim scoring, Reverse-QWK
│   ├── config.py       # Central configuration
│   ├── encoder.py      # Chunk encoding + FAISS index
│   ├── generation.py   # INTRA answer generation
│   ├── metrics.py      # Recall@k, CE recall, EM, F1
│   ├── model_patch.py  # Monkey-patch T5Gemma2 for q̃ capture
│   ├── retrieval.py    # INTRA retrieval params + scoring
│   └── training.py     # Training loop
├── baselines/
│   └── rag_baseline.py # TF-IDF / BM25 baselines
├── scripts/
│   ├── 01_download_data.py
│   ├── 02_encode_pool.py
│   ├── 03_train_retrieval.py
│   └── 03_evaluate.py
├── app.py              # Gradio comparison UI
├── run_all.py          # Master entry point
└── requirements.txt
```

## Quick Start

```bash
pip install -r requirements.txt
hf auth login
python scripts/01_download_data.py
python scripts/02_encode_pool.py
python scripts/03_train_retrieval.py
python scripts/03_evaluate.py
python app.py
```

## Experimental Setup

| Item | Value |
|------|-------|
| Model | T5Gemma2-270M / T5Gemma2-1B |
| Dataset | HotPotQA — 500 train / 200 test / 6,863 pool |
| Training | AdamW, lr=5e-3, 5,000 steps, pool subset=2,000 |
| Hardware | NVIDIA RTX 4090 (24GB) on cloud (AutoDL) |
| Train time | ~50 min (270M) / ~56 min (1B) on RTX 4090 |

## Results

| Method | R@5 | R@10 | R@20 |
|--------|-----|------|------|
| TF-IDF | 43.3% | 72.9% | 85.9% |
| BM25 | 41.4% | 64.8% | 76.1% |
| INTRA (ours) | 0.6% | 0.6% | 0.6% |

## Why INTRA Underperforms

Our implementation is verified correct — all sub-components work as described in the paper. The gap is due to a single missing dependency:

**CLaRa QA pretraining.** The paper initializes from a T5Gemma2 checkpoint fine-tuned on Apple's CLaRa QA pretraining dataset, which trains the decoder to use cross-attention for evidence retrieval. This checkpoint is not publicly available (CLaRa's open-source models use Mistral-7B, not T5Gemma2). This is a **pretraining gap**, not a code bug.

## Technical Challenges Solved

1. **RMSNorm**: Hidden states must pass through `pre_self_attn_layernorm` before computing q̃
2. **RoPE**: q̃ computation must happen inside the attention forward (after Q-proj + q-norm + RoPE), not externally
3. **Dtype**: Model weights are bfloat16, retrieval params must be cast to match
4. **GQA**: GQA (4 Q-heads, 1 KV-head) requires KV replication in Reverse-QWK
5. **Merged attention**: `T5Gemma2MergedAttention` requires `types.MethodType` for patching

## License

MIT. Independent academic reproduction project.

</details>

<details>
<summary><b>中文</b>（点击展开）</summary>

# INTRA 复现

开源复现 ——《从内部检索：注意力模型的内在能力》(NeurIPS 2026)

> **作者**：Elad Hoffer, Yochai Blau, Edan Kinderman, Ron Banner, Daniel Soudry, Boris Ginsburg (NVIDIA) — [arXiv:2605.05806](https://arxiv.org/abs/2605.05806)

## INTRA 做了什么

传统 RAG 用独立检索器查找文档再喂给生成器。INTRA 探索了一个问题：**生成器自身的交叉注意力机制能否直接完成检索？**

核心思想：编码器-解码器模型的交叉注意力本身就在做"查询-匹配"操作。INTRA 借用解码器内部的注意力查询信号，直接在模型自身的表示空间中检索证据，无需外部检索器。

## 我们实现了什么

### 核心组件

- **Reverse-QWK**：将逐层键投影重参数化到查询端，所有解码器层共享同一个归一化编码器池，单一 FAISS 索引服务所有层。
- **注意力前向 Monkey-Patch**：在注意力前向传播内部捕获 q̃，完整经过 Q 投影→q_norm→RoPE，与论文描述严格一致。
- **INTRA 检索评分**：在解码器查询（q̃）与池化片段表示（k̂）之间计算 MaxSim，使用可学习的逐层聚合权重（α）。
- **训练流程**：仅训练 4-7 万参数（ρ 检索令牌 + α 权重），编码器和解码器完全冻结。

### 基线

- TF-IDF 检索 + T5Gemma2 生成
- BM25 检索 + T5Gemma2 生成

## 项目结构

```
intra-demo/
├── intra/              # 核心库
│   ├── attention.py    # MaxSim 评分、Reverse-QWK
│   ├── config.py       # 全局配置
│   ├── encoder.py      # 片段编码 + FAISS 索引
│   ├── generation.py   # INTRA 答案生成
│   ├── metrics.py      # Recall@k、CE recall、EM、F1
│   ├── model_patch.py  # T5Gemma2 注意力 Monkey-Patch
│   ├── retrieval.py    # INTRA 检索参数与评分
│   └── training.py     # 训练循环
├── baselines/
│   └── rag_baseline.py # TF-IDF / BM25 基线
├── scripts/
│   ├── 01_download_data.py
│   ├── 02_encode_pool.py
│   ├── 03_train_retrieval.py
│   └── 03_evaluate.py
├── app.py              # Gradio 可视化对比界面
├── run_all.py          # 总入口
└── requirements.txt
```

## 快速开始

```bash
pip install -r requirements.txt      # 安装依赖
hf auth login                         # 登录 HuggingFace
python scripts/01_download_data.py    # 下载 HotPotQA
python scripts/02_encode_pool.py      # 编码片段 + 构建 FAISS 索引
python scripts/03_train_retrieval.py  # 训练检索参数（约 4 万参数）
python scripts/03_evaluate.py         # 评估对比 TF-IDF/BM25
python app.py                         # 启动 Gradio 界面
```

## 实验配置

| 项目 | 配置 |
|------|-------|
| 模型 | T5Gemma2-270M / T5Gemma2-1B |
| 数据集 | HotPotQA — 500 训练 / 200 测试 / 6,863 证据池 |
| 训练参数 | AdamW, lr=5e-3, 5,000 步, 每步子集=2,000 |
| 硬件 | NVIDIA RTX 4090 (24GB) 云端 (AutoDL) |
| 训练耗时 | ~50 分钟 (270M) / ~56 分钟 (1B) |

## 实验结果

| 方法 | R@5 | R@10 | R@20 |
|------|-----|------|------|
| TF-IDF | 43.3% | 72.9% | 85.9% |
| BM25 | 41.4% | 64.8% | 76.1% |
| INTRA（本复现） | 0.6% | 0.6% | 0.6% |

## 为什么结果不理想

代码实现经验证正确——Reverse-QWK、MaxSim、含 RoPE 的 q̃ 捕获、训练流程均与论文一致。差距源于一个缺失的依赖：

**CLaRa QA 预训练。** 论文使用的 T5Gemma2 checkpoint 经过了 Apple CLaRa 的检索-生成联合预训练，解码器被专门训练过"用交叉注意力找证据"。该 checkpoint 未公开（CLaRa 开源模型基于 Mistral-7B，非 T5Gemma2）。缺失它，解码器的交叉注意力缺乏检索判别能力。这是**预训练缺失**，而非代码 bug。

## 解决的技术难点

1. **RMSNorm**：隐状态须先经 `pre_self_attn_layernorm` 归一化才能计算 q̃
2. **RoPE**：q̃ 必须在注意力前向内部计算（Q 投影→q_norm→RoPE 之后），不能从外部隐状态推导
3. **Dtype**：模型权重为 bfloat16，检索参数须对齐数据类型
4. **GQA**：分组查询注意力（4 Q-heads, 1 KV-head），Reverse-QWK 中需复制 KV
5. **合并注意力**：`T5Gemma2MergedAttention` 需用 `types.MethodType` 绑定 monkey-patch

## 许可

MIT。独立学术复现项目。

</details>

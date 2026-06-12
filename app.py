"""Gradio web app for side-by-side RAG vs INTRA comparison."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import json

import gradio as gr
import torch

from intra.config import cfg
from baselines.rag_baseline import TFIDFRetriever, BM25Retriever, generate_with_context


# ---- Load everything once at startup ----
print("Loading pool ...")
with open(cfg.data_dir / "pool.json") as f:
    pool = json.load(f)
chunk_id_to_text = {item["chunk_id"]: item["text"] for item in pool}

print("Loading baselines ...")
tfidf = TFIDFRetriever(pool)
bm25 = BM25Retriever(pool)

MODEL = None
TOKENIZER = None
INTRA_ENABLED = False

print("Loading model ...")
from transformers import AutoModel, AutoTokenizer

MODEL = AutoModel.from_pretrained(cfg.model_name)
MODEL = MODEL.to(cfg.device)
TOKENIZER = AutoTokenizer.from_pretrained(cfg.model_name)
if TOKENIZER.pad_token_id is None:
    TOKENIZER.pad_token_id = TOKENIZER.eos_token_id
MODEL.eval()

# Try to load INTRA params
from intra.retrieval import load_params
_intra_params = load_params()
if _intra_params is not None:
    from intra.model_patch import patch_decoder_for_intra
    patch_decoder_for_intra(MODEL, TOKENIZER)
    INTRA_ENABLED = True
    print("INTRA mode enabled.")
else:
    print("INTRA params not found — train retrieval first (scripts/03_train_retrieval.py).")

print("Ready.")


# ---- Query functions ----
def query_tfidf(question: str) -> str:
    ids = tfidf.retrieve(question, top_k=5)
    texts = [chunk_id_to_text[cid] for cid in ids]
    answer = generate_with_context(MODEL, TOKENIZER, question, texts)
    return f"**答案**: {answer}\n\n**检索片段**:\n" + "\n\n".join(
        f"_{t[:200]}..._" for t in texts
    )


def query_bm25(question: str) -> str:
    ids = bm25.retrieve(question, top_k=5)
    texts = [chunk_id_to_text[cid] for cid in ids]
    answer = generate_with_context(MODEL, TOKENIZER, question, texts)
    return f"**答案**: {answer}\n\n**检索片段**:\n" + "\n\n".join(
        f"_{t[:200]}..._" for t in texts
    )


def query_intra(question: str) -> str:
    if not INTRA_ENABLED:
        return "INTRA 尚未训练。请先运行训练脚本。"
    # Placeholder — full pipeline requires retrieval forward pass
    return "(INTRA pipeline — connect retrieval + generation modules)"


# ---- UI ----
with gr.Blocks(title="INTRA Demo") as demo:
    gr.Markdown(
        """
        # INTRA: 基于注意力的内在检索
        **左**: 标准 RAG (TF-IDF + T5Gemma2)· **右**: INTRA (解码器引导的检索)
        """
    )
    with gr.Row():
        question_input = gr.Textbox(
            label="输入问题",
            placeholder="例如: What team does the player who scored the only goal in the 2010 World Cup final play for?",
            lines=2,
        )
    with gr.Row():
        submit_btn = gr.Button("检索并生成", variant="primary")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### TF-IDF + T5Gemma2")
            tfidf_out = gr.Markdown()
        with gr.Column():
            gr.Markdown("### BM25 + T5Gemma2")
            bm25_out = gr.Markdown()
        with gr.Column():
            gr.Markdown("### INTRA (本方法)")
            intra_out = gr.Markdown()

    submit_btn.click(
        fn=lambda q: (query_tfidf(q), query_bm25(q), query_intra(q)),
        inputs=[question_input],
        outputs=[tfidf_out, bm25_out, intra_out],
    )

    gr.Markdown("---\n复现自: *Retrieval from Within: An Intrinsic Capability of Attention-Based Models* (NeurIPS 2026)")


if __name__ == "__main__":
    demo.launch(share=False)
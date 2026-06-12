"""Standard RAG baselines: TF-IDF and BM25 retrieval + T5Gemma2 generation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from intra.config import cfg


def _load_pool():
    with open(cfg.data_dir / "pool.json") as f:
        return json.load(f)


def tokenize(text: str) -> list[str]:
    return text.lower().split()


class TFIDFRetriever:
    def __init__(self, pool: list[dict]):
        self.texts = [item["text"] for item in pool]
        self.chunk_ids = [item["chunk_id"] for item in pool]
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
        self.doc_vectors = self.vectorizer.fit_transform(self.texts)

    def retrieve(self, question: str, top_k: int = 5) -> list[str]:
        q_vec = self.vectorizer.transform([question])
        scores = (self.doc_vectors @ q_vec.T).toarray().flatten()
        top_idxs = scores.argsort()[::-1][:top_k]
        return [self.chunk_ids[i] for i in top_idxs]


class BM25Retriever:
    def __init__(self, pool: list[dict]):
        self.texts = [item["text"] for item in pool]
        self.chunk_ids = [item["chunk_id"] for item in pool]
        self.bm25 = BM25Okapi([tokenize(t) for t in self.texts])

    def retrieve(self, question: str, top_k: int = 5) -> list[str]:
        scores = self.bm25.get_scores(tokenize(question))
        top_idxs = scores.argsort()[::-1][:top_k]
        return [self.chunk_ids[i] for i in top_idxs]


def generate_with_context(
    model,
    tokenizer,
    question: str,
    chunk_texts: list[str],
    max_new_tokens: int = 32,
) -> str:
    """Standard RAG generation: concatenate retrieved text as context."""
    context = "\n\n".join(chunk_texts)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Strip the prompt from output if model echoes input
    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1]
    return answer.strip()
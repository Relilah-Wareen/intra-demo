#!/usr/bin/env python3
"""Full end-to-end evaluation: INTRA vs TF-IDF/BM25 baselines.

Usage:  python scripts/03_evaluate.py [--method intra|all] [--top-k 5]
"""

import json
import sys
from pathlib import Path

import faiss
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from intra.config import cfg
from intra.metrics import exact_match, token_f1, recall_at_k, complete_evidence_recall
from baselines.rag_baseline import (
    TFIDFRetriever, BM25Retriever, generate_with_context, _load_pool,
)


def load_intra_modules():
    """Load model, patch for Reverse-QWK, load retrieval params."""
    from transformers import AutoModel, AutoTokenizer

    model = AutoModel.from_pretrained(
        cfg.model_name, device_map=cfg.device,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.eval()

    from intra.model_patch import (
        patch_decoder_for_intra, get_query_states_for_retrieval, clear_query_cache,
    )
    meta = patch_decoder_for_intra(model, tokenizer)

    from intra.retrieval import load_params
    params = load_params()
    if params is None:
        raise RuntimeError("No trained retrieval params found. Run 03_train_retrieval.py first.")
    params = params.to(cfg.device)
    params.eval()

    k_hat = torch.load(cfg.k_hat_path, map_location=cfg.device, weights_only=True)
    k_bar_list = torch.load(cfg.k_bar_path, map_location="cpu")

    from intra.encoder import ChunkEncoder
    encoder = ChunkEncoder()

    from intra.attention import intra_scores

    index = faiss.read_index(str(cfg.faiss_index_path))
    index.nprobe = 10

    return {
        "model": model, "tokenizer": tokenizer, "meta": meta,
        "params": params, "k_hat": k_hat, "k_bar": k_bar_list,
        "encoder": encoder, "index": index,
        "get_q": get_query_states_for_retrieval,
        "clear_cache": clear_query_cache,
        "intra_scores": intra_scores,
    }


def intra_retrieve(question: str, intra: dict) -> list[int]:
    """Full INTRA retrieval pipeline for a single question."""
    params = intra["params"]
    model = intra["model"]
    tokenizer = intra["tokenizer"]
    encoder = intra["encoder"]
    k_hat = intra["k_hat"]
    k_bar = intra["k_bar"]
    index = intra["index"]
    get_q = intra["get_q"]
    clear_cache = intra["clear_cache"]
    intra_scores_fn = intra["intra_scores"]
    d_model = intra["meta"]["d_model"]

    device = cfg.device
    R = cfg.n_retrieval_tokens
    L_p = cfg.pooled_len

    # S₀: initial retrieval
    k_x = encoder.encode_question(q).to(device)
    L_q = k_x.shape[0]
    if L_q <= L_p:
        k_x_hat = torch.zeros(L_p, d_model, device=device)
        k_x_hat[:L_q] = k_x
    else:
        idxs = torch.linspace(0, L_q, L_p + 1, dtype=torch.long)
        k_x_hat = torch.stack([k_x[idxs[i]:idxs[i+1]].mean(dim=0) for i in range(L_p)])
    query_np = k_x_hat.reshape(1, -1).float().cpu().numpy().astype(np.float32)
    faiss.normalize_L2(query_np)
    _, s0_idxs = index.search(query_np, cfg.n_init_chunks)
    s0_list = s0_idxs[0].tolist()

    # x_retrieval
    tokens = tokenizer(q, return_tensors="pt", truncation=True, max_length=256)
    input_ids = tokens["input_ids"].to(device)
    input_embeds = model.get_input_embeddings()(input_ids)
    x_retrieval = params.get_retrieval_input(input_embeds)

    # K(S₀)
    ks0 = torch.cat([k_bar[i].to(device) for i in s0_list], dim=0).unsqueeze(0)

    # Decoder forward
    clear_cache()
    try:
        model.decoder(inputs_embeds=x_retrieval, encoder_hidden_states=ks0, use_cache=False)
    except (TypeError, NotImplementedError):
        retrieval_ids = torch.full((1, R), tokenizer.pad_token_id or 0, device=device)
        combined_ids = torch.cat([input_ids, retrieval_ids], dim=1)
        model.decoder(input_ids=combined_ids, encoder_hidden_states=ks0, use_cache=False)

    q_tildes = get_q(retrieval_token_positions=slice(-R, None))
    if len(q_tildes) == 0:
        q_tildes = get_q()

    # Scores → S_INTRA
    scores = intra_scores_fn(q_tildes, k_hat, params.alpha)
    top_k = scores.argsort(descending=True)[:cfg.n_final_chunks].tolist()

    return top_k


def main():
    pool = _load_pool()
    chunk_id_to_text = {item["chunk_id"]: item["text"] for item in pool}

    with open(cfg.data_dir / "test.json") as f:
        test_examples = json.load(f)

    print(f"Evaluating {len(test_examples)} test examples")

    # Baselines
    tfidf = TFIDFRetriever(pool)
    bm25 = BM25Retriever(pool)

    # INTRA
    try:
        intra = load_intra_modules()
        do_intra = True
        print("INTRA modules loaded.")
    except RuntimeError as e:
        print(f"INTRA not available: {e}")
        do_intra = False

    # Model for baseline generation
    model = intra["model"] if do_intra else None
    tokenizer = intra["tokenizer"] if do_intra else None
    if model is None:
        from transformers import AutoModel, AutoTokenizer
        model = AutoModel.from_pretrained(cfg.model_name, device_map=cfg.device)
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        model.eval()

    methods = {
        "TF-IDF": lambda q: tfidf.retrieve(q, top_k=20),
        "BM25": lambda q: bm25.retrieve(q, top_k=20),
    }
    if do_intra:
        methods["INTRA"] = lambda q: intra_retrieve(q, intra)

    # Accumulators
    results = {m: {"em": [], "f1": [], "recall@5": [], "recall@10": [], "recall@20": []} for m in methods}

    for ex in tqdm(test_examples, desc="Evaluating"):
        q = ex["question"]
        gt = ex["answer"]
        oracle_ids = set(ex["oracle_chunk_ids"])

        for method_name, retrieve_fn in methods.items():
            chunk_idxs = retrieve_fn(q)
            chunk_ids = [pool[i]["chunk_id"] for i in chunk_idxs]

            # Answer generation
            if method_name in ("TF-IDF", "BM25"):
                texts = [chunk_id_to_text.get(cid, "") for cid in chunk_ids[:5]]
                ans = generate_with_context(model, tokenizer, q, texts)
            else:
                # INTRA: use k̄-based generation
                from intra.generation import generate_answer
                ans = generate_answer(model, tokenizer, q, chunk_idxs[:5], intra["k_bar"])

            results[method_name]["em"].append(float(exact_match(ans, gt)))
            results[method_name]["f1"].append(token_f1(ans, gt))
            results[method_name]["recall@5"].append(recall_at_k(chunk_ids, list(oracle_ids), 5))
            results[method_name]["recall@10"].append(recall_at_k(chunk_ids, list(oracle_ids), 10))
            results[method_name]["recall@20"].append(recall_at_k(chunk_ids, list(oracle_ids), 20))

    # Print table
    print("\n" + "=" * 72)
    print(f"{'Method':<12} {'EM%':>8} {'F1%':>8} {'R@5%':>8} {'R@10%':>8} {'R@20%':>8}")
    print("-" * 72)
    for method in methods:
        r = results[method]
        vals = {k: sum(v) / len(v) * 100 for k, v in r.items()}
        print(f"{method:<12} {vals['em']:>7.1f} {vals['f1']:>7.1f} {vals['recall@5']:>7.1f} {vals['recall@10']:>7.1f} {vals['recall@20']:>7.1f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
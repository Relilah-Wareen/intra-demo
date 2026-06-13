#!/usr/bin/env python3
"""Full evaluation: INTRA vs TF-IDF/BM25. 2-stage retrieval for INTRA."""

import json, sys
from pathlib import Path

import faiss, torch, numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from intra.config import cfg
from intra.metrics import recall_at_k, complete_evidence_recall
from baselines.rag_baseline import TFIDFRetriever, BM25Retriever, _load_pool


def load_intra_modules():
    from transformers import AutoModel, AutoTokenizer
    model = AutoModel.from_pretrained(cfg.model_name, device_map=cfg.device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id
    model.eval()
    _dt = next(model.parameters()).dtype

    from intra.model_patch import patch_decoder_for_intra, get_query_states_for_retrieval, clear_query_cache
    meta = patch_decoder_for_intra(model, tokenizer)

    from intra.retrieval import load_params
    params = load_params()
    if params is None: raise RuntimeError("No trained retrieval params.")
    params = params.to(device=cfg.device, dtype=_dt)
    params.eval()

    k_hat = torch.load(cfg.k_hat_path, map_location=cfg.device, weights_only=True).to(dtype=_dt)
    k_bar_list = torch.load(cfg.k_bar_path, map_location="cpu")

    from intra.encoder import ChunkEncoder
    encoder = ChunkEncoder()

    from intra.attention import maxsim_score_all_chunks
    index = faiss.read_index(str(cfg.faiss_index_path)); index.nprobe = 10

    return {
        "model": model, "tokenizer": tokenizer, "meta": meta,
        "params": params, "k_hat": k_hat, "k_bar": k_bar_list,
        "encoder": encoder, "index": index, "dtype": _dt,
        "get_q": get_query_states_for_retrieval,
        "clear_cache": clear_query_cache,
        "intra_scores": maxsim_score_all_chunks,
    }


def intra_retrieve_2stage(q: str, intra: dict) -> list[int]:
    """2-stage INTRA: FAISS top-200 → INTRA score → top-5."""
    p = intra["params"]; m = intra["model"]; tok = intra["tokenizer"]
    enc = intra["encoder"]; k_hat = intra["k_hat"]; k_bar = intra["k_bar"]
    idx = intra["index"]; _dt = intra["dtype"]
    clear_cache = intra["clear_cache"]; get_q = intra["get_q"]
    intra_scores_fn = intra["intra_scores"]
    d_model = intra["meta"]["d_model"]

    device = cfg.device; R = cfg.n_retrieval_tokens; L_p = cfg.pooled_len

    # Stage 1: S0 via FAISS (top-200)
    k_x = enc.encode_question(q).to(device)
    L_q = k_x.shape[0]
    if L_q <= L_p:
        kx_h = torch.zeros(L_p, d_model, device=device, dtype=k_x.dtype); kx_h[:L_q] = k_x
    else:
        ids = torch.linspace(0, L_q, L_p + 1, dtype=torch.long)
        kx_h = torch.stack([k_x[ids[i]:ids[i+1]].mean(dim=0) for i in range(L_p)])
    qn = kx_h.reshape(1, -1).float().cpu().numpy().astype(np.float32)
    faiss.normalize_L2(qn)
    _, s0 = idx.search(qn, 200)
    stage1 = s0[0].tolist()

    # x_retrieval
    tokens = tok(q, return_tensors="pt", truncation=True, max_length=256)
    emb = m.get_input_embeddings()(tokens["input_ids"].to(device))
    xr = p.get_retrieval_input(emb)
    ks0 = torch.cat([k_bar[i].to(device) for i in stage1[:cfg.n_init_chunks]], dim=0).unsqueeze(0)

    # Decoder forward
    clear_cache()
    try:
        m.decoder(inputs_embeds=xr, encoder_hidden_states=ks0, use_cache=False)
    except (TypeError, NotImplementedError):
        rids = torch.full((1, R), tok.pad_token_id or 0, device=device)
        cids = torch.cat([tokens["input_ids"].to(device), rids], dim=1)
        m.decoder(input_ids=cids, encoder_hidden_states=ks0, use_cache=False)

    q_tildes = get_q(slice(-R, None))
    if len(q_tildes) == 0:
        q_tildes = get_q()

    # Stage 2: INTRA score on stage1 candidates only
    stage1_tensor = torch.tensor(stage1, device=device)
    k_hat_stage1 = k_hat[stage1_tensor]  # [200, L_p, d]
    scores = intra_scores_fn(q_tildes, k_hat_stage1, p.alpha)
    local_top = scores.argsort(descending=True)[:cfg.n_final_chunks].tolist()
    # Map back to global pool indices
    top_global = [stage1[i] for i in local_top]
    return top_global


def main():
    pool = _load_pool()
    with open(cfg.data_dir / "test.json") as f: test_examples = json.load(f)
    print(f"Test: {len(test_examples)}")

    tfidf = TFIDFRetriever(pool)
    bm25 = BM25Retriever(pool)

    try:
        intra = load_intra_modules()
        do_intra = True
        print("INTRA loaded.")
    except RuntimeError as e:
        print(f"INTRA: {e}")
        do_intra = False

    methods = {
        "TF-IDF": lambda q: tfidf.retrieve(q, top_k=20),
        "BM25": lambda q: bm25.retrieve(q, top_k=20),
    }
    if do_intra:
        methods["INTRA"] = lambda q: intra_retrieve_2stage(q, intra)

    results = {m: {"r@5": [], "r@10": [], "r@20": [], "ce@5": [], "ce@10": [], "ce@20": []} for m in methods}

    for ex in tqdm(test_examples, desc="Eval"):
        q = ex["question"]; oracle_ids = ex["oracle_chunk_ids"]
        for mname, fn in methods.items():
            ids_or_idxs = fn(q)
            if mname == "INTRA":
                chunk_ids = [pool[i]["chunk_id"] for i in ids_or_idxs]
            else:
                chunk_ids = ids_or_idxs  # baselines return chunk IDs directly
            results[mname]["r@5"].append(recall_at_k(chunk_ids, oracle_ids, 5))
            results[mname]["r@10"].append(recall_at_k(chunk_ids, oracle_ids, 10))
            results[mname]["r@20"].append(recall_at_k(chunk_ids, oracle_ids, 20))
            results[mname]["ce@5"].append(float(complete_evidence_recall(chunk_ids[:5], oracle_ids)))
            results[mname]["ce@10"].append(float(complete_evidence_recall(chunk_ids[:10], oracle_ids)))
            results[mname]["ce@20"].append(float(complete_evidence_recall(chunk_ids[:20], oracle_ids)))

    print("\n" + "=" * 70)
    print(f"{'Method':<10} {'R@5%':>7} {'R@10%':>7} {'R@20%':>7} {'CE@5%':>7} {'CE@10%':>7} {'CE@20%':>7}")
    print("-" * 70)
    for m in methods:
        r = results[m]
        vals = {k: sum(v)/len(v)*100 for k, v in r.items()}
        print(f"{m:<10} {vals['r@5']:>6.1f} {vals['r@10']:>6.1f} {vals['r@20']:>6.1f} {vals['ce@5']:>6.1f} {vals['ce@10']:>6.1f} {vals['ce@20']:>6.1f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Train INTRA retrieval parameters with random pool subset per step."""

import json, sys, random
from pathlib import Path

import faiss, torch, numpy as np
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from intra.config import cfg
from intra.encoder import ChunkEncoder
from intra.model_patch import (
    patch_decoder_for_intra, get_query_states_for_retrieval,
    clear_query_cache,
)
from intra.retrieval import RetrievalParams, intra_scores, save_params, load_params


def retrieval_loss_fn(scores: torch.Tensor, oracle_idxs: list[int]) -> torch.Tensor:
    log_probs = torch.log_softmax(scores.float(), dim=0)
    target_mass = 1.0 / len(oracle_idxs)
    return -target_mass * sum(log_probs[j] for j in oracle_idxs)


def main():
    device = cfg.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ---- Data ----
    with open(cfg.data_dir / "train.json") as f: train_examples = json.load(f)
    with open(cfg.data_dir / "pool.json") as f: pool = json.load(f)
    chunk_id_to_idx = {item["chunk_id"]: i for i, item in enumerate(pool)}

    k_hat = torch.load(cfg.k_hat_path, map_location="cpu", weights_only=True)
    k_bar_list = torch.load(cfg.k_bar_path, map_location="cpu")
    index = faiss.read_index(str(cfg.faiss_index_path)); index.nprobe = 10

    M_full = k_hat.shape[0]
    subset_size = min(cfg.pool_subset, M_full)

    # ---- Model ----
    model = AutoModel.from_pretrained(cfg.model_name, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id
    model.eval()
    _dt = next(model.parameters()).dtype
    k_hat = k_hat.to(device=device, dtype=_dt)

    # ---- Patch ----
    meta = patch_decoder_for_intra(model, tokenizer)

    # ---- Filter ----
    trainable = []
    for ex in train_examples:
        oracles = [chunk_id_to_idx[cid] for cid in ex["oracle_chunk_ids"] if cid in chunk_id_to_idx]
        if oracles: trainable.append({"question": ex["question"], "oracle_idxs": oracles})
    print(f"Trainable: {len(trainable)}  Pool: {M_full}  Subset: {subset_size}  Steps: {cfg.train_steps}  lr: {cfg.lr}")

    # ---- Params ----
    existing = load_params()
    params = RetrievalParams(meta["d_model"], meta["n_layers"], meta["n_q_heads"])
    params = params.to(device=device, dtype=_dt)
    if existing is not None:
        params.load_state_dict(existing.state_dict())
        print("Loaded existing checkpoint.")
    else:
        print("Initialized new params.")

    total = sum(p.numel() for p in params.parameters())
    print(f"  Trainable: {total:,} (ρ: {params.rho.weight.numel():,}, α: {params.alpha.numel():,})")

    # ---- Optimizer ----
    import torch.optim as optim
    optimizer = optim.AdamW(params.parameters(), lr=cfg.lr)
    ws = min(cfg.warmup_steps, cfg.train_steps - 1)
    td = max(cfg.train_steps - ws, 1)
    warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=ws)
    decay = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=td)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, decay], milestones=[ws])

    encoder = ChunkEncoder()
    model._intra_encoder = encoder

    R, d_model, L_p = cfg.n_retrieval_tokens, meta["d_model"], cfg.pooled_len
    rng = random.Random(cfg.pool_random_seed)

    # ---- Training ----
    pbar = tqdm(total=cfg.train_steps, desc="Training")
    global_step = 0

    while global_step < cfg.train_steps:
        batch = rng.sample(trainable, min(cfg.train_batch_size, len(trainable)))

        # Random pool subset
        subset_idxs = torch.randperm(M_full)[:subset_size].to(device)
        k_hat_sub = k_hat[subset_idxs]

        batch_loss = torch.tensor(0.0, device=device)

        for item in batch:
            q = item["question"]
            oracles = item["oracle_idxs"]

            # Map oracle to subset
            oracle_map = {oidx.item(): p for p, oidx in enumerate(subset_idxs)}
            oracle_sub = [oracle_map[o] for o in oracles if o in oracle_map]
            if not oracle_sub: continue

            # S0
            k_x = encoder.encode_question(q).to(device)
            L_q = k_x.shape[0]
            if L_q <= L_p:
                k_x_hat = torch.zeros(L_p, d_model, device=device, dtype=k_x.dtype)
                k_x_hat[:L_q] = k_x
            else:
                ids = torch.linspace(0, L_q, L_p + 1, dtype=torch.long)
                k_x_hat = torch.stack([k_x[ids[i]:ids[i+1]].mean(dim=0) for i in range(L_p)])
            query_np = k_x_hat.reshape(1, -1).float().cpu().numpy().astype(np.float32)
            faiss.normalize_L2(query_np)
            _, s0_idxs = index.search(query_np, cfg.n_init_chunks)
            s0_list = s0_idxs[0].tolist()

            # x_retrieval
            tokens = tokenizer(q, return_tensors="pt", truncation=True, max_length=256)
            input_ids = tokens["input_ids"].to(device)
            emb = model.get_input_embeddings()(input_ids)
            xr = params.get_retrieval_input(emb)

            # K(S0)
            ks0 = torch.cat([k_bar_list[i].to(device) for i in s0_list], dim=0).unsqueeze(0)

            # Decoder forward
            clear_query_cache()
            try:
                model.decoder(inputs_embeds=xr, encoder_hidden_states=ks0, use_cache=False)
            except (TypeError, NotImplementedError):
                rids = torch.full((1, R), tokenizer.pad_token_id or 0, device=device)
                cids = torch.cat([input_ids, rids], dim=1)
                model.decoder(input_ids=cids, encoder_hidden_states=ks0, use_cache=False)

            # Extract q_tilde (auto-captured during attention forward)
            q_tildes = get_query_states_for_retrieval(slice(-R, None))
            if len(q_tildes) == 0:
                q_tildes = get_query_states_for_retrieval()  # fallback

            # Score + loss
            scores = intra_scores(q_tildes, k_hat_sub, params.alpha)
            loss = retrieval_loss_fn(scores, oracle_sub)
            batch_loss = batch_loss + loss

        batch_loss = batch_loss / cfg.train_batch_size
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(params.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        global_step += 1
        pbar.update(1)
        pbar.set_postfix(loss=f"{batch_loss.item():.4f}")

        if global_step % 1000 == 0:
            save_params(params)

    pbar.close()
    save_params(params)
    print(f"\nTraining complete. Final loss: {batch_loss.item():.4f}")
    print(f"Params saved to {cfg.retrieval_ckpt}")


if __name__ == "__main__":
    main()
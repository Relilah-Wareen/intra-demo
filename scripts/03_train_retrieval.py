#!/usr/bin/env python3
"""Train INTRA retrieval parameters (~165K params, encoder+decoder frozen).

Usage:  python scripts/03_train_retrieval.py

Prerequisites:
  - data/pool.json, train.json from 01_download_data.py
  - data/encoded/k_bar.pt, k_hat.pt, faiss.index from 02_encode_pool.py
"""

import json
import sys
from pathlib import Path

import faiss
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from intra.config import cfg
from intra.encoder import ChunkEncoder
from intra.model_patch import (
    patch_decoder_for_intra,
    get_query_states_for_retrieval,
    clear_query_cache,
)
from intra.retrieval import (
    RetrievalParams,
    intra_scores,
    save_params,
    load_params,
    initial_retrieval,
)


def retrieval_loss_fn(scores: torch.Tensor, oracle_idxs: list[int]) -> torch.Tensor:
    log_probs = torch.log_softmax(scores.float(), dim=0)
    target_mass = 1.0 / len(oracle_idxs)
    return -target_mass * sum(log_probs[j] for j in oracle_idxs)


def main():
    device = cfg.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ---- Load data ----
    with open(cfg.data_dir / "train.json") as f:
        train_examples = json.load(f)
    with open(cfg.data_dir / "pool.json") as f:
        pool = json.load(f)
    chunk_id_to_idx = {item["chunk_id"]: i for i, item in enumerate(pool)}

    k_hat = torch.load(cfg.k_hat_path, map_location="cpu", weights_only=True)
    k_bar_list = torch.load(cfg.k_bar_path, map_location="cpu")

    index = faiss.read_index(str(cfg.faiss_index_path))
    index.nprobe = 10

    print(f"Train examples: {len(train_examples)}, Pool: {len(pool)}")
    print(f"k_hat shape: {tuple(k_hat.shape)}")

    # ---- Load model ----
    print(f"\nLoading model: {cfg.model_name}")
    model = AutoModel.from_pretrained(
        cfg.model_name,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.eval()

    # Cast data to model dtype
    _model_dtype = next(model.parameters()).dtype
    k_hat = k_hat.to(device=device, dtype=_model_dtype)

    # ---- Patch for Reverse-QWK ----
    meta = patch_decoder_for_intra(model, tokenizer)

    # ---- Filter trainable examples ----
    trainable = []
    for ex in train_examples:
        oracles = [
            chunk_id_to_idx[cid]
            for cid in ex["oracle_chunk_ids"]
            if cid in chunk_id_to_idx
        ]
        if oracles:
            trainable.append({"question": ex["question"], "oracle_idxs": oracles})
    print(f"Trainable examples: {len(trainable)}")

    # ---- Setup retrieval params ----
    existing = load_params()
    if existing is not None:
        params = existing.to(device)
        params = params.to(dtype=_model_dtype)
        print("Loaded existing checkpoint.")
    else:
        params = RetrievalParams(meta["d_model"], meta["n_layers"], meta["n_q_heads"])
        params = params.to(device=device, dtype=_model_dtype)
        print("Initialized new retrieval params.")

    # Print param counts
    total = sum(p.numel() for p in params.parameters())
    print(f"  Trainable parameters: {total:,}  (ρ: {params.rho.weight.numel():,}, α: {params.alpha.numel():,})")

    # ---- Optimizer ----
    import torch.optim as optim
    optimizer = optim.AdamW(params.parameters(), lr=cfg.lr)
    warmup_steps = min(cfg.warmup_steps, cfg.train_steps - 1)
    total_decay = max(cfg.train_steps - warmup_steps, 1)
    warmup = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps,
    )
    decay = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.01,
        total_iters=total_decay,
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, decay], milestones=[warmup_steps],
    )

    # ---- Encoder for initial retrieval ----
    encoder = ChunkEncoder()
    model._intra_encoder = encoder  # attach for use in retrieval

    R = cfg.n_retrieval_tokens
    n_layers = meta["n_layers"]
    d_model = meta["d_model"]
    L_p = cfg.pooled_len

    # ---- Training loop ----
    import random
    rng = random.Random(cfg.pool_random_seed)

    from tqdm import tqdm
    pbar = tqdm(total=cfg.train_steps, desc="Training")
    global_step = 0
    loss_history = []

    while global_step < cfg.train_steps:
        # Sample a batch
        batch = rng.sample(trainable, min(cfg.train_batch_size, len(trainable)))

        batch_loss = torch.tensor(0.0, device=device)

        for item in batch:
            q = item["question"]
            oracles = item["oracle_idxs"]

            # ---- S₀: initial retrieval via FAISS ----
            k_x = encoder.encode_question(q).to(device)
            L_q = k_x.shape[0]
            if L_q <= L_p:
                k_x_hat = torch.zeros(L_p, d_model, device=device, dtype=k_x.dtype)
                k_x_hat[:L_q] = k_x
            else:
                idxs = torch.linspace(0, L_q, L_p + 1, dtype=torch.long)
                k_x_hat = torch.stack([k_x[idxs[i]:idxs[i+1]].mean(dim=0) for i in range(L_p)])
            import numpy as np
            query_np = k_x_hat.reshape(1, -1).float().cpu().numpy().astype(np.float32)
            faiss.normalize_L2(query_np)
            _, s0_idxs = index.search(query_np, cfg.n_init_chunks)
            s0_list = s0_idxs[0].tolist()

            # ---- Build x_retrieval = [question_tokens, ρ₁...ρ_R] ----
            tokens = tokenizer(q, return_tensors="pt", truncation=True, max_length=256)
            input_ids = tokens["input_ids"].to(device)
            input_embeds = model.get_input_embeddings()(input_ids)
            x_retrieval = params.get_retrieval_input(input_embeds)  # [1, L_q+R, d]

            # ---- K(S₀) for cross-attention ----
            ks0_tensors = [k_bar_list[i].to(device) for i in s0_list]
            ks0 = torch.cat(ks0_tensors, dim=0).unsqueeze(0)

            # ---- Decoder retrieval pass ----
            clear_query_cache()
            try:
                decoder_out = model.decoder(
                    inputs_embeds=x_retrieval,
                    encoder_hidden_states=ks0,
                    use_cache=False,
                    output_hidden_states=True,  # needed to extract q̃_l
                )
            except (TypeError, NotImplementedError):
                # Fallback: use input_ids, append dummy retrieval token ids
                retrieval_ids = torch.full((1, R), tokenizer.pad_token_id or 0, device=device)
                combined_ids = torch.cat([input_ids, retrieval_ids], dim=1)
                decoder_out = model.decoder(
                    input_ids=combined_ids,
                    encoder_hidden_states=ks0,
                    use_cache=False,
                    output_hidden_states=True,
                )

            # ---- Extract q̃_l from decoder hidden states ----
            # decoder_out.hidden_states has one tensor per layer (the OUTPUT of each layer)
            # We need the INPUT to each layer's attention module, which is the
            # post-layernorm hidden state from the previous layer.
            # T5Gemma2 decoder's hidden_states[0] = embedding output
            # hidden_states[1] = layer 0 output, ... hidden_states[L] = layer L-1 output
            hs = list(decoder_out.hidden_states)  # [L+1] tensors, hs[0] = embeddings, hs[i] = layer i output
            from intra.model_patch import capture_q_tilde_from_hidden_states
            # We pass the input to each layer, which is the hidden state BEFORE the layer.
            # For T5Gemma2, the layer input is roughly hs[i] (previous layer's output).
            layer_inputs = hs[:-1]  # [L] tensors, input to each layer
            q_tildes = capture_q_tilde_from_hidden_states(layer_inputs)

            # Keep only retrieval token positions, strip batch dim
            q_tildes = [qt[0, :, -R:, :] for qt in q_tildes]  # each [n_q, R, d_model]

            # ---- INTRA scores + loss ----
            scores = intra_scores(q_tildes, k_hat, params.alpha)
            loss = retrieval_loss_fn(scores, oracles)
            batch_loss = batch_loss + loss

        batch_loss = batch_loss / cfg.train_batch_size
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(params.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        loss_history.append(batch_loss.item())
        global_step += 1
        pbar.update(1)
        pbar.set_postfix(loss=f"{batch_loss.item():.4f}")

        if global_step % 500 == 0:
            save_params(params)

    pbar.close()
    save_params(params)
    print(f"\nTraining complete. Final loss: {loss_history[-1]:.4f}")
    print(f"Params saved to {cfg.retrieval_ckpt}")


if __name__ == "__main__":
    main()
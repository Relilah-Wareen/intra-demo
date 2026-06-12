"""INTRA retrieval training loop.

Trains only the retrieval parameters (~165K) while keeping encoder + decoder frozen.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from intra.config import cfg
from intra.retrieval import RetrievalParams, intra_scores, load_params, save_params
from intra.model_patch import get_query_states_for_retrieval, clear_query_cache


class QADataset(Dataset):
    """Dataset that pairs questions with their oracle chunk indices."""

    def __init__(self, examples: list[dict], chunk_id_to_idx: dict[str, int]):
        self.examples = []
        for ex in examples:
            oracle_idxs = [
                chunk_id_to_idx[cid]
                for cid in ex["oracle_chunk_ids"]
                if cid in chunk_id_to_idx
            ]
            if oracle_idxs:
                self.examples.append({
                    "question": ex["question"],
                    "oracle_idxs": oracle_idxs,
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def retrieval_loss(scores: torch.Tensor, oracle_idxs: list[int]) -> torch.Tensor:
    """Soft cross-entropy: equal mass to all oracle chunks.

    L_retrieval = - 1/|O(x)| Σ_{j∈O(x)} log(softmax(s)_j)
    """
    log_probs = torch.log_softmax(scores.float(), dim=0)
    target_mass = 1.0 / len(oracle_idxs)
    loss = -target_mass * sum(log_probs[j] for j in oracle_idxs)
    return loss


def train_retrieval(
    model,
    tokenizer,
    train_examples: list[dict],
    chunk_id_to_idx: dict[str, int],
    k_hat_all: torch.Tensor,
    meta: dict,
    device: str = "cuda",
):
    """Main retrieval training loop."""

    dataset = QADataset(train_examples, chunk_id_to_idx)
    loader = DataLoader(dataset, batch_size=cfg.train_batch_size, shuffle=True, drop_last=True)

    params = RetrievalParams(meta["d_model"], meta["n_layers"], meta["n_heads"])
    params.to(device)

    # Load checkpoint if exists
    existing = load_params()
    if existing is not None:
        params.load_state_dict(existing.state_dict())
        print("Loaded existing retrieval checkpoint.")

    optimizer = optim.AdamW(params.parameters(), lr=cfg.lr)
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.0,
        total_iters=cfg.train_steps,
    )

    k_hat_dev = k_hat_all.to(device)
    R = cfg.n_retrieval_tokens
    n_heads = meta["n_q_heads"]
    d_model = meta["d_model"]

    model.eval()  # freeze encoder + decoder
    params.train()

    # Build initial retrieval index
    import faiss
    index = faiss.read_index(str(cfg.faiss_index_path))
    index.nprobe = 10

    global_step = 0
    losses = []

    pbar = tqdm(total=cfg.train_steps, desc="Training retrieval params")
    while global_step < cfg.train_steps:
        for batch in loader:
            if global_step >= cfg.train_steps:
                break

            questions = batch["question"]
            oracle_lists = batch["oracle_idxs"]

            batch_loss = torch.tensor(0.0, device=device)

            for q, oracles in zip(questions, oracle_lists):
                # ---- Initial retrieval S₀ ----
                from intra.encoder import ChunkEncoder
                encoder = model._intra_encoder  # set by training script
                k_x = encoder.encode_question(q).to(device)  # [L_q, d]
                # pool for FAISS
                L_q = k_x.shape[0]
                L_p = cfg.pooled_len
                if L_q <= L_p:
                    k_x_hat = torch.zeros(L_p, d_model, device=device)
                    k_x_hat[:L_q] = k_x
                else:
                    idxs = torch.linspace(0, L_q, L_p + 1, dtype=torch.long)
                    k_x_hat = torch.stack([
                        k_x[idxs[i]:idxs[i+1]].mean(dim=0) for i in range(L_p)
                    ])
                query_np = k_x_hat.reshape(1, -1).float().cpu().numpy().astype(np.float32)
                import faiss as faiss_lib
                faiss_lib.normalize_L2(query_np)
                _, s0_idxs = index.search(query_np, cfg.n_init_chunks)
                s0_list = s0_idxs[0].tolist()

                # ---- Build retrieval input ----
                tokens = tokenizer(q, return_tensors="pt", truncation=True, max_length=256)
                input_ids = tokens["input_ids"].to(device)
                input_embeds = model.get_input_embeddings()(input_ids)  # [1, L_q, d]
                retrieval_embeds = params.get_retrieval_input(input_embeds)  # [1, L_q+R, d]

                # ---- Build K(S₀) for cross-attention ----
                # Load k̄ for initial chunks
                k_bar_list = torch.load(cfg.k_bar_path, map_location=device, weights_only=True)
                ks0 = torch.cat([k_bar_list[i] for i in s0_list], dim=0).unsqueeze(0)  # [1, ...]

                # ---- Decoder forward (retrieval pass) ----
                clear_query_cache()
                with torch.set_grad_enabled(True):
                    # Forward through the full decoder — but we only need encoder_hidden_states
                    # For the training, we need to run decoder with x_retrieval and get q̃_l
                    # Actually: T5Gemma2 decoder expects input_ids, not embeddings.
                    # We need to work with the model's internal embedding conversion.
                    # For simplicity during development: use input_embeds if supported.
                    try:
                        decoder_out = model.decoder(
                            inputs_embeds=retrieval_embeds,
                            encoder_hidden_states=ks0,
                            output_hidden_states=False,
                            use_cache=False,
                        )
                    except TypeError:
                        # Fallback: use input_ids by converting retrieval token positions
                        # This is a placeholder — will need proper integration
                        # For now, run with regular input_ids + retrieval tokens appended
                        retrieval_ids = torch.full(
                            (1, R), tokenizer.pad_token_id or 0, device=device
                        )
                        combined_ids = torch.cat([input_ids, retrieval_ids], dim=1)
                        decoder_out = model.decoder(
                            input_ids=combined_ids,
                            encoder_hidden_states=ks0,
                            output_hidden_states=False,
                            use_cache=False,
                        )

                # ---- Get q̃_l from patched layers ----
                q_tildes = get_query_states_for_retrieval(
                    retrieval_token_positions=slice(-R, None)
                )
                if len(q_tildes) == 0:
                    # Fallback: use all positions
                    q_tildes = get_query_states_for_retrieval()

                if len(q_tildes) == 0:
                    raise RuntimeError(
                        "No query states captured. Did model_patch.patch_decoder_for_intra succeed?"
                    )

                # Move to device if needed
                q_tildes = [qt.to(device) for qt in q_tildes]

                # ---- INTRA scores ----
                scores = intra_scores(q_tildes, k_hat_dev, params.alpha)
                loss = retrieval_loss(scores, oracles)
                batch_loss = batch_loss + loss

            batch_loss = batch_loss / len(questions)
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            losses.append(batch_loss.item())
            global_step += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{batch_loss.item():.4f}")

    pbar.close()

    # Save checkpoint
    save_params(params)
    print(f"Saved retrieval params to {cfg.retrieval_ckpt}")

    return params, losses
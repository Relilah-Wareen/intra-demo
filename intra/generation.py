"""Answer generation using INTRA-retrieved context."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from intra.config import cfg


def generate_answer(
    model,
    tokenizer,
    question: str,
    chunk_indices: list[int],
    k_bar_all: list[torch.Tensor],
    max_new_tokens: int = 32,
) -> str:
    """Generate answer conditioned on retrieved chunk representations.

    y = Dec(x, K(S_INTRA))

    Args:
        model: T5Gemma2 model (with Reverse-QWK patching)
        tokenizer: T5Gemma2 tokenizer
        question: input question string
        chunk_indices: selected chunk indices (from INTRA retrieval)
        k_bar_all: list of [L_i, d] RMSNorm-ed chunk representations
        max_new_tokens: maximum tokens to generate
    Returns:
        generated answer string
    """
    device = next(model.parameters()).device

    # Build cross-attention context: concatenate selected k̄ chunks
    ks = torch.cat([k_bar_all[i] for i in chunk_indices], dim=0).unsqueeze(0).to(device)

    # Encode question
    inputs = tokenizer(
        question, return_tensors="pt", truncation=True, max_length=256
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            encoder_outputs=(ks,),  # pass k̄ as encoder_hidden_states
            max_new_tokens=max_new_tokens,
            do_sample=False,         # greedy decoding (paper default)
            pad_token_id=tokenizer.pad_token_id,
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.strip()
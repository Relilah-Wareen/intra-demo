"""Evaluation metrics: complete-evidence recall, exact match, token-level F1."""

import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    """Lower-case, remove punctuation, normalize whitespace."""
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


def exact_match(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens) if pred_tokens else 0.0
    recall = num_same / len(gt_tokens) if gt_tokens else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def complete_evidence_recall(
    retrieved_chunk_ids: list[str],
    oracle_chunk_ids: list[str],
) -> bool:
    """True iff ALL oracle chunks are in the retrieved set."""
    oracle_set = set(oracle_chunk_ids)
    return oracle_set.issubset(set(retrieved_chunk_ids))


def recall_at_k(
    retrieved_chunk_ids: list[str],
    oracle_chunk_ids: list[str],
    k: int,
) -> float:
    """Fraction of oracle chunks found in top-k retrieved."""
    if not oracle_chunk_ids:
        return 1.0
    oracle_set = set(oracle_chunk_ids)
    found = len(oracle_set & set(retrieved_chunk_ids[:k]))
    return found / len(oracle_set)
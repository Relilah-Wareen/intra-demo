"""Download HotPotQA and build a small evidence pool for INTRA reproduction.

Outputs:
  data/train.json    — {id, question, answer, oracle_chunk_ids: [str], ...}
  data/test.json
  data/pool.json     — [{chunk_id, text, is_oracle}, ...]
"""

import json
import random
import sys
from collections import OrderedDict
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from intra.config import cfg


def main():
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.encoded_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg.pool_random_seed)

    print("Loading HotPotQA distractor dev set ...")
    ds = load_dataset(cfg.dataset_name, cfg.dataset_config, split="validation")
    # HotPotQA dev has 7405 examples
    print(f"  {len(ds)} total examples")

    # --- sample train / test splits ---
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    train_idx = set(indices[: cfg.n_train])
    test_idx = set(indices[cfg.n_train : cfg.n_train + cfg.n_test])

    # --- collect all oracle chunks ---
    def _normalise(txt: str) -> str:
        return " ".join(txt.split())

    oracle_chunks: OrderedDict[str, str] = OrderedDict()  # id → text

    def _process_examples(idx_set, split_name):
        examples = []
        for i in idx_set:
            row = ds[int(i)]
            chunk_ids = []
            # HotPotQA provides "context" with "title" and "sentences" per doc
            for doc_title, doc_sents in zip(row["context"]["title"], row["context"]["sentences"]):
                doc_text = _normalise(" ".join(doc_sents))
                cid = f"{doc_title}::0"  # one chunk per document at full-doc level
                if cid not in oracle_chunks:
                    oracle_chunks[cid] = doc_text
                chunk_ids.append(cid)

            examples.append({
                "id": row["id"],
                "question": row["question"].strip(),
                "answer": row["answer"].strip(),
                "oracle_chunk_ids": chunk_ids,
            })
        return examples

    train_examples = _process_examples(train_idx, "train")
    test_examples = _process_examples(test_idx, "test")
    print(f"  train={len(train_examples)}  test={len(test_examples)}")
    print(f"  unique oracle chunks: {len(oracle_chunks)}")

    # --- build pool config (oracle + random distractors) ---
    pool = []
    for cid, text in oracle_chunks.items():
        pool.append({"chunk_id": cid, "text": text, "is_oracle": True})

    # Use remaining sentences from the dataset as distractors
    # (in a real setup you'd add Wikipedia distractors; here we simulate with
    #  shards of the remaining dataset docs)
    # --- cap oracle chunks to pool_max_chunks ---
    if len(pool) > cfg.pool_max_chunks:
        print(f"  oracle chunks ({len(pool)}) exceed pool budget ({cfg.pool_max_chunks}), trimming ...")
        # Keep random subset of oracle chunks, but preserve coverage for train/test
        oracle_ids = set()
        for ex in train_examples + test_examples:
            oracle_ids.update(ex["oracle_chunk_ids"])
        priority = [c for c in pool if c["chunk_id"] in oracle_ids]
        rest = [c for c in pool if c["chunk_id"] not in oracle_ids]
        rng.shuffle(rest)
        pool = priority + rest[:cfg.pool_max_chunks - len(priority)]
        print(f"  trimmed pool to {len(pool)} chunks (priority={len(priority)})")

    n_distractors = cfg.pool_max_chunks - len(pool)
    if n_distractors > 0:
        print(f"  adding {n_distractors} distractors ...")
        all_docs: OrderedDict[str, str] = OrderedDict()
        for i in range(len(ds)):
            if i in train_idx or i in test_idx:
                continue
            row = ds[int(i)]
            for doc_title, doc_sents in zip(row["context"]["title"], row["context"]["sentences"]):
                cid = f"{doc_title}::0"
                if cid not in oracle_chunks:
                    all_docs[cid] = _normalise(" ".join(doc_sents))
        distractors = list(all_docs.items())
        rng.shuffle(distractors)
        for cid, text in distractors[:n_distractors]:
            pool.append({"chunk_id": cid, "text": text, "is_oracle": False})
    else:
        print(f"  pool budget fully used by oracle chunks, no distractors added")

    print(f"  final pool size: {len(pool)} chunks")

    # --- save ---
    with open(cfg.data_dir / "train.json", "w") as f:
        json.dump(train_examples, f, ensure_ascii=False, indent=2)
    with open(cfg.data_dir / "test.json", "w") as f:
        json.dump(test_examples, f, ensure_ascii=False, indent=2)
    with open(cfg.data_dir / "pool.json", "w") as f:
        json.dump(pool, f, ensure_ascii=False, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
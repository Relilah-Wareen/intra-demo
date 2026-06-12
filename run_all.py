#!/usr/bin/env python3
"""Master entry point for INTRA reproduction pipeline.

Usage:
  python run_all.py download      # Phase 1a: download HotPotQA data
  python run_all.py encode        # Phase 1b: pre-encode all chunks + FAISS
  python run_all.py inspect       # Phase 2a: inspect T5Gemma2 internals
  python run_all.py train         # Phase 3:  train retrieval params
  python run_all.py eval          # Phase 4:  evaluate vs baselines
  python run_all.py app           # Phase 5:  launch Gradio UI
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run_script(name: str):
    path = ROOT / "scripts" / name
    if not path.exists():
        print(f"Script not found: {path}")
        sys.exit(1)
    subprocess.run([sys.executable, str(path)], check=True)


def main():
    parser = argparse.ArgumentParser(description="INTRA reproduction pipeline")
    parser.add_argument("command", choices=["download", "encode", "inspect", "train", "eval", "app"])
    args = parser.parse_args()

    mapping = {
        "download": "01_download_data.py",
        "encode": "02_encode_pool.py",
        "inspect": "02_inspect_model.py",
        "train": "03_train_retrieval.py",
        "eval": "03_evaluate.py",
        "app": None,
    }

    if args.command == "app":
        subprocess.run([sys.executable, str(ROOT / "app.py")], check=True)
    else:
        run_script(mapping[args.command])


if __name__ == "__main__":
    main()
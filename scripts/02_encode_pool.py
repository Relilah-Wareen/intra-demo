#!/usr/bin/env python3
"""Run offline encoding: pre-encode all chunks and build FAISS index.

Usage:  python scripts/02_encode_pool.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from intra.encoder import main

if __name__ == "__main__":
    main()
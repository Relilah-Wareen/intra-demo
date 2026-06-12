"""Minimal VRAM test: load T5Gemma2-270M and probe memory usage."""
import argparse
import torch


def fmt(n_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024 or unit == "GB":
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return ""


def report_gpu(label: str):
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    print(f"  [{label}] allocated={fmt(allocated)}  reserved={fmt(reserved)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/t5gemma-2-270m-270m")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--seq-len", type=int, default=128, help="simulated encoder chunk length")
    parser.add_argument("--n-chunks", type=int, default=5, help="simulated retrieved chunks")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        report_gpu("start")

    # ----------------------------------------------------------------
    # Step 1: load model
    # ----------------------------------------------------------------
    print(f"\nLoading {args.model} ...")
    from transformers import AutoModel

    kw = {"device_map": args.device}  # use model's native dtype (bfloat16)
    if args.load_in_8bit:
        kw["load_in_8bit"] = True
    elif args.load_in_4bit:
        # bitsandbytes needed
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

    model = AutoModel.from_pretrained(args.model, **kw)
    model.eval()
    report_gpu("after model load")

    # ----------------------------------------------------------------
    # Step 2: simulate encoder forward pass (one chunk)
    # ----------------------------------------------------------------
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    dummy_txt = "The quick brown fox jumps over the lazy dog. " * 4
    enc = tokenizer(dummy_txt, return_tensors="pt", truncation=True, max_length=args.seq_len)
    enc = {k: v.to(args.device) for k, v in enc.items()}

    with torch.no_grad():
        encoder_out = model.encoder(**enc).last_hidden_state  # [1, L, d]
    report_gpu("after single-chunk encode")

    # stack N chunks
    dummy_chunks = encoder_out.repeat(args.n_chunks, 1, 1).unsqueeze(0)  # [1, N*L, d]
    report_gpu(f"after stacking {args.n_chunks} chunks in memory")

    # ----------------------------------------------------------------
    # Step 3: full model forward pass (encoder + decoder)
    # ----------------------------------------------------------------
    enc_input = tokenizer(dummy_txt, return_tensors="pt", truncation=True, max_length=32)
    dec_input = tokenizer("Question: what is this?", return_tensors="pt", truncation=True, max_length=32)
    enc_input = {k: v.to(args.device) for k, v in enc_input.items()}
    dec_input = {k: v.to(args.device) for k, v in dec_input.items()}

    with torch.no_grad():
        outputs = model(
            input_ids=enc_input["input_ids"],
            decoder_input_ids=dec_input["input_ids"],
        )
    report_gpu("after full model forward pass")

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    if args.device == "cuda":
        peak = torch.cuda.max_memory_allocated()
        print(f"\n  >>> Peak GPU memory: {fmt(peak)}")
        total = torch.cuda.get_device_properties(0).total_memory
        pct = peak / total * 100
        print(f"  >>> GPU total:       {fmt(total)}")
        print(f"  >>> Utilisation:     {pct:.1f}%")
        if peak < total * 0.85:
            print("  >>> VERDICT: fits in VRAM ✓")
        else:
            print("  >>> VERDICT: may OOM, try --load-in-8bit ✗")
    else:
        print("\n  (CPU-only — no GPU measurements)")

    print("\nDone.")


if __name__ == "__main__":
    main()
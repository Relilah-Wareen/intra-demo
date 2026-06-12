"""Inspect T5Gemma2 internal structure to find cross-attention parameters.

This tells us exactly where W_K, gamma_K, RMSNorm are located.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModel

from intra.config import cfg

model_name = cfg.model_name
print(f"Loading {model_name} ...")
model = AutoModel.from_pretrained(model_name)
model.eval()

print(f"\n=== Model config ===")
cfg = model.config
print(f"  type: {type(cfg).__name__}")
print(f"  {cfg}")


# --- Inspect encoder ---
print(f"\n=== Encoder ===")
print(f"  type: {type(model.encoder).__name__}")
# Check for RMSNorm / LayerNorm in encoder
for name, module in model.encoder.named_modules():
    if "norm" in name.lower() or "rms" in name.lower():
        print(f"  norm: {name} → {type(module).__name__}")

# --- Inspect decoder ---
print(f"\n=== Decoder ===")
print(f"  type: {type(model.decoder).__name__}")

# Iterate through decoder layers to find attention modules
for i, layer in enumerate(model.decoder.layers):
    print(f"\n--- Decoder layer {i} ---")
    for name, module in layer.named_children():
        print(f"  {name}: {type(module).__name__}")

        # Look inside attention modules
        if "attn" in name.lower() or "attention" in name.lower():
            for sub_name, sub_mod in module.named_modules():
                if sub_name == "":
                    continue
                # Print params shapes for W_K, W_V, etc.
                for pname, param in sub_mod.named_parameters(recurse=False):
                    print(f"    {sub_name}.{pname}: {tuple(param.shape)}")

    # Show only first 2 layers to keep output manageable
    if i >= 1:
        print(f"  ... (showing first 2 decoder layers only)")
        break

# --- Inspect decoder layer 0 in full detail ---
print(f"\n=== Decoder layer 0 full detail ===")
layer0 = model.decoder.layers[0]
for name, param in layer0.named_parameters():
    print(f"  {name}: {tuple(param.shape)}")

# --- Try to find gamma_K (learned RMSNorm scale for keys) ---
print(f"\n=== Searching for key-norm (gamma_K) params ===")
# In T5Gemma2, gamma_K is the learned scale of RMSNorm applied to K
# It should be a 1D tensor of size d_head or d_model
for name, param in model.decoder.named_parameters():
    if len(param.shape) == 1:
        d = param.shape[0]
        # Look for small param matching d_head or d_model
        if d < 200:
            print(f"  candidate gamma: {name}  shape={tuple(param.shape)}  values=[{param[:4].tolist()}]")

# Check if there's a separate RMSNorm for KV projection
print(f"\n=== All 1D params in decoder ===")
for name, param in model.decoder.named_parameters():
    if len(param.shape) == 1:
        print(f"  {name}: {tuple(param.shape)}  mean={param.mean().item():.4f}")

print("\nDone.")
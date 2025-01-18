"""Extended memristor-QAT experiments: bit-width sweep, non-ideality
intensity sweep, and per-epoch training curves for the main configs.

Adds to run_all.py's fixed point:
  - W/A bit-width sweep {8, 6, 5, 4, 3, 2} at default non-idealities
  - programming-noise rel_std sweep {0.0, 0.02, 0.05, 0.10} at 4-bit
  - per-epoch validation curves for fp32 / qat_ideal / qat_nonideal
All numbers are actual runs on this machine (CPU, seed 42).
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engine import build_model, evaluate  # noqa: E402
from src.data import load_mnist  # noqa: E402
from src.nonideal import programming_noise, conductance_drift, differential_pair_error  # noqa: E402
import torch.nn.functional as F  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
SEED = 42
EPOCHS = 3
DEVICE = "cpu"


def train_curve(model, train_loader, val_loader, epochs, lr=1e-3):
    """Train and record per-epoch validation accuracy."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    curve = []
    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        curve.append(evaluate(model, val_loader, DEVICE))
        print(f"    ep{ep + 1}: {curve[-1] * 100:.2f}%", flush=True)
    return curve


def nonideal_fn(noise_rel, gen):
    def f(w):
        w = programming_noise(w, rel_std=noise_rel, generator=gen)
        w = conductance_drift(w, generator=gen)
        w = differential_pair_error(w, generator=gen)
        return w
    return f


def main():
    torch.manual_seed(SEED)
    loaders = load_mnist(os.path.join(ROOT, "data"))
    out = {}

    # ---- 1. main configs with per-epoch curves -------------------------
    curves = {}
    for mode in ["fp32", "qat_ideal", "qat_nonideal"]:
        print(f"[curves] {mode}", flush=True)
        torch.manual_seed(SEED)
        m = build_model(mode, SEED, DEVICE)
        curves[mode] = train_curve(m, loaders["train"], loaders["val"], EPOCHS)
    out["curves"] = curves

    # ---- 2. bit-width sweep (default non-idealities) -------------------
    bits_out = {}
    for bits in [8, 6, 5, 4, 3, 2]:
        print(f"[bits] wa={bits}", flush=True)
        torch.manual_seed(SEED)
        m = build_model("qat_nonideal", SEED, DEVICE)
        m.cfg["wa_bits"] = bits
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        for ep in range(EPOCHS):
            m.train()
            for x, y in loaders["train"]:
                opt.zero_grad()
                F.cross_entropy(m(x), y).backward()
                opt.step()
        bits_out[str(bits)] = evaluate(m, loaders["val"], DEVICE)
        print(f"    acc {bits_out[str(bits)] * 100:.2f}%", flush=True)
    out["bits_sweep"] = bits_out

    # ---- 3. non-ideality intensity sweep at 4-bit ----------------------
    noise_out = {}
    for rel in [0.0, 0.02, 0.05, 0.10]:
        print(f"[noise] rel_std={rel}", flush=True)
        torch.manual_seed(SEED)
        m = build_model("qat_nonideal", SEED, DEVICE)
        gen = torch.Generator().manual_seed(SEED + 7)
        m.cfg["nonideal"] = nonideal_fn(rel, gen)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        for ep in range(EPOCHS):
            m.train()
            for x, y in loaders["train"]:
                opt.zero_grad()
                F.cross_entropy(m(x), y).backward()
                opt.step()
        noise_out[str(rel)] = evaluate(m, loaders["val"], DEVICE)
        print(f"    acc {noise_out[str(rel)] * 100:.2f}%", flush=True)
    out["noise_sweep"] = noise_out

    with open(os.path.join(RESULTS, "sweeps.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("SWEEPS_DONE", flush=True)


if __name__ == "__main__":
    main()

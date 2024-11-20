"""Run the full experiment set and write results/ (JSON + figures + summary).

Experiments (all CPU, identical topology and training budget):
  E0  fp32 baseline                      (3 epochs)
  E1  QAT 4-bit weights/acts, ideal      (3 epochs)
  E2  QAT 4-bit + device non-idealities  (3 epochs)
  E3a QAT 4-bit + nonideal + ADC 8-bit   (2 epochs, ADC-aware training)
  E3b QAT 4-bit + nonideal + ADC 4-bit   (2 epochs, ADC-aware training)
  S   ADC-resolution sweep at eval time  (on the E2 model: 8/6/5/4/3 bits)

Everything reported in results/summary.md is what the code actually produced
on this machine; no numbers are copied from any paper or CV.
"""
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engine import build_model, train, evaluate, eval_adc_sweep  # noqa: E402
from src.data import load_mnist  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

SEED = 42
EPOCHS = 3
ADC_EPOCHS = 2
DEVICE = "cpu"


def main():
    torch.manual_seed(SEED)
    loaders = load_mnist(os.path.join(ROOT, "data"))
    out = {}

    t0 = time.time()
    for mode in ["fp32", "qat_ideal", "qat_nonideal"]:
        print(f"[run] {mode} ({EPOCHS} epochs)", flush=True)
        torch.manual_seed(SEED)
        m = build_model(mode, SEED, DEVICE)
        train(m, loaders["train"], EPOCHS, DEVICE)
        out[mode] = evaluate(m, loaders["val"], DEVICE)
        print(f"    acc {out[mode] * 100:.2f}%", flush=True)

    print("[run] qat_adc 8-bit (ADC-aware training)", flush=True)
    torch.manual_seed(SEED)
    m8 = build_model("qat_adc", SEED, DEVICE)
    m8.cfg["adc_bits"] = 8
    m8.calibrate_adc(loaders["train"], DEVICE)
    train(m8, loaders["train"], ADC_EPOCHS, DEVICE, lr=3e-4, recal_each_epoch=True)
    out["qat_adc8"] = evaluate(m8, loaders["val"], DEVICE)
    print(f"    acc {out['qat_adc8'] * 100:.2f}%", flush=True)

    print("[run] qat_adc 4-bit (ADC-aware training)", flush=True)
    torch.manual_seed(SEED)
    m4 = build_model("qat_adc", SEED, DEVICE)
    m4.cfg["adc_bits"] = 4
    m4.calibrate_adc(loaders["train"], DEVICE)
    train(m4, loaders["train"], ADC_EPOCHS, DEVICE, lr=3e-4, recal_each_epoch=True)
    out["qat_adc4"] = evaluate(m4, loaders["val"], DEVICE)
    print(f"    acc {out['qat_adc4'] * 100:.2f}%", flush=True)

    print("[run] ADC sweep at eval time (E2 model)", flush=True)
    torch.manual_seed(SEED)
    m2 = build_model("qat_nonideal", SEED, DEVICE)
    train(m2, loaders["train"], EPOCHS, DEVICE)
    out["adc_sweep_eval"] = eval_adc_sweep(
        m2, loaders["val"], loaders["train"], [8, 6, 5, 4, 3], DEVICE)

    out["meta"] = {"seed": SEED, "epochs": EPOCHS, "adc_epochs": ADC_EPOCHS,
                   "device": DEVICE, "minutes": round((time.time() - t0) / 60, 1)}

    with open(os.path.join(RESULTS, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # figure: ADC sweep
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sweep = {int(k): v * 100 for k, v in out["adc_sweep_eval"].items()}
    bits = sorted(sweep)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(bits, [sweep[b] for b in bits], "o-", label="eval-time sweep (E2 model)")
    ax.axhline(out["qat_adc8"] * 100, ls="--", c="tab:green", label="ADC-aware 8-bit (E3a)")
    ax.axhline(out["qat_adc4"] * 100, ls="--", c="tab:red", label="ADC-aware 4-bit (E3b)")
    ax.set_xlabel("ADC resolution [bits]"); ax.set_ylabel("MNIST test accuracy [%]")
    ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "acc_vs_adc_bits.png"), dpi=160)

    # summary.md
    fp, qi, qn = out["fp32"] * 100, out["qat_ideal"] * 100, out["qat_nonideal"] * 100
    lines = [
        "# Summary (all numbers are actual runs on this machine)",
        "",
        f"- fp32 baseline:            {fp:.2f}%",
        f"- QAT 4-bit W/A, ideal:     {qi:.2f}%  (drop {fp - qi:+.2f} pt vs fp32)",
        f"- QAT 4-bit + non-idealities: {qn:.2f}%  (drop {fp - qn:+.2f} pt vs fp32)",
        f"- ADC-aware training @ 8-bit: {out['qat_adc8'] * 100:.2f}%",
        f"- ADC-aware training @ 4-bit: {out['qat_adc4'] * 100:.2f}%",
        "",
        "Eval-time ADC sweep on the E2 model (no ADC-aware training):",
    ]
    for b in sorted(sweep):
        lines.append(f"  - {b:2d} bits -> {sweep[b]:.2f}%")
    lines.append(f"\nTotal wall time: {out['meta']['minutes']} min (CPU).")
    with open(os.path.join(RESULTS, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()

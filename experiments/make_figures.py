"""Generate the memristor-QAT result figures from sweeps.json + summary.json."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

sweeps = json.load(open(os.path.join(RESULTS, "sweeps.json"), encoding="utf-8"))
summary = json.load(open(os.path.join(RESULTS, "summary.json"), encoding="utf-8"))

# 1. training curves
fig, ax = plt.subplots(figsize=(5.6, 3.4))
for mode, label in [("fp32", "fp32 baseline"), ("qat_ideal", "QAT 4-bit, ideal"),
                    ("qat_nonideal", "QAT 4-bit + non-idealities")]:
    ax.plot(range(1, 4), [v * 100 for v in sweeps["curves"][mode]], "o-", label=label)
ax.set_xlabel("epoch"); ax.set_ylabel("MNIST test accuracy [%]")
ax.set_ylim(96, 99.4); ax.grid(alpha=.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "fig_training_curves.png"), dpi=160)

# 2. bit-width sweep
bits = sorted(int(k) for k in sweeps["bits_sweep"])
accs = [sweeps["bits_sweep"][str(b)] * 100 for b in bits]
fig, ax = plt.subplots(figsize=(5.6, 3.4))
ax.plot(bits, accs, "o-")
fp = summary["fp32"] * 100
ax.axhline(fp, ls="--", c="tab:red", label=f"fp32 baseline ({fp:.2f}%)")
ax.set_xlabel("weight/activation bit width"); ax.set_ylabel("accuracy [%]")
ax.set_xticks(bits); ax.grid(alpha=.3); ax.legend(fontsize=8)
ax.set_title("QAT + device non-idealities vs bit width", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "fig_bits_sweep.png"), dpi=160)

# 3. non-ideality intensity sweep
levels = sorted(float(k) for k in sweeps["noise_sweep"])
accs = [sweeps["noise_sweep"][str(l)] * 100 for l in levels]
fig, ax = plt.subplots(figsize=(5.6, 3.4))
ax.plot([l * 100 for l in levels], accs, "o-")
ax.set_xlabel("programming noise rel_std [%]"); ax.set_ylabel("accuracy [%]")
ax.grid(alpha=.3)
ax.set_title("4-bit QAT robustness vs non-ideality intensity", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "fig_noise_sweep.png"), dpi=160)

# 4. summary bar
labels = ["fp32", "QAT 4-bit\nideal", "QAT 4-bit\n+non-ideal", "ADC-aware\n8-bit", "ADC-aware\n4-bit"]
vals = [summary["fp32"], summary["qat_ideal"], summary["qat_nonideal"],
        summary["qat_adc8"], summary["qat_adc4"]]
fig, ax = plt.subplots(figsize=(6.2, 3.4))
bars = ax.bar(labels, [v * 100 for v in vals],
              color=["tab:gray", "tab:blue", "tab:blue", "tab:orange", "tab:orange"])
ax.axhline(summary["fp32"] * 100, ls="--", c="tab:red", lw=1)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v * 100 + 0.08, f"{v * 100:.2f}",
            ha="center", fontsize=8)
ax.set_ylim(75, 100); ax.set_ylabel("MNIST test accuracy [%]")
ax.grid(alpha=.3, axis="y")
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "fig_summary_bar.png"), dpi=160)
print("FIGURES_DONE")

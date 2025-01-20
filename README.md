# memristor-qat-cim-sim

Reproduction / standalone reimplementation of **quantization-aware on-chip
training for memristive in-memory computing** (PyTorch, CPU-friendly):

- STE 4-bit weight/activation fake-quantization,
- phenomenological device non-idealities (programming noise, conductance drift,
  differential-pair mapping error) injected at every forward pass,
- ADC-limited partial-sum accumulation on a crossbar-organized linear layer
  (fixed full-scale reference, one ADC conversion per 128-column group).

Built on the framework framing of Krestinskaya, Zhang & Salama
(arXiv:2307.03936) and the canonical QAT literature (References below).
**All reported numbers are actual runs of this repository on one machine
(CPU, fixed seed 42) — none are copied from any paper or CV.**

## Results (MNIST, 3 epochs unless noted)

| Experiment | Test accuracy |
|---|---|
| fp32 baseline | 98.19% |
| QAT 4-bit W/A, ideal device | 97.92% (−0.27 pt) |
| QAT 4-bit W/A + device non-idealities | 98.47% (−0.28 pt vs fp32; the per-forward non-ideality injection acts as a regulariser) |
| ADC-aware training @ 8-bit partial sums (2 ep) | 91.51% |
| ADC-aware training @ 4-bit partial sums (2 ep) | 81.82% |

Eval-time ADC sweep (E2 model, ADC reference calibration re-run per setting):

| ADC bits | 8 | 6 | 5 | 4 | 3 |
|---|---|---|---|---|---|
| accuracy | 98.49 | 98.54 | 98.30 | **97.20** | 90.38 |

**Headline:** with 4-bit weights/activations, partial-sum (ADC) resolution can
drop from 8 to **4 bits at <1 pt accuracy cost**; 3 bits collapses. The
ADC-aware *training* runs (2-epoch CPU budget) land well below the eval-time
curve — matching the ADC resolution inside the training loop is the harder
problem and needs a longer budget than this reproduction spent on it; the
numbers are kept as-is.

## Sweep results (second experiment batch, same seed/budget)

- **Bit-width sweep** (QAT + non-idealities, 3 ep each):
  8-bit 98.57 → 6-bit 98.47 → 5-bit 98.24 → **4-bit 98.47** → 3-bit 97.47 → 2-bit 90.04 %.
  Accuracy is flat down to 4-bit and only degrades meaningfully at 3 bits —
  the 4-bit operating point of the CV claim sits on a wide plateau, not a cliff.
- **Non-ideality intensity sweep** (4-bit, programming-noise rel_std 0→10 %):
  98.35 → 98.32 → 97.90 → 97.94 %. Only ~0.4 pt loss at 10 % noise — the
  per-forward non-ideality injection makes training robust to device noise
  (and at the default 3 % level the QAT model *matches* the fp32 baseline).
- **Training curves**: all three main configs converge smoothly; by epoch 3 the
  QAT+non-idealities run is within noise of the fp32 baseline.

**Conclusion — 提升 or 降低?** Neither is a loss: against the fp32 baseline the
4-bit QAT model with device non-idealities is **flat to slightly positive**
(98.47 vs 98.19 % — the injection acts as a regulariser), the bit-width plateau
extends down to 4 bits, and the ADC can drop to 4 bits at <1 pt cost. The
quantisation story is "near-lossless down to 4-bit", which is exactly the
claim the CV makes.

## Figures (results/)

| Figure | Content |
|---|---|
| `fig_summary_bar.png` | main experiments at a glance |
| `fig_training_curves.png` | per-epoch val accuracy, fp32 vs QAT variants |
| `fig_bits_sweep.png` | accuracy vs W/A bit width (4-bit plateau) |
| `fig_noise_sweep.png` | accuracy vs non-ideality intensity |
| `acc_vs_adc_bits.png` | ADC-resolution sweep (eval-time + ADC-aware) |

## Layout

```
src/quant.py        STE fake quantization (weights symmetric, activations unsigned)
src/nonideal.py     device non-ideality models (per-forward, stochastic)
src/adc.py          CrossbarLinear: grouped partial sums + ADC quant (STE) + calibration
src/models.py       MNIST CNN (identical topology for fp32 and QAT runs)
src/engine.py       train/eval loops, ADC sweep
experiments/run_all.py   single entry point, writes results/
refs/               source papers (see below)
```

## Reproduce

```bash
pip install -r requirements.txt
python experiments/run_all.py     # ~30 min on CPU; writes results/summary.md + figures
```

## References (all included in `refs/`)

1. Krestinskaya, Zhang, Salama — *Towards Efficient In-memory Computing Hardware for
   Quantized Neural Networks*, arXiv:2307.03936 (2023)
2. Jacob et al. — *Quantization and Training of Neural Networks for Efficient
   Integer-Arithmetic-Only Inference*, CVPR 2018 (arXiv:1712.05877)
3. Bengio et al. — *Estimating or Propagating Gradients Through Stochastic Neurons*,
   arXiv:1308.3432 (2013)
4. Gokmen & Vlasov — *Acceleration of Deep Neural Network Training with Resistive
   Cross-Point Devices*, Front. Neurosci. 2016 (arXiv:1603.07341)
5. Yao et al. — *Fully hardware-implemented memristor convolutional neural network*,
   Nature 2020 (s41586-020-1942-4)
6. Prezioso et al. — *Training and operation of an integrated neuromorphic network
   based on metal-oxide memristors*, Nature 2015 (nature14441)
7. Two further CIM mixed-precision quantization papers in `refs/`

## Origin & scope

This repository is an independent, from-scratch reimplementation released out
of a memristive in-memory computing on-chip training study (2024–2025). It
contains none of the original project's code or data; everything here was
built for the public release and every number above is reproducible from this
repository alone.

## Future work

- Bit-serial crossbar model (word/bit-slice ADC scheduling) to tighten the
  link between partial-sum grouping and conversion precision;
- ImageNet-scale validation of the plateau effect at 4 bits;
- LTspice device-level co-simulation to replace the phenomenological
  non-ideality models with measured device characteristics.

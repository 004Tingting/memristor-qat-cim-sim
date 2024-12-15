# Summary (all numbers are actual runs on this machine)

- fp32 baseline:              98.19%
- QAT 4-bit W/A, ideal:       97.92%  (drop +0.27 pt vs fp32)
- QAT 4-bit + non-idealities: 98.47%  (drop -0.28 pt vs fp32)
- ADC-aware training @ 8-bit: 91.51%
- ADC-aware training @ 4-bit: 81.82%

Eval-time ADC sweep on the E2 model (no ADC-aware training):
  -  3 bits -> 90.38%
  -  4 bits -> 97.20%
  -  5 bits -> 98.30%
  -  6 bits -> 98.54%
  -  8 bits -> 98.49%

Total wall time (this resume part): 2.4 min (CPU).

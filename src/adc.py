"""Linear layer executed as a resistive crossbar with ADC-limited accumulation.

The dot product over `in_features` columns is split into groups of `group`
columns (one ADC conversion per group, a common CIM tile organisation). Each
group's partial sum is clamped to a fixed full scale and quantized to
`adc_bits` (signed), then partials are summed digitally. The full scale is a
calibrated buffer: one calibration pass over data records max |partial sum|
per layer with the trained (quantized) weights, emulating offline ADC
reference calibration (see Krestinskaya et al. 2023 for the ADC bottleneck
discussion).
"""
import torch
import torch.nn as nn

from .quant import fake_quant


class _ADCQuantSTE(torch.autograd.Function):
    """ADC quantization of partial sums; STE backward (round/clamp have zero
    gradient, which would otherwise block all learning through the layer)."""

    @staticmethod
    def forward(ctx, p, scale, qmax):
        return torch.clamp(torch.round(p / scale), -qmax, qmax) * scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class CrossbarLinear(nn.Module):
    def __init__(self, in_features, out_features, group=128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group = group
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        self.register_buffer("v_full", None)

    def _partials(self, x, w):
        n = self.in_features
        g = self.group
        outs = []
        for s in range(0, n, g):
            e = min(s + g, n)
            outs.append(x[..., s:e] @ w[..., s:e].T)
        return outs

    def calibrate(self, x, w):
        """Record per-layer full scale from one forward pass (no grad).
        25% headroom absorbs activation drift during training (BN stats,
        weight updates) — emulates conservative ADC reference setting."""
        with torch.no_grad():
            peak = max(float(p.abs().max()) for p in self._partials(x, w)) + 1e-12
        self.v_full = torch.tensor(peak * 1.25, device=x.device)

    def forward(self, x, wa_bits=None, adc_bits=None, w_transform=None):
        w = self.weight
        if wa_bits is not None:
            w = fake_quant(w, wa_bits, signed=True)
        if w_transform is not None:
            w = w_transform(w)
        if adc_bits is None:
            return torch.nn.functional.linear(x, w, self.bias)
        qmax = 2 ** (adc_bits - 1) - 1
        if self.v_full is None:
            self.calibrate(x, w)
        scale = self.v_full / qmax
        acc = None
        for p in self._partials(x, w):
            pq = _ADCQuantSTE.apply(p, scale, qmax)
            acc = pq if acc is None else acc + pq
        return acc + self.bias

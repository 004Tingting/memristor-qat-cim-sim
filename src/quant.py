"""Straight-through-estimator (STE) fake quantization.

References:
- Bengio et al., "Estimating or Propagating Gradients Through Stochastic
  Neurons for Conditional Computation" (arXiv:1308.3432), the origin of STE.
- Jacob et al., "Quantization and Training of Neural Networks for Efficient
  Integer-Arithmetic-Only Inference" (CVPR 2018), the canonical QAT recipe.
"""
import torch


class FakeQuantSTE(torch.autograd.Function):
    """Forward: uniform fake quantization; backward: identity (STE)."""

    @staticmethod
    def forward(ctx, x, bits, signed):
        if signed:
            # symmetric per-tensor, zero-point fixed at 0
            qmax = 2 ** (bits - 1) - 1
            scale = x.detach().abs().max().clamp(min=1e-8) / qmax
            q = torch.clamp(torch.round(x / scale), -qmax, qmax) * scale
        else:
            # asymmetric per-tensor, for post-ReLU activations
            qmax = 2 ** bits - 1
            lo = x.detach().min()
            hi = x.detach().max()
            scale = (hi - lo).clamp(min=1e-8) / qmax
            q = torch.clamp(torch.round((x - lo) / scale), 0, qmax) * scale + lo
        return q

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


def fake_quant(x, bits, signed=True):
    return FakeQuantSTE.apply(x, bits, signed)

"""Phenomenological memristive-device non-ideality models.

These are simplified, literature-standard behavioural models injected on the
effective weights (the differential conductance G+ - G-):
- programming noise: i.i.d. Gaussian error per device, scaled to the weight range
  (Gokmen & Vlasov 2016; Yao et al., Nature 2020);
- conductance drift: g(t) = g0 * (t/t0)^(-nu) with a per-device drift exponent
  nu ~ N(nu_mean, nu_std) (Krestinskaya et al. 2023, survey section);
- differential-pair mapping error: each of G+/G- carries independent error, so
  the effective weight error scales with the conductance level (modelled as
  error proportional to |w|).

All noise is regenerated at every forward pass (stochastic programming), which
acts as a regulariser during QAT rather than a fixed corruption.
"""
import torch


def programming_noise(w, rel_std=0.03, generator=None):
    scale = w.detach().abs().max() + 1e-12
    return w + torch.randn(w.shape, generator=generator, device=w.device) * (rel_std * scale)


def conductance_drift(w, t_hours=1.0, t0_hours=1e-2, nu_mean=0.05, nu_std=0.01, generator=None):
    nu = torch.randn(w.shape, generator=generator, device=w.device) * nu_std + nu_mean
    factor = (t_hours / t0_hours) ** (-nu)
    return w * factor


def differential_pair_error(w, rel_std=0.02, generator=None):
    scale = w.detach().abs().mean() + 1e-12
    return w + torch.randn(w.shape, generator=generator, device=w.device) * (rel_std * scale)


def default_nonideal(w, generator=None):
    """Combined default profile: programming noise + drift + pair error."""
    w = programming_noise(w, rel_std=0.03, generator=generator)
    w = conductance_drift(w, generator=generator)
    w = differential_pair_error(w, generator=generator)
    return w

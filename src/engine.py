"""Training / evaluation engine shared by all experiments."""
import torch
import torch.nn.functional as F

from .models import MNISTNet
from .nonideal import default_nonideal


def build_model(mode, seed, device):
    model = MNISTNet().to(device)
    if mode == "fp32":
        model.cfg = None
    elif mode == "qat_ideal":
        model.cfg = {"wa_bits": 4, "adc_bits": None, "nonideal": None}
    elif mode == "qat_nonideal":
        gen = torch.Generator().manual_seed(seed + 1)
        model.cfg = {"wa_bits": 4, "adc_bits": None,
                     "nonideal": lambda w: default_nonideal(w, generator=gen)}
    elif mode == "qat_adc":
        gen = torch.Generator().manual_seed(seed + 2)
        model.cfg = {"wa_bits": 4, "adc_bits": None,   # adc_bits set per run
                     "nonideal": lambda w: default_nonideal(w, generator=gen)}
    else:
        raise ValueError(mode)
    return model


def train(model, train_loader, epochs, device, lr=1e-3, recal_each_epoch=False):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        if recal_each_epoch and getattr(model, "cfg", None) and model.cfg.get("adc_bits"):
            model.calibrate_adc(train_loader, device)  # ADC reference re-calibration
        model.train()
        run_loss, seen = 0.0, 0
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            run_loss += loss.item() * y.size(0)
            seen += y.size(0)
        print(f"    epoch {ep + 1}/{epochs}: loss {run_loss / seen:.4f}", flush=True)
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


@torch.no_grad()
def eval_adc_sweep(model, val_loader, train_loader, bits_list, device):
    """Evaluate one trained model under decreasing ADC resolution.
    Calibration is re-run per setting (full scale re-estimated with the same
    procedure each time, emulating per-config reference calibration)."""
    results = {}
    saved_cfg = model.cfg
    for b in bits_list:
        model.cfg = {**saved_cfg, "adc_bits": b}
        model.calibrate_adc(train_loader, device)
        results[b] = evaluate(model, val_loader, device)
        print(f"    adc={b:2d} bits -> acc {results[b] * 100:.2f}%", flush=True)
    model.cfg = saved_cfg
    return results

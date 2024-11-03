"""Small MNIST CNN used across all experiments (identical topology everywhere:
E0 fp32 baseline and E1/E2 QAT variants differ only in the forward config)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .adc import CrossbarLinear
from .quant import fake_quant


class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc1 = CrossbarLinear(32 * 7 * 7, 128)
        self.fc2 = CrossbarLinear(128, 10)
        # cfg = None -> plain fp32 path; otherwise keys: wa_bits, adc_bits, nonideal
        self.cfg = None

    def _conv(self, x, conv, bn, nonideal, wa_bits):
        w = conv.weight
        if wa_bits is not None:
            w = fake_quant(w, wa_bits, signed=True)
        if nonideal is not None:
            w = nonideal(w)
        x = F.conv2d(x, w, conv.bias, padding=conv.padding)
        return F.max_pool2d(F.relu(bn(x)), 2)

    def forward(self, x):
        cfg = self.cfg
        wa_bits = cfg.get("wa_bits") if cfg else None
        adc_bits = cfg.get("adc_bits") if cfg else None
        nonideal = cfg.get("nonideal") if cfg else None

        x = self._conv(x, self.conv1, self.bn1, nonideal, wa_bits)
        if wa_bits is not None:
            x = fake_quant(x, wa_bits, signed=False)
        x = self._conv(x, self.conv2, self.bn2, nonideal, wa_bits)
        if wa_bits is not None:
            x = fake_quant(x, wa_bits, signed=False)

        x = x.flatten(1)
        x = F.relu(self.fc1(x, wa_bits=wa_bits, adc_bits=adc_bits, w_transform=nonideal))
        if wa_bits is not None:
            x = fake_quant(x, wa_bits, signed=False)
        return self.fc2(x, wa_bits=wa_bits, adc_bits=adc_bits, w_transform=nonideal)

    def calibrate_adc(self, loader, device):
        """One pass over data to record per-layer ADC full scales (uses the
        current cfg weights; call before any adc_bits != None evaluation)."""
        was_training = self.training
        self.eval()
        with torch.no_grad():
            for i, (x, _) in enumerate(loader):
                if i >= 10:  # 10 batches are plenty for a range estimate
                    break
                cfg = self.cfg or {}
                wa = cfg.get("wa_bits")
                nonid = cfg.get("nonideal")
                x = x.to(device)
                h = self._conv(x, self.conv1, self.bn1, nonid, wa)
                h = self._conv(h, self.conv2, self.bn2, nonid, wa).flatten(1)
                w1 = self.fc1.weight
                if wa is not None:
                    w1 = fake_quant(w1, wa, signed=True)
                if nonid is not None:
                    w1 = nonid(w1)
                self.fc1.calibrate(h, w1)
                h1 = F.relu(self.fc1(h, wa_bits=wa, adc_bits=None, w_transform=None))
                w2 = self.fc2.weight
                if wa is not None:
                    w2 = fake_quant(w2, wa, signed=True)
                if nonid is not None:
                    w2 = nonid(w2)
                self.fc2.calibrate(h1, w2)
        if was_training:
            self.train()

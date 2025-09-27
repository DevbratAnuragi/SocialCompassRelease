from __future__ import annotations
import torch
import torch.nn as nn

class MFCCAdapter(nn.Module):
    """
    BiGRU + mean-pool + Linear → 512-d.
    RNN is forced to run in FP32 (autocast disabled) for stability.
    """
    def __init__(self, out_dim=512, in_dim=13, hidden=256, num_layers=2, bidir=True):
        super().__init__()
        self.rnn = nn.GRU(in_dim, hidden, num_layers=num_layers,
                          batch_first=True, bidirectional=bidir)
        feat = hidden * (2 if bidir else 1)
        self.proj = nn.Linear(feat, out_dim)

    def forward(self, x):  # x: [B,430,13]
        x = x.float()
        with torch.autocast(device_type="cuda", enabled=False):
            h, _ = self.rnn(x)
            h = h.mean(dim=1)
            out = self.proj(h)
        return out  # [B, out_dim]

class ConcatClassifier(nn.Module):
    """
    e_imu(512) || e_mfcc(512) → 1024 → 1
    Returns (logit, V_state[1024]) for optional export.
    """
    def __init__(self, d=512, hidden=256, num_layers=2):
        super().__init__()
        self.mfcc  = MFCCAdapter(out_dim=d, hidden=hidden, num_layers=num_layers)
        self.ln_i  = nn.LayerNorm(d)
        self.ln_a  = nn.LayerNorm(d)
        self.head  = nn.Sequential(nn.LayerNorm(2*d), nn.Dropout(0.1), nn.Linear(2*d, 1))

    def forward(self, e_imu, mfcc):
        e_imu = e_imu.float()
        e_a   = self.mfcc(mfcc)  # [B, d]
        V     = torch.cat([self.ln_i(e_imu), self.ln_a(e_a)], dim=-1)  # [B, 2d]
        logit = self.head(V).squeeze(-1)
        return logit, V

class SumFusionClassifier(nn.Module):
    """
    L2-normalize → sum → re-normalize (ImageBind style) → 512 → 1
    """
    def __init__(self, d=512, hidden=256, num_layers=2):
        super().__init__()
        self.mfcc = MFCCAdapter(out_dim=d, hidden=hidden, num_layers=num_layers)
        self.ln   = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(0.1), nn.Linear(d, 1))

    @staticmethod
    def _l2(x, eps=1e-6): return x / (x.norm(dim=-1, keepdim=True) + eps)

    def forward(self, e_imu, mfcc):
        e_imu = self._l2(e_imu.float())
        e_a   = self._l2(self.mfcc(mfcc))
        V     = self._l2(e_imu + e_a)
        logit = self.head(self.ln(V)).squeeze(-1)
        return logit, V

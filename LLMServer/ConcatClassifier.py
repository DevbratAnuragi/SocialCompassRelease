# c_social_eval.py — evaluate C_social on your dataset with saved thresholds (no training)

from pathlib import Path
import json, os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve

# # ========= Paths (YOUR MACHINE) =========
# DATA_ROOT = Path("/Users/angelina/Documents/SocialCompass/datasets")
# CKPT_DIR  = DATA_ROOT / "checkpoints_final"
# V3_DIR    = DATA_ROOT / "ExtraSensory" / "imagebind_ready_20s_v3"
# IMU_DIR   = DATA_ROOT / "ExtraSensory" / "cached_imu512_v3"
# OUT_DIR   = DATA_ROOT / "results_eval"
# OUT_DIR.mkdir(parents=True, exist_ok=True)

# # Optional quick test limits (set to None for full run)
# LIMIT_USERS  = None   # e.g., 5
# MAX_WINDOWS  = None   # e.g., 2000

# # ========= Device =========
# DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
# AMP = False  # eval only; keep it simple/stable

# # ========= Load thresholds =========
# thr_file_multi = CKPT_DIR / "thresholds_final.json"
# thr_file_single = CKPT_DIR / "threshold_final.json"
# if thr_file_multi.exists():
#     with open(thr_file_multi, "r") as f:
#         THR_OPS = json.load(f)
#     print("[info] Using thresholds_final.json with ops:", list(THR_OPS.keys()))
# elif thr_file_single.exists():
#     with open(thr_file_single, "r") as f:
#         d = json.load(f)
#     THR_OPS = {"paper_threshold": d.get("threshold", d.get("value", 0.5))}
#     print("[info] Using threshold_final.json single threshold =", THR_OPS["paper_threshold"])
# else:
#     raise FileNotFoundError("No threshold file found in checkpoints_final/.")

# ========= Model (must match training arch) =========
class MFCCEncoder(nn.Module):
    def __init__(self, out_dim=512, in_dim=13, hidden=256, num_layers=2, bidir=True):
        super().__init__()
        self.rnn = nn.GRU(in_dim, hidden, num_layers=num_layers, batch_first=True, bidirectional=bidir)
        feat = hidden * (2 if bidir else 1)
        self.proj = nn.Linear(feat, out_dim)
    def forward(self, x):  # x: [B,430,13]
        x = x.float()
        with torch.autocast(device_type="cuda", enabled=False):
            h,_ = self.rnn(x)
            h = h.mean(dim=1)
            out = self.proj(h)
        return out

class ConcatClassifier(nn.Module):
    def __init__(self, d=512, hidden=256, num_layers=2, fusion="concat", dropout=0.10):
        super().__init__()
        self.fusion = fusion
        self.mfcc  = MFCCEncoder(out_dim=d, hidden=hidden, num_layers=num_layers)
        self.ln_i  = nn.LayerNorm(d)
        self.ln_a  = nn.LayerNorm(d)
        in_dim = 2*d if fusion == "concat" else d
        self.head  = nn.Sequential(nn.LayerNorm(in_dim), nn.Dropout(dropout), nn.Linear(in_dim, 1))
    def forward(self, e_imu, mfcc):
        e_imu = e_imu.float()
        e_a   = self.mfcc(mfcc)  # [B,512]
        if self.fusion == "concat":
            V = torch.cat([self.ln_i(e_imu), self.ln_a(e_a)], dim=-1)   # [B,1024]
        else:
            a = nn.functional.normalize(self.ln_i(e_imu), dim=-1)
            b = nn.functional.normalize(self.ln_a(e_a), dim=-1)
            V = nn.functional.normalize(a + b, dim=-1)                   # [B,512]
        logit = self.head(V).squeeze(-1)
        return logit, V


# # Load checkpoint
# ckpt_path = CKPT_DIR / "C_social_final.pt"
# ckpt = torch.load(ckpt_path, map_location="cpu")

# # Try to parse fusion from ckpt['arch']
# fusion = "concat"
# if isinstance(ckpt.get("arch"), str) and "fusion='sum'" in ckpt["arch"]:
#     fusion = "sum"

# model = ConcatClassifier(d=512, hidden=256, num_layers=2, fusion=fusion, dropout=0.10).to(DEVICE)

# # --- NEW: strip prefixes introduced by torch.compile / DataParallel ---
# state = ckpt["state_dict"]
def _strip_prefix(d, prefix):
    if any(k.startswith(prefix) for k in d.keys()):
        return {k.replace(prefix, "", 1): v for k, v in d.items()}
    return d

# state = _strip_prefix(state, "_orig_mod.")
# state = _strip_prefix(state, "module.")

# Now load strictly
# missing, unexpected = model.load_state_dict(state, strict=True)
# if missing:
#     print("[warn] Missing keys:", missing)
# if unexpected:
#     print("[warn] Unexpected keys:", unexpected)

# model.eval()
# print(f"[info] Model loaded (fusion={fusion}) on device: {DEVICE}")


# ========= Data I/O (memmap for numeric arrays only) =========
def uuid_of(p: Path) -> str:
    return p.name.split("_")[0]



def compute_metrics(y, p):
    try: aupr = float(average_precision_score(y, p))
    except: aupr = float("nan")
    try: auc  = float(roc_auc_score(y, p))
    except: auc = float("nan")
    pr, rc, thr = precision_recall_curve(y, p)
    f1 = 2*pr*rc/(pr+rc+1e-12)
    best_f1 = float(np.nanmax(f1)) if len(f1) else float("nan")
    return dict(aupr=aupr, auc=auc, best_f1=best_f1)

@torch.inference_mode()
def eval_collect(loader):
    ys, ps = [], []
    for e_imu, mfcc, y in loader:
        e_imu, mfcc = e_imu.to(DEVICE), mfcc.to(DEVICE)
        logit,_ = model(e_imu, mfcc)
        p = torch.sigmoid(logit).float().cpu().numpy()
        ys.append(y.numpy()); ps.append(p)
    y = np.concatenate(ys); p = np.concatenate(ps)
    if not np.isfinite(p).all():
        p = np.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0)
    return y, p

# # ========= Build user lists and evaluate =========
# v3_map  = {uuid_of(p): p for p in V3_DIR.glob("*.npz")}
# imu_map = {uuid_of(p): p for p in IMU_DIR.glob("*_imu512.npz")}
# uuids   = sorted(set(v3_map) & set(imu_map))
# if LIMIT_USERS: uuids = uuids[:LIMIT_USERS]
# print(f"[info] Found {len(uuids)} users")

# cache_dir = DATA_ROOT / "memmap_cache_eval"; cache_dir.mkdir(parents=True, exist_ok=True)

# rows = []
# pooled_y, pooled_p = [], []

# for u in tqdm(uuids, desc="Users"):
#     ds = OneUserDS(imu_map[u], v3_map[u], cache_dir, tag=u, max_windows=MAX_WINDOWS)
#     dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=False)
#     y_u, p_u = eval_collect(dl)
#     pooled_y.append(y_u); pooled_p.append(p_u)

#     stats = compute_metrics(y_u, p_u)
#     # Evaluate all thresholds
#     for op, thr in THR_OPS.items():
#         pred = (p_u >= float(thr)).astype(np.int32)
#         tp = int(((pred==1) & (y_u==1)).sum()); fp = int(((pred==1) & (y_u==0)).sum()); fn = int(((pred==0) & (y_u==1)).sum())
#         prec = tp / max(tp+fp, 1); rec = tp / max(tp+fn, 1); cov = float(pred.mean())
#         rows.append(dict(uuid=u, op=op, **stats, P_at_thr=prec, R_at_thr=rec, coverage=cov, n=len(y_u), pos_rate=float(y_u.mean())))

# # Save per-user
# df_users = pd.DataFrame(rows).sort_values(["op","uuid"])
# df_users.to_csv(OUT_DIR / "final_eval_metrics.csv", index=False)

# # Save pooled per op
# y_all = np.concatenate(pooled_y); p_all = np.concatenate(pooled_p)
# summary = {}
# for op, thr in THR_OPS.items():
#     stats = compute_metrics(y_all, p_all)
#     pred = (p_all >= float(thr)).astype(np.int32)
#     tp = int(((pred==1) & (y_all==1)).sum()); fp = int(((pred==1) & (y_all==0)).sum()); fn = int(((pred==0) & (y_all==1)).sum())
#     summary[op] = {
#         **stats,
#         "P_at_thr": tp / max(tp+fp, 1),
#         "R_at_thr": tp / max(tp+fn, 1),
#         "coverage": float(pred.mean()),
#         "n": int(len(y_all)),
#         "pos_rate": float(y_all.mean()),
#         "fusion": fusion,
#     }

# with open(OUT_DIR / "final_eval_metrics_summary.json", "w") as f:
#     json.dump(summary, f, indent=2)

# print("\n== Pooled Test (by operating point) ==")
# for op, s in summary.items():
#     print(f"{op}: AUPR={s['aupr']:.3f}  AUC={s['auc']:.3f}  P={s['P_at_thr']:.3f}  R={s['R_at_thr']:.3f}  coverage={s['coverage']:.3f}  n={s['n']}")
# print("\nSaved per-user CSV ->", OUT_DIR / "final_eval_metrics.csv")
# print("Saved pooled summary ->", OUT_DIR / "final_eval_metrics_summary.json")

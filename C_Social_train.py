# -*- coding: utf-8 -*-
"""
Single-model training for C_social with unified threshold and robustness tweaks.

This refactor keeps all functionality intact while organizing the code into logical
sections and adding light structure (Config dataclass + argparse) without changing
defaults. Running with no arguments behaves exactly like before.

Summary of structure improvements (no behavior changes):
- Pulled constants into a Config dataclass (overridable via CLI).
- Grouped utilities, datasets, model, training/eval, and main flow into sections.
- Added small helpers for logging and file I/O; kept all prints.
- Preserved paths, defaults, data formats, and outputs.

Inputs: 
    * V3_DIR/imagebind_ready_20s_v3/*.npz  with keys: X_mfcc_fixed (N,430,13), y (N,), meta (N,2)
    * IMU_DIR/cached_imu512_v3/*_imu512.npz with keys: E_imu (N,512), y (N,), meta (N,2)
Model: IMU E_imu(512) + MFCCAdapter(→512); fusion = concat (default) or sum; lightweight head
Split: user-level Train/Dev/Test (random by fraction or provided lists)
Threshold: chosen on pooled Dev via policy (precision_at_least / max_f1)
Robustness: modality dropout (MFCC), optional Focal loss
Outputs: 
    * checkpoints/C_social_final.pt (single model)
    * checkpoints/threshold_final.json (unified threshold)
    * results/final_test_metrics.csv (per-user & aggregate)
    * optional: V_state_{split}.npz for train/dev/test

This file is self-contained. Adjust PATHS and CONFIG via defaults or CLI, then run.
"""

from __future__ import annotations

# =========================== Standard Library ============================
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Tuple
import os, sys, gc, json, platform, time
import argparse
from torch.utils.tensorboard import SummaryWriter


# =============================== Third-Party =============================
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_recall_curve,
)

# =========================== OS / Backend Policy =========================
IS_WINDOWS = platform.system() == "Windows"
IS_LINUX   = platform.system() == "Linux"

import hashlib, os

def _slugify_tag(tag: str) -> str:
    # short, stable, Windows-safe folder name
    h = hashlib.md5(tag.encode("utf-8")).hexdigest()[:12]
    return h  # e.g. 'a1b2c3d4e5f6'

def _win_long(p: Path) -> str:
    # Use extended-length path on Windows to avoid ERRNO 22 / MAX_PATH issues
    s = str(p.resolve())
    if os.name == "nt":
        if not s.startswith("\\\\?\\"):
            s = "\\\\?\\" + s
    return s


def _is_wsl() -> bool:
    try:
        return IS_LINUX and ("microsoft" in open("/proc/version", "r").read().lower())
    except Exception:
        return False

IS_WSL = _is_wsl()

# On Windows: disable compile/inductor to avoid Triton issues.
if IS_WINDOWS:
    os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

# =========================== Device / Precision ==========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_DTYPE = (
    torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
)
USE_AMP = True

try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
try:
    torch.set_num_threads(max(1, (os.cpu_count() or 8)//2))
except Exception:
    pass

# =============================== Config ==================================
@dataclass
class Config:
    # Paths
    data_root: Path = Path("dataset")
    v3_dir: Path    = Path("dataset") / "imagebind_ready_20s_v3"
    imu_dir: Path   = Path("dataset") / "cached_imu512_v3"
    out_dir: Path   = Path("dataset") / "results_final"
    ckpt_dir: Path  = Path("dataset") / "checkpoints_final"

    # Split config
    seed: int = 42
    use_fixed_split: bool = False
    train_frac: float = 0.75
    dev_frac: float = 0.10
    test_frac: float = 0.15
    train_users: List[str] = None  # e.g., ["u1", ...]
    dev_users:   List[str] = None
    test_users:  List[str] = None

    # Training knobs
    epochs: int = 8
    batch_train: int = 128
    batch_eval: int = 256
    num_workers: int = 2
    persistent: bool = False
    prefetch: int = 2
    max_windows: int | None = None

    # Fusion / model knobs
    fusion: str = "concat"  # "concat" (1024) or "sum" (512)
    hidden: int = 256
    num_layers: int = 2
    d_out: int = 512
    dropout_head: float = 0.10

    # Loss / imbalance / robustness
    use_focal: bool = False
    focal_gamma: float = 1.5
    pos_weight_max: float = 20.0
    modality_dropout_p: float = 0.15

    # Unified threshold policy
    thr_policy: str = "precision_at_least"   # or "max_f1"
    thr_p_target: float = 0.75

    # Exports
    save_v_state: bool = True
    save_per_sample_pred: bool = True

    def finalize_paths(self):
        # Ensure derived paths reflect data_root if custom provided
        if self.v3_dir.is_relative_to(Path("dataset")) and self.data_root != Path("dataset"):
            self.v3_dir = self.data_root / self.v3_dir.name
        if self.imu_dir.is_relative_to(Path("dataset")) and self.data_root != Path("dataset"):
            self.imu_dir = self.data_root / self.imu_dir.name
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

# ============================== Utilities ================================

def set_seed(s: int = 42) -> None:
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def uuid_of(p: Path) -> str:
    return p.name.split("_")[0]


# --------------------------- Memmap helpers ------------------------------

def _mm_paths(base_dir: Path, tag: str):
    # Put each user in a short hashed subfolder; simple filenames inside
    base = base_dir / _slugify_tag(tag)
    return {
        "E_imu": base / "E_imu.npy",
        "X_mfcc": base / "X_mfcc.npy",
        "y": base / "y.npy",
        # intentionally no 'meta' (object dtype not memmap-safe)
    }

def _safe_save(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(_win_long(path), arr)
def _ensure_memmap_from_npz(npz_path: Path, cache_dir: Path, tag: str):
    paths = _mm_paths(cache_dir, tag)
    need_build = any(not p.exists() for p in paths.values())
    if need_build:
        with np.load(npz_path, allow_pickle=True) as z:
            if "E_imu" in z:
                _safe_save(paths["E_imu"], z["E_imu"].astype(np.float32, copy=False))
            if "y" in z:
                _safe_save(paths["y"], z["y"].astype(np.float32, copy=False))
            if "X_mfcc_fixed" in z:
                _safe_save(paths["X_mfcc"], z["X_mfcc_fixed"].astype(np.float32, copy=False))
    mm = {}
    for k, p in paths.items():
        if p.exists():
            try:
                mm[k] = np.load(_win_long(p), mmap_mode="r")
            except ValueError:
                mm[k] = np.load(_win_long(p), allow_pickle=True)
    return mm

# =============================== Dataset =================================
class OneUserDS(Dataset):
    """
    RAM-friendly per-user dataset using on-disk memmap.
    __getitem__ does: sanitize NaN/Inf + mild clipping + per-sample CMVN over time for MFCC.
    """
    def __init__(self, imu_npz: Path, v3_npz: Path, cache_dir: Path, tag: str, max_windows=None):
        self.tag = tag
        self.cache_dir = cache_dir
        self.mm_imu = _ensure_memmap_from_npz(imu_npz, cache_dir, tag + ".imu")
        self.mm_v3  = _ensure_memmap_from_npz(v3_npz,  cache_dir, tag + ".v3")

        req = ("E_imu" in self.mm_imu) and ("y" in self.mm_imu) and ("X_mfcc" in self.mm_v3)
        if not req:
            raise ValueError(f"Memmap missing arrays for tag={tag}. Check npz keys.")

        self.E_imu  = self.mm_imu["E_imu"]
        self.y      = self.mm_imu["y"]
        self.X_mfcc = self.mm_v3["X_mfcc"]
        self.meta   = self.mm_v3.get("meta", None)

        assert len(self.E_imu) == len(self.X_mfcc) == len(self.y), "Mismatched lengths"

        self._len = min(max_windows, len(self.y)) if max_windows is not None else len(self.y)

    def __len__(self):
        return self._len

    def __getitem__(self, i):
        e_imu  = np.asarray(self.E_imu[i], dtype=np.float32, order="C")  # [512]
        x_mfcc = np.asarray(self.X_mfcc[i], dtype=np.float32, order="C") # [430,13]
        y      = np.float32(self.y[i])

        # sanitize
        e_imu  = np.nan_to_num(e_imu,  nan=0.0, posinf=1e3, neginf=-1e3)
        x_mfcc = np.nan_to_num(x_mfcc, nan=0.0, posinf=1e2, neginf=-1e2)

        # per-sample over-time CMVN for MFCC
        m = x_mfcc.mean(axis=0, keepdims=True)
        s = x_mfcc.std(axis=0,  keepdims=True) + 1e-5
        x_mfcc = (x_mfcc - m) / s

        return (
            torch.from_numpy(e_imu).contiguous(),          # [512]
            torch.from_numpy(x_mfcc).contiguous(),         # [430,13]
            torch.tensor(y, dtype=torch.float32),
        )

# ============================= Data Helpers ===============================

def build_loaders(ds_tr, ds_dev, ds_te, batch_train, batch_eval, workers, persistent=False, prefetch=2):
    common = dict(pin_memory=True, persistent_workers=(persistent and workers>0))
    if workers > 0 and prefetch:
        common["prefetch_factor"] = prefetch
    dl_tr  = DataLoader(ds_tr,  batch_size=batch_train, shuffle=True,  num_workers=workers, **common)
    dl_dev = DataLoader(ds_dev, batch_size=batch_eval,  shuffle=False, num_workers=workers, **common)
    dl_te  = DataLoader(ds_te,  batch_size=batch_eval,  shuffle=False, num_workers=workers, **common)
    return dl_tr, dl_dev, dl_te


def compute_pos_weight_from_datasets(ds_list) -> float:
    pos = 0.0; tot = 0
    for ds in ds_list:
        pos += float(np.sum(ds.y[:ds._len]))
        tot += ds._len
    pos = max(int(pos), 1)
    return float((tot - pos) / pos)

# ================================ Model ==================================
class MFCCEncoder(nn.Module):
    def __init__(self, out_dim=512, in_dim=13, hidden=256, num_layers=2, bidir=True):
        super().__init__()
        self.rnn = nn.GRU(in_dim, hidden, num_layers=num_layers, batch_first=True, bidirectional=bidir)
        feat = hidden * (2 if bidir else 1)
        self.proj = nn.Linear(feat, out_dim)
    def forward(self, x):             # x: [B,430,13]
        x = x.float()                 # force FP32 for GRU stability
        with torch.autocast(device_type="cuda", enabled=False):
            h, _ = self.rnn(x)
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
        e_a   = self.mfcc(mfcc)                      # [B,512]
        if self.fusion == "concat":
            V = torch.cat([self.ln_i(e_imu), self.ln_a(e_a)], dim=-1)  # [B,1024]
        else:  # sum fusion
            a = nn.functional.normalize(self.ln_i(e_imu), dim=-1)
            b = nn.functional.normalize(self.ln_a(e_a),   dim=-1)
            V = nn.functional.normalize(a + b, dim=-1)               # [B,512]
        logit = self.head(V).squeeze(-1)
        return logit, V

_PRINTED_COMPILE = False

def maybe_compile(model: nn.Module):
    global _PRINTED_COMPILE
    if (IS_WSL or (IS_LINUX and not IS_WINDOWS)) and torch.cuda.is_available():
        try:
            model = torch.compile(model, mode="max-autotune")
            if not _PRINTED_COMPILE:
                print("[compile] torch.compile enabled (inductor).")
                _PRINTED_COMPILE = True
        except Exception as e:
            if not _PRINTED_COMPILE:
                print("[warn] compile disabled:", repr(e))
                _PRINTED_COMPILE = True
    else:
        if not _PRINTED_COMPILE:
            print("[compile] disabled (Windows or no CUDA).")
            _PRINTED_COMPILE = True
    return model

# =========================== Loss & Metrics ===============================
class BCEFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        p = torch.sigmoid(logits).detach()
        mod = (1 - p)**self.gamma
        return (mod * bce).mean()


def compute_metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    try: aupr = float(average_precision_score(y, p))
    except: aupr = float("nan")
    try: auc = float(roc_auc_score(y, p))
    except: auc = float("nan")
    pr, rc, thr = precision_recall_curve(y, p)
    f1 = 2*pr*rc/(pr+rc+1e-12)
    best_f1 = float(np.nanmax(f1)) if len(f1) else float("nan")
    return dict(aupr=aupr, auc=auc, best_f1=best_f1)


def select_threshold(y, p, policy="precision_at_least", p_target=0.75) -> float:
    pr, rc, thr = precision_recall_curve(y, p)
    if policy == "max_f1":
        f1 = 2*pr*rc/(pr+rc+1e-12)
        i = int(np.nanargmax(f1)) if len(f1) else 0
        return float(thr[max(i-1, 0)]) if len(thr) else 0.5
    mask = pr >= p_target
    if mask.any():
        idxs = np.where(mask)[0]
        best = idxs[np.argmax(rc[idxs])]
        return float(thr[max(min(best, len(thr)-1), 0)])
    # fallback
    f1 = 2*pr*rc/(pr+rc+1e-12)
    i = int(np.nanargmax(f1)) if len(f1) else 0
    return float(thr[max(i-1, 0)]) if len(thr) else 0.5

@torch.inference_mode()
def eval_collect(model, loader):
    model.eval()
    ys, ps = [], []
    for e_imu, mfcc, y in loader:
        e_imu = e_imu.to(DEVICE, non_blocking=True)
        mfcc  = mfcc.to(DEVICE, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=False):
            logit,_ = model(e_imu, mfcc)
            p = torch.sigmoid(logit)
            p = torch.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0).float().cpu().numpy()
        ys.append(y.numpy()); ps.append(p)
    y = np.concatenate(ys); p = np.concatenate(ps)
    if not np.isfinite(p).all():
        p = np.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0)
    return y, p

# ============================== Split Helpers ============================

def list_users(v3_dir: Path, imu_dir: Path) -> List[str]:
    v3_map  = {uuid_of(p): p for p in v3_dir.glob("*.npz")}
    imu_map = {uuid_of(p): p for p in imu_dir.glob("*_imu512.npz")}
    uuids   = sorted(set(v3_map) & set(imu_map))
    return uuids


def build_maps(v3_dir: Path, imu_dir: Path) -> Tuple[Dict[str,Path], Dict[str,Path]]:
    v3_map  = {uuid_of(p): p for p in v3_dir.glob("*.npz")}
    imu_map = {uuid_of(p): p for p in imu_dir.glob("*_imu512.npz")}
    return v3_map, imu_map


def split_users_random(uuids: List[str], train_frac=0.75, dev_frac=0.10, seed=42):
    rng = np.random.default_rng(seed)
    uu = np.array(uuids)
    rng.shuffle(uu)
    n = len(uu)
    n_tr = int(n * train_frac)
    n_dev = int(n * dev_frac)
    train = uu[:n_tr].tolist()
    dev   = uu[n_tr:n_tr+n_dev].tolist()
    test  = uu[n_tr+n_dev:].tolist()
    return train, dev, test

# =========================== Training Pipeline ===========================

def train_single_model(cfg: Config):
    # Paths setup & checks
    cfg.finalize_paths()
    for p in [cfg.v3_dir, cfg.imu_dir]:
        if not p.exists():
            raise FileNotFoundError(f"Data folder not found: {p.resolve()}")

    set_seed(cfg.seed)

    v3_map, imu_map = build_maps(cfg.v3_dir, cfg.imu_dir)
    uuids = sorted(set(v3_map) & set(imu_map))
    print("Available users:", len(uuids))

    if cfg.use_fixed_split and (cfg.train_users and cfg.dev_users and cfg.test_users):
        train_users, dev_users, test_users = cfg.train_users, cfg.dev_users, cfg.test_users
    else:
        train_users, dev_users, test_users = split_users_random(
            uuids, cfg.train_frac, cfg.dev_frac, seed=cfg.seed
        )
    print(f"Split → Train:{len(train_users)}  Dev:{len(dev_users)}  Test:{len(test_users)}")

    # Datasets
    cache_dir = Path("D://Download/memmap_cache_final")

    cache_dir.mkdir(parents=True, exist_ok=True)

    ds_tr_all  = [OneUserDS(imu_map[u], v3_map[u], cache_dir, tag=u, max_windows=cfg.max_windows) for u in train_users]
    ds_dev_all = [OneUserDS(imu_map[u], v3_map[u], cache_dir, tag=u, max_windows=cfg.max_windows) for u in dev_users]
    ds_te_all  = [OneUserDS(imu_map[u], v3_map[u], cache_dir, tag=u, max_windows=cfg.max_windows) for u in test_users]

    train_ds = ConcatDataset(ds_tr_all)
    dev_ds   = ConcatDataset(ds_dev_all)
    test_ds  = ConcatDataset(ds_te_all)

    dl_tr, dl_dev, dl_te = build_loaders(
        train_ds, dev_ds, test_ds,
        cfg.batch_train, cfg.batch_eval,
        cfg.num_workers, cfg.persistent, cfg.prefetch,
    )
    writer = SummaryWriter(log_dir=str(cfg.out_dir / "runs"))

    # Loss
    pos_w = compute_pos_weight_from_datasets(ds_tr_all)
    pos_w = float(min(max(pos_w, 1.0), cfg.pos_weight_max))
    if cfg.use_focal:
        crit = BCEFocalLoss(gamma=cfg.focal_gamma, pos_weight=torch.tensor([pos_w], device=DEVICE))
    else:
        crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=DEVICE))

    # Model & opt
    model = ConcatClassifier(d=cfg.d_out, hidden=cfg.hidden, num_layers=cfg.num_layers, fusion=cfg.fusion, dropout=cfg.dropout_head).to(DEVICE)
    model = maybe_compile(model)
    opt   = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4, fused=torch.cuda.is_available())

    scaler = torch.amp.GradScaler(enabled=USE_AMP and (AMP_DTYPE==torch.float16))
    best = {"aupr": -1.0, "state": None}
    patience, best_ep, PATIENCE = 3, -1, 3

    for ep in range(1, cfg.epochs+1):
        model.train(); run_loss = 0.0; n_seen = 0
        for i, (e_imu, mfcc, y) in enumerate(dl_tr):
            e_imu = e_imu.to(DEVICE, non_blocking=True)
            mfcc  = mfcc.to(DEVICE, non_blocking=True)
            y     = y.to(DEVICE, non_blocking=True)

            # modality dropout (MFCC)
            if cfg.modality_dropout_p > 0:
                drop = (torch.rand(mfcc.size(0), device=mfcc.device) < cfg.modality_dropout_p).view(-1,1,1)
                mfcc = torch.where(drop, torch.zeros_like(mfcc), mfcc)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=USE_AMP and torch.cuda.is_available()):
                logit, _ = model(e_imu, mfcc)
                loss = crit(logit, y)

            if not torch.isfinite(loss):
                print("[warn] NaN/Inf loss — skip batch")
                for g in opt.param_groups: g["lr"] *= 0.5
                continue

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()

            bs = y.size(0); run_loss += float(loss.item()) * bs; n_seen += bs
            global_step = (ep-1)*len(dl_tr) + i  # where i is batch index
            writer.add_scalar("train/loss", float(loss.item()), global_step)
        writer.add_scalar("epoch/train_loss", run_loss/max(n_seen,1), ep)

        # Dev evaluation
        y_dev, p_dev = eval_collect(model, dl_dev)
        dev_stats = compute_metrics(y_dev, p_dev)
        writer.add_scalar("dev/aupr", dev_stats["aupr"], ep)
        writer.add_scalar("dev/auc",  dev_stats["auc"], ep)
        writer.add_scalar("dev/best_f1", dev_stats["best_f1"], ep)

        print(f"[Ep{ep:02d}] loss={run_loss/max(n_seen,1):.4f}  dev-AUPR={dev_stats['aupr']:.3f}  AUC={dev_stats['auc']:.3f}")
        
        if dev_stats["aupr"] > best["aupr"]:
            best["aupr"]  = dev_stats["aupr"]
            best["state"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_ep = ep; patience = PATIENCE
        else:
            patience -= 1
            if patience <= 0:
                print(f"Early stopping at epoch {ep} (best at {best_ep})")
                break
       


    # Load best
    if best["state"] is not None:
        model.load_state_dict({k: v.to(DEVICE) for k,v in best["state"].items()})

    # Select unified threshold on Dev
    y_dev, p_dev = eval_collect(model, dl_dev)
    thr = select_threshold(y_dev, p_dev, policy=cfg.thr_policy, p_target=cfg.thr_p_target)
    print("Unified threshold (Dev):", thr)

    # Evaluate on Test (per-user & pooled)
    per_user_rows = []
    pooled_y, pooled_p = [], []

    # helper to make loader for single user's dataset
    def make_user_loader(uids: List[str]):
        return [
            (u, DataLoader(OneUserDS(imu_map[u], v3_map[u], cache_dir, tag=u, max_windows=cfg.max_windows),
                           batch_size=cfg.batch_eval, shuffle=False, num_workers=cfg.num_workers,
                           pin_memory=True, persistent_workers=False,
            )) for u in uids
        ]

    test_loaders = make_user_loader(test_users)

    for u, ld in test_loaders:
        y_u, p_u = eval_collect(model, ld)
        pooled_y.append(y_u); pooled_p.append(p_u)
        # metrics
        stats = compute_metrics(y_u, p_u)
        pred = (p_u >= thr).astype(np.int32)
        tp = int(((pred==1) & (y_u==1)).sum()); fp = int(((pred==1) & (y_u==0)).sum()); fn = int(((pred==0) & (y_u==1)).sum())
        prec = tp / max(tp+fp, 1); rec = tp / max(tp+fn, 1); cov = float(pred.mean())
        per_user_rows.append(dict(uuid=u, **stats, P_at_thr=prec, R_at_thr=rec, coverage=cov, n=len(y_u), pos_rate=float(y_u.mean())))

        # optional: export V_state for this user's test set
        if cfg.save_v_state:
            Vs, Ys = [], []
            for e_imu, mfcc, y in ld:
                e_imu = e_imu.to(DEVICE, non_blocking=True)
                mfcc  = mfcc.to(DEVICE, non_blocking=True)
                with torch.autocast(device_type="cuda", enabled=False):
                    _, V = model(e_imu, mfcc)
                Vs.append(V.detach().cpu().numpy()); Ys.append(y.numpy())
            V_all = np.concatenate(Vs); y_all = np.concatenate(Ys)
            # align meta length for this user
            with np.load(v3_map[u], allow_pickle=True) as z:
                meta_full = z["meta"]
            meta = meta_full[:len(y_all)]
            np.savez_compressed(
                cfg.ckpt_dir / f"V_state_test_{u}.npz",
                V=V_all.astype(np.float32), y=y_all.astype(np.int64), meta=meta,
                fusion=cfg.fusion, dim=int(V_all.shape[1])
            )

        if cfg.save_per_sample_pred:
            df_u = pd.DataFrame({"uuid": u, "y": y_u.astype(int), "p": p_u})
            df_u.to_csv(cfg.out_dir / f"pred_test_{u}.csv", index=False)

    # Pooled test metrics
    y_all = np.concatenate(pooled_y); p_all = np.concatenate(pooled_p)
    pooled_stats = compute_metrics(y_all, p_all)
    pred_all = (p_all >= thr).astype(np.int32)
    tp = int(((pred_all==1) & (y_all==1)).sum()); fp = int(((pred_all==1) & (y_all==0)).sum()); fn = int(((pred_all==0) & (y_all==1)).sum())
    prec = tp / max(tp+fp, 1); rec = tp / max(tp+fn, 1); cov = float(pred_all.mean())
    writer.add_scalar("test/aupr", pooled_stats["aupr"], 0)
    writer.add_scalar("test/auc", pooled_stats["auc"], 0)
    writer.add_scalar("test/best_f1", pooled_stats["best_f1"], 0)

    # Save metrics
    df_users = pd.DataFrame(per_user_rows).sort_values("uuid")
    df_users.to_csv(cfg.out_dir / "final_test_metrics.csv", index=False)
    with open(cfg.out_dir / "final_test_metrics_summary.json", "w") as f:
        json.dump({
            "pooled": {**pooled_stats, "P_at_thr": prec, "R_at_thr": rec, "coverage": cov, "n": int(len(y_all)), "pos_rate": float(y_all.mean())},
            "threshold": {"policy": cfg.thr_policy, "p_target": cfg.thr_p_target, "value": float(thr)},
            "split": {"train": train_users, "dev": dev_users, "test": test_users},
            "config": {
                "fusion": cfg.fusion, "hidden": cfg.hidden, "num_layers": cfg.num_layers,
                "dropout_head": cfg.dropout_head, "modality_dropout": cfg.modality_dropout_p,
                "use_focal": cfg.use_focal, "focal_gamma": cfg.focal_gamma,
                "epochs": cfg.epochs, "batch_train": cfg.batch_train, "batch_eval": cfg.batch_eval,
                "max_windows": cfg.max_windows,
            }
        }, f, indent=2)

    print("\n== Final pooled Test ==")
    print({**pooled_stats, "P_at_thr": prec, "R_at_thr": rec, "coverage": cov, "n": int(len(y_all))})

    # Save final model & threshold
    torch.save({
        "state_dict": {k: v.detach().cpu() for k,v in model.state_dict().items()},
        "arch": f"ConcatClassifier(d={cfg.d_out}, hidden={cfg.hidden}, num_layers={cfg.num_layers}, fusion='{cfg.fusion}')",
        "normalization": "IMU=E_imu cached; MFCC per-sample CMVN over time",
        "uuids_train": train_users,
        "uuids_dev": dev_users,
        "uuids_test": test_users,
    }, cfg.ckpt_dir / "C_social_final.pt")

    with open(cfg.ckpt_dir / "threshold_final.json", "w") as f:
        json.dump({"policy": cfg.thr_policy, "p_target": cfg.thr_p_target, "threshold": float(thr)}, f, indent=2)
    print("Saved:", cfg.ckpt_dir / "C_social_final.pt", "and", cfg.ckpt_dir / "threshold_final.json")
    writer.close()

# ================================ CLI ====================================

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train C_social model (IMU + MFCC)")
    # Paths
    p.add_argument("--data_root", type=Path, default=Config.data_root)
    p.add_argument("--v3_dir", type=Path, default=Config.v3_dir)
    p.add_argument("--imu_dir", type=Path, default=Config.imu_dir)
    p.add_argument("--out_dir", type=Path, default=Config.out_dir)
    p.add_argument("--ckpt_dir", type=Path, default=Config.ckpt_dir)

    # Split
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--use_fixed_split", action="store_true")
    p.add_argument("--train_frac", type=float, default=Config.train_frac)
    p.add_argument("--dev_frac", type=float, default=Config.dev_frac)

    # Train
    p.add_argument("--epochs", type=int, default=Config.epochs)
    p.add_argument("--batch_train", type=int, default=Config.batch_train)
    p.add_argument("--batch_eval", type=int, default=Config.batch_eval)
    p.add_argument("--num_workers", type=int, default=Config.num_workers)
    p.add_argument("--persistent", action="store_true")
    p.add_argument("--prefetch", type=int, default=Config.prefetch)
    p.add_argument("--max_windows", type=int, default=None)

    # Model
    p.add_argument("--fusion", choices=["concat","sum"], default=Config.fusion)
    p.add_argument("--hidden", type=int, default=Config.hidden)
    p.add_argument("--num_layers", type=int, default=Config.num_layers)
    p.add_argument("--d_out", type=int, default=Config.d_out)
    p.add_argument("--dropout_head", type=float, default=Config.dropout_head)

    # Loss / robustness
    p.add_argument("--use_focal", action="store_true")
    p.add_argument("--focal_gamma", type=float, default=Config.focal_gamma)
    p.add_argument("--pos_weight_max", type=float, default=Config.pos_weight_max)
    p.add_argument("--modality_dropout_p", type=float, default=Config.modality_dropout_p)

    # Threshold
    p.add_argument("--thr_policy", choices=["precision_at_least","max_f1"], default=Config.thr_policy)
    p.add_argument("--thr_p_target", type=float, default=Config.thr_p_target)

    # Exports
    p.add_argument("--no_save_v_state", action="store_true")
    p.add_argument("--no_save_per_sample_pred", action="store_true")

    return p


def cfg_from_args(args: argparse.Namespace) -> Config:
    cfg = Config(
        data_root=args.data_root,
        v3_dir=args.v3_dir,
        imu_dir=args.imu_dir,
        out_dir=args.out_dir,
        ckpt_dir=args.ckpt_dir,
        seed=args.seed,
        use_fixed_split=args.use_fixed_split,
        train_frac=args.train_frac,
        dev_frac=args.dev_frac,
        test_frac=1.0 - args.train_frac - args.dev_frac if (args.train_frac + args.dev_frac) < 1.0 else 0.15,
        train_users=None, dev_users=None, test_users=None,
        epochs=args.epochs,
        batch_train=args.batch_train,
        batch_eval=args.batch_eval,
        num_workers=args.num_workers,
        persistent=args.persistent,
        prefetch=args.prefetch,
        max_windows=args.max_windows,
        fusion=args.fusion,
        hidden=args.hidden,
        num_layers=args.num_layers,
        d_out=args.d_out,
        dropout_head=args.dropout_head,
        use_focal=args.use_focal,
        focal_gamma=args.focal_gamma,
        pos_weight_max=args.pos_weight_max,
        modality_dropout_p=args.modality_dropout_p,
        thr_policy=args.thr_policy,
        thr_p_target=args.thr_p_target,
        save_v_state=not args.no_save_v_state,
        save_per_sample_pred=not args.no_save_per_sample_pred,
    )
    return cfg

# ================================= Main ==================================
if __name__ == "__main__":
    # Windows needs spawn; WSL/Linux is fine too.
    try:
        import multiprocessing as mp
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    if IS_LINUX:
        try:
            torch.multiprocessing.set_sharing_strategy("file_system")
        except Exception:
            pass

    parser = build_argparser()
    args = parser.parse_args() if len(sys.argv) > 1 else parser.parse_args("")
    cfg = cfg_from_args(args)

    train_single_model(cfg)

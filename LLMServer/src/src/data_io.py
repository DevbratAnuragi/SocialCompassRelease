from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split

# ---------- small utils ----------
def uuid_of(p: Path) -> str:
    """Extract UUID from a file name like '<UUID>_xxxx.npz'"""
    return p.name.split("_")[0]

def detect_ts_unit(timestamps: np.ndarray) -> str:
    """Heuristic: detect seconds vs milliseconds."""
    ts = np.asarray(timestamps).astype(float)
    ts = ts[np.isfinite(ts)]
    if ts.size == 0: return "unknown"
    # 2000-01-01 in seconds ≈ 946684800; in ms ≈ 946684800000
    m = np.median(ts)
    return "ms" if m > 1e11 else "s"

def inspect_npz(path: Path) -> Dict[str, Tuple[Tuple[int, ...], str]]:
    """Return {key: (shape, dtype)} without loading arrays into RAM."""
    out = {}
    with np.load(path, allow_pickle=True) as z:
        for k in z.files:
            arr = z[k]
            out[k] = (tuple(arr.shape), str(arr.dtype))
    return out

# ---------- memmap cache ----------
def _mm_paths(base_dir: Path, tag: str) -> Dict[str, Path]:
    base = base_dir / tag
    return {
        "E_imu":  base.with_suffix(".E_imu.npy"),
        "X_mfcc": base.with_suffix(".X_mfcc.npy"),
        "y":      base.with_suffix(".y.npy"),
    }

def ensure_memmap_from_npz(npz_path: Path, cache_dir: Path, tag: str) -> Dict[str, np.memmap]:
    """
    First call: split .npz into .npy files (memmap-friendly).
    Expected keys (any subset ok):
      - IMU cache:   E_imu [N,512], y [N]
      - v3 feature:  X_mfcc_fixed [N,430,13]  -> stored as X_mfcc.npy
    Returns dict of np.memmap arrays with mmap_mode='r'.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = _mm_paths(cache_dir, tag)
    need_build = any(not p.exists() for p in paths.values())
    if need_build:
        with np.load(npz_path, allow_pickle=True) as z:
            if "E_imu" in z:
                np.save(paths["E_imu"], z["E_imu"].astype(np.float32, copy=False))
            if "y" in z:
                np.save(paths["y"], z["y"].astype(np.float32, copy=False))
            if "X_mfcc_fixed" in z:
                np.save(paths["X_mfcc"], z["X_mfcc_fixed"].astype(np.float32, copy=False))
            elif "X_mfcc" in z:
                np.save(paths["X_mfcc"], z["X_mfcc"].astype(np.float32, copy=False))

    mm = {}
    for k, p in paths.items():
        if p.exists():
            mm[k] = np.load(p, mmap_mode="r")
    return mm

# ---------- sanitizers ----------
def sanitize_eimu(e: np.ndarray) -> np.ndarray:
    e = np.nan_to_num(e, nan=0.0, posinf=1e3, neginf=-1e3)
    return e

def sanitize_mfcc(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x, nan=0.0, posinf=1e2, neginf=-1e2)
    # Per-sample CMVN over time axis (430x13 -> normalize over time)
    m = x.mean(axis=0, keepdims=True)
    s = x.std(axis=0,  keepdims=True) + 1e-5
    return (x - m) / s

# ---------- Dataset ----------
class OneUserDS(Dataset):
    """
    RAM-friendly dataset backed by on-disk memmaps.
    __getitem__ performs: sanitize + CMVN (MFCC).
    Returns: (e_imu[512], mfcc[430,13], y)
    """
    def __init__(
        self,
        imu_npz: Path,
        v3_npz: Path,
        cache_dir: Path,
        tag: str,
        max_windows: Optional[int] = None,
    ):
        self.tag = tag
        imu_mm = ensure_memmap_from_npz(imu_npz, cache_dir, tag + ".imu")
        v3_mm  = ensure_memmap_from_npz(v3_npz,  cache_dir, tag + ".v3")
        if "E_imu" not in imu_mm or "y" not in imu_mm or "X_mfcc" not in v3_mm:
            raise ValueError(f"Memmap missing arrays for tag={tag}: found {list(imu_mm.keys())} {list(v3_mm.keys())}")

        self.E_imu  = imu_mm["E_imu"]
        self.y      = imu_mm["y"]
        self.X_mfcc = v3_mm["X_mfcc"]

        assert len(self.E_imu) == len(self.X_mfcc) == len(self.y), "length mismatch"
        self._len = min(max_windows, len(self.y)) if max_windows is not None else len(self.y)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, i: int):
        e_imu  = np.asarray(self.E_imu[i], dtype=np.float32, order="C")
        x_mfcc = np.asarray(self.X_mfcc[i], dtype=np.float32, order="C")
        y      = np.float32(self.y[i])

        e_imu  = sanitize_eimu(e_imu)
        x_mfcc = sanitize_mfcc(x_mfcc)

        return (
            torch.from_numpy(e_imu).contiguous(),     # [512]
            torch.from_numpy(x_mfcc).contiguous(),    # [430,13]
            torch.tensor(y, dtype=torch.float32),
        )

# ---------- builders ----------
def build_per_user_datasets(
    uuids: List[str],
    v3_map: Dict[str, Path],
    imu_map: Dict[str, Path],
    cache_dir: Path,
    max_windows: Optional[int] = None,
) -> Dict[str, OneUserDS]:
    return {
        u: OneUserDS(imu_map[u], v3_map[u], cache_dir=cache_dir, tag=u, max_windows=max_windows)
        for u in uuids
    }

def build_loaders(
    ds_tr, ds_val, ds_te,
    batch_train: int, batch_eval: int,
    workers: int = 2, persistent: bool = False, prefetch: int = 2
):
    common = dict(pin_memory=True, persistent_workers=(persistent and workers > 0))
    if workers > 0 and prefetch:
        common["prefetch_factor"] = prefetch
    dl_tr  = DataLoader(ds_tr,  batch_size=batch_train, shuffle=True,  num_workers=workers, **common)
    dl_val = DataLoader(ds_val, batch_size=batch_eval,  shuffle=False, num_workers=workers, **common)
    dl_te  = DataLoader(ds_te,  batch_size=batch_eval,  shuffle=False, num_workers=workers, **common)
    return dl_tr, dl_val, dl_te

def compute_pos_weight_from_datasets(ds_list: List[OneUserDS]) -> float:
    pos = 0.0; tot = 0
    for ds in ds_list:
        pos += float(np.sum(ds.y[:ds._len]))
        tot += ds._len
    pos = max(int(pos), 1)
    return (tot - pos) / pos

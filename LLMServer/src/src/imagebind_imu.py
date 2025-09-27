from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import torch
import torch.nn.functional as F

# --- positional embedding resize ---
@torch.no_grad()
def resize_pos_embed_(imu_preprocessor, new_T: int, kernel_size: int = 8):
    """
    Resize IMU positional embeddings from 5s/2k → 20s/800 (token L: 251 → 101).
    """
    assert new_T % kernel_size == 0
    L_new = new_T // kernel_size
    pe = imu_preprocessor.pos_embed                   # [1, cls+L_old, D]
    num_cls = imu_preprocessor.num_cls_tokens
    cls = pe[:, :num_cls, :]
    pos = pe[:, num_cls:, :].transpose(1, 2)         # [1, D, L_old]
    pos = F.interpolate(pos, size=L_new, mode="linear", align_corners=False)
    pos = pos.transpose(1, 2)                        # [1, L_new, D]
    imu_preprocessor.pos_embed = torch.nn.Parameter(torch.cat([cls, pos], dim=1))

# --- build imagebind imu ---
def build_imagebind_imu(model_name: str = "imagebind_huge", pretrained: bool = True, device: Optional[torch.device] = None):
    """
    Returns: (model, imu_pre, imu_trunk, D)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        from imagebind.models import imagebind_model
    except Exception as e:
        raise ImportError("imagebind is not installed. pip install imagebind") from e

    if not hasattr(imagebind_model, model_name):
        raise ValueError(f"Unknown ImageBind model: {model_name}")

    model = getattr(imagebind_model, model_name)(pretrained=pretrained)
    imu_pre   = model.modality_preprocessors["imu"].to(device).eval()
    imu_trunk = model.modality_trunks["imu"].to(device).eval()

    # infer D
    with torch.no_grad():
        dummy = torch.randn(2, 6, 800, device=device)
        tokens = imu_pre(dummy)["trunk"]["tokens"]  # [2, L, D]
        out = imu_trunk(tokens)
        if isinstance(out, dict):
            for v in out.values():
                if torch.is_tensor(v): out = v; break
        D = int(out.shape[-1])
    return model, imu_pre, imu_trunk, D

# --- encode imu batch ---
@torch.no_grad()
def encode_imu_batch(x_imu6_np: np.ndarray, imu_pre, imu_trunk, device: Optional[torch.device] = None) -> np.ndarray:
    """
    x_imu6_np: [B,800,6] float32 → returns [B, D] float32 (CLS).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.from_numpy(x_imu6_np).to(device).float().permute(0, 2, 1)  # [B,6,800]
    tokens = imu_pre(x)["trunk"]["tokens"]                               # [B,L,D]
    out = imu_trunk(tokens)
    if isinstance(out, dict):
        for v in out.values():
            if torch.is_tensor(v): out = v; break
    e = out[:, 0]                                                        # CLS
    return e.detach().cpu().numpy()

# --- step A: precompute imu embeddings per UUID ---
def precompute_imu_embeddings(
    v3_dir: Path,
    out_dir: Path,
    model_name: str = "imagebind_huge",
    pretrained: bool = True,
    T_INPUT: int = 800,
    KERNEL_SIZE: int = 8,
    batch_size: int = 128,
    limit_files: Optional[int] = None,
):
    """
    For each '<UUID>_windows20s.npz' in v3_dir, compute E_imu (512) and save:
      out_dir / '<UUID>_imu512.npz' with keys: E_imu [N,512], y [N], meta [N,2]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(v3_dir.glob("*.npz"))
    if limit_files: files = files[:limit_files]

    model, imu_pre, imu_trunk, D = build_imagebind_imu(model_name=model_name, pretrained=pretrained, device=device)
    with torch.no_grad():
        resize_pos_embed_(imu_pre, new_T=T_INPUT, kernel_size=KERNEL_SIZE)

    for p in files:
        uuid = p.name.split("_")[0]
        out_p = out_dir / f"{uuid}_imu512.npz"
        if out_p.exists():
            print(f"[skip] {out_p.name} exists")
            continue
        z = np.load(p, allow_pickle=True)
        X_imu6 = z["X_imu6"].astype(np.float32)      # [N,800,6]
        y      = z["y"].astype(np.float32)
        meta   = z["meta"]

        N = len(y)
        e_list: List[np.ndarray] = []
        for i in range(0, N, batch_size):
            e_list.append(encode_imu_batch(X_imu6[i:i+batch_size], imu_pre, imu_trunk, device=device))
        E_imu = np.concatenate(e_list, axis=0)       # [N, D]
        assert E_imu.shape[0] == N

        np.savez_compressed(out_p, E_imu=E_imu.astype(np.float32), y=y, meta=meta, source=str(p))
        print(f"[ok] {uuid}: N={N} -> {out_p.name} | D={D}")

# src/socialcompass/arousal.py
# -*- coding: utf-8 -*-
"""
Arousal scoring and calibration utilities.

What this file does
-------------------
1) Build *text anchors* for arousal using ImageBind's **text** encoder:
   - Read high/low arousal prompts from JSON
   - Encode with ImageBind → average each group → L2-normalize
   - Save anchors to a .pt file

2) Score arousal for IMU embeddings already in ImageBind's **shared space**:
   - s = cos(e, high_anchor) - cos(e, low_anchor)

3) Calibrate a *single* arousal threshold on Dev users:
   - Policy: choose a threshold so the **mean triggers per social hour** lies in [2, 5]
   - Exactly matches the paper-style “rate-in-social-hours” calibration

Notes
-----
- This module **does not** compute IMU embeddings; pass in e_imu_shared from your encoder.
- We never use raw audio here; arousal is IMU-only.
- Timestamps should be in **seconds**; if you keep ms/us elsewhere, convert before calling.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Dict, Any
import json
import numpy as np
import torch


# ----------------------------- Small helpers -----------------------------

def _l2norm(t: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return torch.nn.functional.normalize(t, dim=dim, eps=eps)


def _pick_first_tensor(d: Dict[str, Any]) -> torch.Tensor:
    """ImageBind trunks/heads sometimes return dicts; pick the first tensor value."""
    for v in d.values():
        if torch.is_tensor(v):
            return v
    raise ValueError("No tensor found in dict output.")


# -------------------------- Text anchor building -------------------------

@torch.inference_mode()
def build_text_anchors(
    prompts_json: Path,
    out_pt: Path,
    device: str | torch.device = "cpu",
) -> None:
    """
    Encode high/low arousal prompts (JSON) with ImageBind text encoder and save anchors.

    JSON format:
    {
      "high": ["...", "...", ...],
      "low":  ["...", "...", ...]
    }
    """
    from imagebind.models import imagebind_model

    # Load prompts
    prompts_json = Path(prompts_json)
    with open(prompts_json, "r") as f:
        prm = json.load(f)
    high_list: List[str] = prm.get("high", [])
    low_list: List[str] = prm.get("low", [])
    assert len(high_list) > 0 and len(low_list) > 0, "prompts_json must have non-empty 'high' and 'low' lists."

    # Build ImageBind model (text only)
    ib = imagebind_model.imagebind_huge(pretrained=True).to(device).eval()
    text_pre = ib.modality_preprocessors["text"]
    text_tr  = ib.modality_trunks["text"]
    text_hd  = ib.modality_heads["text"]

    def _encode_text(lines: List[str]) -> torch.Tensor:
        tok = text_pre(lines)["trunk"]["tokens"]            # tokens
        x   = text_tr(tok)
        if isinstance(x, dict):
            x = _pick_first_tensor(x)
        if x.ndim == 3:                                     # [N, T, D] → mean pool T
            x = x.mean(dim=1)
        z   = text_hd(x)
        if isinstance(z, dict):
            z = _pick_first_tensor(z)
        return _l2norm(z, dim=-1)                           # [N, D_shared], L2

    e_high_all = _encode_text(high_list)                    # [H, D]
    e_low_all  = _encode_text(low_list)                     # [L, D]

    # Average each group → anchors
    e_high = _l2norm(e_high_all.mean(dim=0, keepdim=True))  # [1, D]
    e_low  = _l2norm(e_low_all.mean(dim=0,  keepdim=True))  # [1, D]

    out = {
        "e_high": e_high.cpu(),
        "e_low":  e_low.cpu(),
        "prompts": {"high": high_list, "low": low_list},
        "dim": int(e_high.shape[-1]),
        "note": "ImageBind text anchors for arousal (L2-normalized); s = cos(e, high) - cos(e, low).",
    }
    out_pt = Path(out_pt)
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_pt)


def load_text_anchors(anchor_pt: Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    """Load anchors saved by `build_text_anchors`."""
    state = torch.load(anchor_pt, map_location=map_location)
    # sanity
    if "e_high" not in state or "e_low" not in state:
        raise ValueError("anchor file missing e_high/e_low.")
    state["e_high"] = _l2norm(state["e_high"].float(), dim=-1)
    state["e_low"]  = _l2norm(state["e_low"].float(),  dim=-1)
    return state


# ------------------------------ Scoring ----------------------------------

@torch.inference_mode()
def score_arousal_with_anchors(
    e_imu_shared: torch.Tensor,   # [N, D_shared], L2-normalized
    e_high: torch.Tensor,         # [1, D_shared], L2-normalized
    e_low: torch.Tensor,          # [1, D_shared], L2-normalized
) -> torch.Tensor:
    """
    Compute arousal score s = cos(e, high) - cos(e, low).
    Inputs must already be L2-normalized (this function does NOT renormalize e_imu_shared).
    """
    # [N,1] each, broadcast matmul
    s_high = (e_imu_shared @ e_high.T)
    s_low  = (e_imu_shared @ e_low.T)
    return (s_high - s_low).squeeze(-1)                     # [N]


# --------------------------- Dev calibration -----------------------------

def _per_user_trigger_rate_per_social_hour(
    scores: np.ndarray,          # [N] arousal scores
    ts_sec: np.ndarray,          # [N] timestamps in SECONDS (int/float ok)
    social_mask: np.ndarray,     # [N] bool: within-social windows
    thr: float,
    hour_divisor: int = 3600,    # 1 hour in seconds
) -> float:
    """
    For one user, compute mean triggers/hour within social periods at threshold `thr`.

    Steps
    - Find all hours that contain any social windows (social hours).
    - Within those hours, count windows where score >= thr.
    - Return average trigger count per social hour (0 if no social hours).
    """
    assert len(scores) == len(ts_sec) == len(social_mask)
    m_soc = social_mask.astype(bool)
    if not m_soc.any():
        return 0.0

    hrs_social = np.unique((ts_sec[m_soc] // hour_divisor).astype(np.int64))
    if hrs_social.size == 0:
        return 0.0

    trig_mask = m_soc & (scores >= thr)
    if trig_mask.any():
        hrs_trig, cnts = np.unique((ts_sec[trig_mask] // hour_divisor).astype(np.int64), return_counts=True)
        count_map = {int(h): int(c) for h, c in zip(hrs_trig, cnts)}
        counts = [count_map.get(int(h), 0) for h in hrs_social]
    else:
        counts = [0 for _ in hrs_social]

    return float(np.mean(counts)) if len(counts) else 0.0


def calibrate_arousal_threshold(
    dev_scores: List[np.ndarray],         # list of [Ni] score arrays (Dev users)
    dev_ts_sec: List[np.ndarray],         # list of [Ni] timestamps in seconds
    dev_social_mask: List[np.ndarray],    # list of [Ni] bool masks (social)
    hour_divisors: List[int] | None = None,  # usually [3600]*len(Dev)
    target_min: float = 2.0,
    target_max: float = 5.0,
    q_grid: np.ndarray | None = None,
) -> Tuple[float, float]:
    """
    Choose a single threshold so that the **mean triggers per social hour** across Dev users
    lies within [target_min, target_max]. Returns (best_thr, achieved_mean_rate).

    Implementation
    - Pool scores over social windows to get candidate thresholds from quantiles.
    - For each candidate thr, compute per-user rate and average across users.
    - Pick thr that lands inside [min,max] and closest to the midpoint; if none inside,
      pick the closest with a penalty.
    """
    assert len(dev_scores) == len(dev_ts_sec) == len(dev_social_mask), "Dev lists must be aligned."
    n_users = len(dev_scores)
    if hour_divisors is None:
        hour_divisors = [3600] * n_users

    # Pool social-only scores to propose candidates
    pooled_social = np.concatenate(
        [s[m] for s, m in zip(dev_scores, dev_social_mask) if np.any(m)],
        dtype=np.float32
    ) if any(np.any(m) for m in dev_social_mask) else np.array([], dtype=np.float32)

    if pooled_social.size == 0:
        # No social windows present → return neutral threshold
        return 0.0, 0.0

    if q_grid is None:
        q_grid = np.concatenate([np.linspace(0.05, 0.95, 19), np.linspace(0.96, 0.995, 8)])
    candidates = np.unique(np.quantile(pooled_social, q_grid))

    target_mid = 0.5 * (target_min + target_max)
    best_thr, best_gap, best_rate = float(candidates[0]), 1e9, 0.0

    for thr in candidates:
        rates = []
        for s, ts, m, H in zip(dev_scores, dev_ts_sec, dev_social_mask, hour_divisors):
            rates.append(_per_user_trigger_rate_per_social_hour(s, ts, m, thr, hour_divisor=int(H)))
        if not rates:
            continue
        mean_rate = float(np.mean(rates))
        # Prefer thresholds inside the target interval; penalize outside
        in_range = (target_min <= mean_rate <= target_max)
        gap = abs(mean_rate - target_mid) if in_range else abs(mean_rate - target_mid) + 1.0
        if gap < best_gap:
            best_thr, best_gap, best_rate = float(thr), gap, mean_rate

    return best_thr, best_rate


# ------------------------------- Examples --------------------------------
if __name__ == "__main__":
    """
    Minimal smoke test (no real data):

    1) Build anchors (requires ImageBind installed):
       build_text_anchors("configs/prompts_arousal.json",
                          "datasets/checkpoints_final/arousal_text_anchors.pt")

    2) Load anchors, score dummy IMU embeddings, and calibrate:
       anchors = load_text_anchors("datasets/checkpoints_final/arousal_text_anchors.pt")
       e_high, e_low = anchors["e_high"], anchors["e_low"]  # [1,D]

       # Suppose you already have IMU shared embeddings e (L2-normalized) for Dev users:
       # dev_scores = [ score_arousal_with_anchors(e_u, e_high, e_low).cpu().numpy() for e_u in dev_embeddings ]
       # dev_ts_sec = [...]
       # dev_social_mask = [...]
       # thr, rate = calibrate_arousal_threshold(dev_scores, dev_ts_sec, dev_social_mask)

    This block is not meant to run end-to-end; it documents typical usage.
    """
    print("arousal.py: module loaded. See docstring and __main__ example for usage.")

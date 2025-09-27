# src/socialcompass/hase.py
# -*- coding: utf-8 -*-
"""
HASE (High Arousal Social Events) synthesis utilities.

What this file does
-------------------
Given per-window scores and timestamps, produce user-facing events:
1) **Masking (AND rule)**: HASE window = (P_social ≥ θ_social) AND (arousal ≥ θ_arousal)
2) **Motion gate (optional)**: filter out windows with very low IMU activity
3) **Debounce (merge)**: merge adjacent triggers that are ≤ N seconds apart into one event
4) **Export**: per-window table + per-event table (+ quick summary)

Inputs are simple NumPy arrays; this module does **not** load or save dataset files.
Timestamps should be in **seconds**. If your timestamps are ms/us elsewhere, convert first.

Typical usage
-------------
from socialcompass.hase import run_hase_for_arrays

df_win, df_evt = run_hase_for_arrays(
    ts_sec,             # [N] timestamps in seconds
    p_social,           # [N] probability from C_social
    s_arousal,          # [N] arousal score (IMU-only)
    theta_social,       # scalar threshold for C_social
    theta_arousal,      # scalar threshold for arousal
    X_imu6=X_imu6,      # optional [N,800,6] to enable motion gate
    motion_gate=True,
    motion_q=0.10,      # 10% quantile as "low motion" cutoff
    motion_margin=1.0,  # multiply cutoff by this factor
    debounce_sec=60.0,  # merge within 60 seconds
    window_len_sec=20.0 # 20-second windows
)

df_evt contains merged events with start/end and duration.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


# -------------------------------------------------------------------------
# Motion energy (for gating)
# -------------------------------------------------------------------------

def compute_motion_energy(X_imu6: np.ndarray) -> np.ndarray:
    """
    Per-window IMU energy proxy.

    Parameters
    ----------
    X_imu6 : np.ndarray
        Shape [N, 800, 6]. 20s @ 40Hz (or consistent) with 6 axes (acc+gyro).

    Returns
    -------
    np.ndarray
        energy: [N] float32. Larger means "more motion" in the 20s window.

    Method
    ------
    1) std over time per axis → [N, 6]
    2) L2 norm over axes → [N]
    """
    if X_imu6.ndim != 3 or X_imu6.shape[-1] != 6:
        raise ValueError("X_imu6 must be [N, T, 6].")
    std6 = np.std(X_imu6, axis=1)                 # [N,6]
    energy = np.linalg.norm(std6, axis=1)         # [N]
    return np.nan_to_num(energy, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


# -------------------------------------------------------------------------
# Debounce (merge adjacent triggers into events)
# -------------------------------------------------------------------------

def debounce_events(ts_sec: np.ndarray, mask: np.ndarray, gap_sec: float) -> np.ndarray:
    """
    Merge adjacent triggered windows if their timestamps are within `gap_sec`.

    Parameters
    ----------
    ts_sec : np.ndarray
        [N] timestamps in seconds (monotonic non-decreasing).
    mask : np.ndarray
        [N] bool mask of triggered windows.
    gap_sec : float
        Maximum allowed gap (in seconds) to consider two windows part of the same event.

    Returns
    -------
    np.ndarray
        event_id per window: [-1 for non-trigger, 0..E-1 for merged events]
    """
    if len(ts_sec) != len(mask):
        raise ValueError("ts_sec and mask must have the same length.")
    event_id = np.full(len(ts_sec), -1, dtype=np.int64)
    idx = np.where(mask)[0]
    if idx.size == 0:
        return event_id
    eid = 0
    start = prev = idx[0]
    for k in range(1, idx.size):
        if ts_sec[idx[k]] - ts_sec[prev] <= gap_sec:
            prev = idx[k]
        else:
            event_id[start:prev + 1] = eid
            eid += 1
            start = prev = idx[k]
    event_id[start:prev + 1] = eid
    return event_id


# -------------------------------------------------------------------------
# HASE window mask + per-window/per-event tables
# -------------------------------------------------------------------------

def make_hase_windows(
    ts_sec: np.ndarray,
    p_social: np.ndarray,
    s_arousal: np.ndarray,
    theta_social: float,
    theta_arousal: float,
    X_imu6: np.ndarray | None = None,
    motion_q: float = 0.10,
    motion_margin: float = 1.0,
) -> dict[str, np.ndarray]:
    """
    Compute HASE mask per window with optional motion gate.

    Parameters
    ----------
    ts_sec : np.ndarray
        [N] timestamps in seconds.
    p_social : np.ndarray
        [N] C_social probabilities.
    s_arousal : np.ndarray
        [N] arousal scores (IMU-only).
    theta_social : float
        Unified threshold for C_social (e.g., from threshold_final.json).
    theta_arousal : float
        Unified threshold for arousal (e.g., from arousal_threshold.json).
    X_imu6 : np.ndarray | None
        Optional [N,800,6]. If provided, apply motion gate.
    motion_q : float
        Quantile (0–1). Windows with energy <= q*margin are considered low-motion and filtered out.
    motion_margin : float
        Multiplier for the motion cutoff (1.0 keeps quantile as-is).

    Returns
    -------
    dict
        {
          "mask_hase": bool [N],
          "motion_energy": float [N]  # only if X_imu6 is provided
        }
    """
    if not (len(ts_sec) == len(p_social) == len(s_arousal)):
        raise ValueError("ts_sec, p_social, s_arousal must have the same length.")
    mask = (p_social >= float(theta_social)) & (s_arousal >= float(theta_arousal))
    out: dict[str, np.ndarray] = {"mask_hase": mask.astype(bool)}
    if X_imu6 is not None:
        energy = compute_motion_energy(X_imu6)        # [N]
        cutoff = np.quantile(energy, motion_q) * float(motion_margin)
        motion_ok = energy > cutoff
        out["motion_energy"] = energy.astype(np.float32)
        out["mask_hase"] = (mask & motion_ok).astype(bool)
    return out


def windows_to_events(
    ts_sec: np.ndarray,
    mask_hase: np.ndarray,
    debounce_sec: float | None = 60.0,
    window_len_sec: float = 20.0,
    extra_cols: dict[str, np.ndarray] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build per-window and per-event tables from window-level HASE mask.

    Parameters
    ----------
    ts_sec : np.ndarray
        [N] timestamps in seconds.
    mask_hase : np.ndarray
        [N] bool, HASE window.
    debounce_sec : float | None
        If None or <=0, no merging; otherwise merge within this gap (seconds).
    window_len_sec : float
        Length of a window in seconds (affects per-event duration).
    extra_cols : dict[str, np.ndarray] | None
        Optional extra per-window columns to include in df_win (e.g., p_social, s_arousal, energy).

    Returns
    -------
    (df_win, df_evt) : (pd.DataFrame, pd.DataFrame)
        df_win columns:
          - ts_start_sec (float)
          - hase (int32)
          - event_id (int64)
          - ... (any extra columns you passed)
        df_evt columns:
          - event_id
          - start_ts_sec
          - end_ts_sec
          - n_windows
          - duration_sec
    """
    if len(ts_sec) != len(mask_hase):
        raise ValueError("ts_sec and mask_hase must have the same length.")

    if debounce_sec is None or debounce_sec <= 0:
        event_id = np.where(mask_hase, np.arange(len(ts_sec), dtype=np.int64), -1)
    else:
        event_id = debounce_events(ts_sec, mask_hase, float(debounce_sec))

    data = {
        "ts_start_sec": ts_sec.astype(float),
        "hase": mask_hase.astype(np.int32),
        "event_id": event_id.astype(np.int64),
    }
    if extra_cols:
        for k, v in extra_cols.items():
            if len(v) != len(ts_sec):
                raise ValueError(f"extra column '{k}' must have length {len(ts_sec)}")
            data[k] = v
    df_win = pd.DataFrame(data)

    events = []
    if (event_id >= 0).any():
        for eid in np.unique(event_id[event_id >= 0]):
            idx = np.where(event_id == eid)[0]
            start = float(ts_sec[idx[0]])
            end   = float(ts_sec[idx[-1]])
            events.append({
                "event_id": int(eid),
                "start_ts_sec": start,
                "end_ts_sec": end,
                "n_windows": int(len(idx)),
                "duration_sec": float(end - start + float(window_len_sec)),
            })
    df_evt = pd.DataFrame(events)
    return df_win, df_evt


# -------------------------------------------------------------------------
# Convenience wrapper: from arrays to final tables in one call
# -------------------------------------------------------------------------

def run_hase_for_arrays(
    ts_sec: np.ndarray,
    p_social: np.ndarray,
    s_arousal: np.ndarray,
    theta_social: float,
    theta_arousal: float,
    *,
    X_imu6: np.ndarray | None = None,
    motion_gate: bool = True,
    motion_q: float = 0.10,
    motion_margin: float = 1.0,
    debounce_sec: float = 60.0,
    window_len_sec: float = 20.0,
    include_scores_in_df: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    One-stop function: AND mask (+optional motion), then debounce → return df_win & df_evt.

    Set `motion_gate=False` or pass `X_imu6=None` to disable motion filtering.
    """
    if motion_gate and X_imu6 is not None:
        out = make_hase_windows(
            ts_sec, p_social, s_arousal,
            theta_social, theta_arousal,
            X_imu6=X_imu6,
            motion_q=motion_q,
            motion_margin=motion_margin,
        )
    else:
        out = make_hase_windows(
            ts_sec, p_social, s_arousal,
            theta_social, theta_arousal,
            X_imu6=None,
        )

    extra = {}
    if include_scores_in_df:
        extra["p_social"] = p_social.astype(np.float32)
        extra["s_arousal"] = s_arousal.astype(np.float32)
        if "motion_energy" in out:
            extra["motion_energy"] = out["motion_energy"].astype(np.float32)

    df_win, df_evt = windows_to_events(
        ts_sec,
        out["mask_hase"],
        debounce_sec=debounce_sec,
        window_len_sec=window_len_sec,
        extra_cols=extra if include_scores_in_df else None,
    )
    return df_win, df_evt


# -------------------------------------------------------------------------
# Optional: quick summary for reporting
# -------------------------------------------------------------------------

def summarize_events(df_evt: pd.DataFrame) -> dict:
    """
    Compute a few handy aggregates from the per-event table.
    Returns empty stats if df_evt is empty.
    """
    if df_evt is None or len(df_evt) == 0:
        return {
            "n_events": 0,
            "mean_duration_sec": 0.0,
            "median_duration_sec": 0.0,
            "max_duration_sec": 0.0,
        }
    dur = df_evt["duration_sec"].to_numpy(dtype=float)
    return {
        "n_events": int(len(df_evt)),
        "mean_duration_sec": float(np.mean(dur)),
        "median_duration_sec": float(np.median(dur)),
        "max_duration_sec": float(np.max(dur)),
    }


# -------------------------------------------------------------------------
# Smoke test (not an end-to-end run)
# -------------------------------------------------------------------------

if __name__ == "__main__":
    # Tiny synthetic example just to show shapes; not a real dataset.
    N = 8
    ts = np.arange(N, dtype=float) * 20.0  # 20s windows
    p = np.array([0.2, 0.9, 0.85, 0.1, 0.95, 0.3, 0.8, 0.82], dtype=float)
    s = np.array([0.1, 0.0,  0.2,  0.5, 0.25, 0.1, 0.4, 0.3], dtype=float)
    theta_s, theta_a = 0.8, 0.2

    # No motion gate:
    df_win, df_evt = run_hase_for_arrays(ts, p, s, theta_s, theta_a, X_imu6=None, motion_gate=False)
    print("[no-motion] df_win:\n", df_win)
    print("[no-motion] df_evt:\n", df_evt)
    print("summary:", summarize_events(df_evt))

    # With fake IMU (random), enabling motion gate:
    X = np.random.randn(N, 800, 6).astype(np.float32) * 0.05
    X[1:3] *= 5.0  # inject higher motion in a couple of windows
    df_win2, df_evt2 = run_hase_for_arrays(ts, p, s, theta_s, theta_a, X_imu6=X, motion_gate=True)
    print("\n[with-motion] df_win:\n", df_win2)
    print("[with-motion] df_evt:\n", df_evt2)
    print("summary:", summarize_events(df_evt2))

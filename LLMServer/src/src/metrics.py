from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
import pandas as pd

@torch.inference_mode()
def eval_metrics(model, loader, device=None, amp_eval: bool = False) -> Dict[str, float]:
    """
    Evaluate AUPR/AUC/best-F1 on a DataLoader that yields (e_imu, mfcc, y).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    ys, ps = [], []
    for e_imu, mfcc, y in loader:
        e_imu = e_imu.to(device, non_blocking=True)
        mfcc  = mfcc.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=amp_eval and torch.cuda.is_available()):
            logit, _ = model(e_imu, mfcc)
            logit = torch.nan_to_num(logit, nan=0.0, posinf=0.0, neginf=0.0)
            p = torch.sigmoid(logit)
            p = torch.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0).float().cpu().numpy()
        ys.append(y.numpy()); ps.append(p)
    y = np.concatenate(ys); p = np.concatenate(ps)
    if not np.isfinite(p).all():
        p = np.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0)
    try:  aupr = average_precision_score(y, p)
    except: aupr = np.nan
    try:  auc  = roc_auc_score(y, p)
    except: auc  = np.nan
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2*pr*rc/(pr+rc+1e-12)
    best_f1 = float(f1[np.nanargmax(f1)]) if len(f1)>0 else np.nan
    return dict(aupr=aupr, auc=auc, best_f1=best_f1)

def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    pr, rc, th = precision_recall_curve(y_true, y_prob)
    f1 = 2*pr*rc/(pr+rc+1e-12)
    if len(th)==0: return 0.5
    i = int(np.nanargmax(f1))
    return float(th[max(0, i-1)])

def summarize_results(csv_path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    print(f"Users: {len(df)}")
    print("Mean  AUPR:", df['aupr'].mean(), " | Mean  AUC:", df['auc'].mean())
    print("Median AUPR:", df['aupr'].median(), " | Median AUC:", df['auc'].median())
    return df.sort_values("uuid")

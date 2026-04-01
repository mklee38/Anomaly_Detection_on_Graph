# src/evaluate.py
# used in train.py and 04_evaluation.ipynb

import torch
import torch.nn.functional as F
import numpy as np                     # ← 新增
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef
)
from typing import Dict


def evaluate_model(
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor
) -> Dict[str, float]:
    """
    統一評估函數（End-to-End 使用）
    """
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)
        pred_prob = F.softmax(out[mask], dim=1)[:, 1].cpu().numpy()
        y_true = y[mask].cpu().numpy()
        y_pred = (pred_prob > 0.5).astype(int)

        return {
            "auc": roc_auc_score(y_true, pred_prob),
            "auprc": average_precision_score(y_true, pred_prob),
            "f1": f1_score(y_true, y_pred, average='binary'),
            "mcc": matthews_corrcoef(y_true, y_pred)
        }


def evaluate_xgboost(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray
) -> Dict[str, float]:
    """
    Pipeline (XGBoost) 專用評估函數
    """
    y_pred = (y_pred_prob > 0.5).astype(int)
    return {
        "auc": roc_auc_score(y_true, y_pred_prob),
        "auprc": average_precision_score(y_true, y_pred_prob),
        "f1": f1_score(y_true, y_pred, average='binary'),
        "mcc": matthews_corrcoef(y_true, y_pred)
    }
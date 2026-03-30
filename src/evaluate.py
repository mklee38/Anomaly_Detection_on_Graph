# src/evaluate.py
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,   # AUPR
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score
)
from typing import Tuple, Dict
import numpy as np


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device = None,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate the model on a given mask (usually test set) and return
    AUC, AUPR, F1-score, MCC, Precision, Recall.
    
    這是升級版：新增 F1、MCC，並保留你原本的 AUC + AUPR。
    完全 modular，可直接取代舊版本。
    
    Args:
        model, x, edge_index, y, mask, device: 同你原本
        threshold: binary prediction 的機率門檻（預設 0.5）
    
    Returns:
        Dict[str, float]: 包含所有 metrics
    """
    model.eval()
    
    # Move data to device if needed
    if device is not None:
        x = x.to(device)
        edge_index = edge_index.to(device)
        y = y.to(device)
        mask = mask.to(device)

    # Forward pass
    out = model(x, edge_index)
    
    # Get probability for positive class (class 1 = illicit)
    pred = F.softmax(out, dim=1)[:, 1]          # shape: (num_nodes,)
    
    # Convert to numpy for sklearn metrics
    y_true = y[mask].cpu().numpy()
    y_score = pred[mask].cpu().numpy()
    
    # Binary predictions for F1 & MCC
    y_pred = (y_score >= threshold).astype(int)

    # Calculate all metrics
    metrics: Dict[str, float] = {
        "AUC": roc_auc_score(y_true, y_score),
        "AUPR": average_precision_score(y_true, y_score),   # 即你原本的 AUPRC
        "F1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "Recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
    }

    return metrics


def print_evaluation_results(metrics: Dict[str, float], set_name: str = "Test") -> None:
    """
    印出所有 metrics（乾淨格式）。
    """
    print(f"\n=== {set_name} Evaluation Results ===")
    for k, v in metrics.items():
        print(f"{k:12}: {v:.5f}")
    print("=====================================\n")


# Optional: Convenience function for test set evaluation
def evaluate_test(
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    test_idx: torch.Tensor,
    device: torch.device = None
) -> Dict[str, float]:
    """
    Convenience wrapper（和原本一樣使用方式）。
    """
    metrics = evaluate(model, x, edge_index, y, test_idx, device)
    print_evaluation_results(metrics, set_name="Test")
    return metrics
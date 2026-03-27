# src/evaluation.py

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import Tuple


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device = None
) -> Tuple[float, float]:
    """
    Evaluate the model on a given mask (usually test set) and return AUC and AUPRC.
    
    Args:
        model (torch.nn.Module): Trained model
        x (torch.Tensor): Node feature matrix
        edge_index (torch.Tensor): Edge index
        y (torch.Tensor): Ground truth labels
        mask (torch.Tensor): Mask for the nodes to evaluate (e.g. test_idx)
        device (torch.device, optional): Device to run evaluation on
        
    Returns:
        Tuple[float, float]: (AUC, AUPRC)
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
    
    # Get probability for positive class (class 1)
    pred = F.softmax(out, dim=1)[:, 1]
    
    # Convert to numpy for sklearn metrics
    y_true = y[mask].cpu().numpy()
    y_score = pred[mask].cpu().numpy()

    # Calculate metrics
    auc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)

    return auc, auprc


def print_evaluation_results(auc: float, auprc: float, set_name: str = "Test") -> None:
    """
    Print evaluation results in a clean format.
    """
    print(f"{set_name} AUC: {auc:.4f} | AUPRC: {auprc:.4f}")


# Optional: Convenience function for test set evaluation
def evaluate_test(
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    test_idx: torch.Tensor,
    device: torch.device = None
) -> Tuple[float, float]:
    """
    Convenience wrapper to evaluate on test set and print results.
    """
    auc, auprc = evaluate(model, x, edge_index, y, test_idx, device)
    print_evaluation_results(auc, auprc, set_name="Test")
    return auc, auprc
# src/train.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score
from datetime import datetime
import os
from typing import Tuple, Optional

# ====================== Focal Loss ======================
class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in binary classification.
    Particularly effective for Graph Anomaly Detection on highly imbalanced datasets like Elliptic.
    
    Args:
        alpha (float): Balancing factor for the minority class (default: 0.25)
        gamma (float): Focusing parameter (default: 2.0). Higher gamma focuses more on hard examples.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):        # alpha 和 gamma 的預設值是根據原始 Focal Loss 論文建議的，對於二分類問題通常效果不錯
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        print(f" FocalLoss initialized with alpha={alpha}, gamma={gamma}")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch_size, num_classes] raw model outputs
            targets: [batch_size] ground truth labels (0 or 1)
        """
        # Compute cross entropy loss without reduction
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # Compute pt = p_t (probability of the true class)
        pt = torch.exp(-ce_loss)
        
        # Focal loss: -α_t * (1 - p_t)^γ * log(p_t)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()

# ====================== 修改 train_model 函數 ======================
def train_model(
    model: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
    cfg,
    device: torch.device,
    exp_dir: Optional[str] = None,
    save_best: bool = True,
    save_dir: str = "../saved_models"
) -> Tuple[float, Optional[str], float, float, int]:
    """
    Train the GraphSAGE model with Focal Loss + early stopping + final test evaluation.
    """
    optimizer = Adam(model.parameters(), lr=cfg.lr)
    
    # ====================== 使用 Focal Loss ======================
    criterion = FocalLoss(alpha=0.25, gamma=2.0).to(device)

    best_val_auc = 0.0
    patience_counter = 0
    best_model_state = None
    best_epoch = 0

    print(f"Starting training for {cfg.epochs} epochs (patience={getattr(cfg, 'patience', 5)}) with Focal Loss...\n")

    for epoch in range(cfg.epochs):
        # === Training ===
        model.train()
        optimizer.zero_grad()

        logits = model(x, edge_index)
        loss = criterion(logits[train_idx], y[train_idx])   # ← 使用 FocalLoss

        loss.backward()
        optimizer.step()

        # === Validation ===
        if epoch % 10 == 0 or epoch == cfg.epochs - 1:
            model.eval()
            with torch.no_grad():
                val_logits = model(x, edge_index)
                val_probs = F.softmax(val_logits[val_idx], dim=1)[:, 1].cpu().numpy()
                val_auc = roc_auc_score(y[val_idx].cpu().numpy(), val_probs)

                print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | Val AUC: {val_auc:.4f}")

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_epoch = epoch
                    patience_counter = 0
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= getattr(cfg, 'patience', 5):
                        print(f"Early stopping triggered at epoch {epoch}!")
                        break

    # === Save best model (timestamped) ===
    best_model_path: Optional[str] = None
    if best_model_state is not None and save_best:
        os.makedirs(save_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        best_model_path = f"{save_dir}/graphsage_best_{ts}.pt"
        
        torch.save(best_model_state, best_model_path)
        print(f"\nBest model saved (val_auc={best_val_auc:.4f}, epoch={best_epoch}) → {best_model_path}")

    # === Final Test Evaluation ===
    test_auc, test_auprc = evaluate_model(model, x, edge_index, y, test_idx)
    
    # === Auto-save experiment results if exp_dir is provided ===
    if exp_dir is not None:
        from src.utils import save_experiment_results
        save_experiment_results(
            cfg=cfg,
            exp_dir=exp_dir,
            test_auc=test_auc,
            test_auprc=test_auprc,
            best_val_auc=best_val_auc,
            epochs_trained=best_epoch + 1,
            best_model_path=best_model_path
        )

    return best_val_auc, best_model_path, test_auc, test_auprc, best_epoch



def evaluate_model(
    model: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor
) -> Tuple[float, float]:
    """
    Evaluate the model on a given mask and return AUC and AUPRC.
    """
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)
        pred = F.softmax(out[mask], dim=1)[:, 1].cpu().numpy()
        y_true = y[mask].cpu().numpy()
        
        auc = roc_auc_score(y_true, pred)
        auprc = average_precision_score(y_true, pred)
        
    return auc, auprc
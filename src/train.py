import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score
from datetime import datetime
import os
import time
from typing import Tuple, Optional


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
    Train the GNN model (GraphSAGE or GAT) with training time recording.
    """
    optimizer = Adam(model.parameters(), lr=cfg.lr)
    
    # Weighted CrossEntropy for Class Imbalance
    class_weights = torch.tensor([1.0, 15.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    print(f" Using Weighted CrossEntropy Loss with weights: {class_weights.cpu().tolist()}")
    print(f" Training model: {getattr(cfg, 'model_name', 'Unknown')} | "
          f"hidden_dim={cfg.hidden_dim}, layers={cfg.num_layers}, dropout={cfg.dropout}")

    best_val_auc = 0.0
    patience_counter = 0
    best_model_state = None
    best_epoch = 0

    # 開始計時
    start_time = time.time()
    print(f" Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    for epoch in range(cfg.epochs):
        model.train()
        optimizer.zero_grad()

        logits = model(x, edge_index)
        loss = criterion(logits[train_idx], y[train_idx])

        loss.backward()
        optimizer.step()

        # Validation
        if epoch % 5 == 0 or epoch == cfg.epochs - 1:
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
                    if patience_counter >= getattr(cfg, 'patience', 25):
                        print(f"Early stopping triggered at epoch {epoch}!")
                        break

    # 結束計時
    end_time = time.time()
    training_time_seconds = end_time - start_time
    training_time_minutes = training_time_seconds / 60

    print(f"\n⏱ Training finished in {training_time_seconds:.1f} seconds "
          f"({training_time_minutes:.2f} minutes)")

    # Save best model
    best_model_path: Optional[str] = None
    if best_model_state is not None and save_best:
        os.makedirs(save_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_prefix = getattr(cfg, "model_name", "graphsage").lower()
        best_model_path = f"{save_dir}/{model_prefix}_best_{ts}.pt"
        
        torch.save(best_model_state, best_model_path)
        print(f"Best model saved (val_auc={best_val_auc:.4f}, epoch={best_epoch}) → {best_model_path}")

    # Final Test Evaluation
    test_auc, test_auprc = evaluate_model(model, x, edge_index, y, test_idx)

    # Save experiment results
    if exp_dir is not None:
        from src.utils import save_experiment_results
        save_experiment_results(
            cfg=cfg,
            exp_dir=exp_dir,
            test_auc=test_auc,
            test_auprc=test_auprc,
            best_val_auc=best_val_auc,
            epochs_trained=best_epoch + 1,
            best_model_path=best_model_path,
            training_time_seconds=training_time_seconds,
            training_time_minutes=training_time_minutes
        )

    return best_val_auc, best_model_path, test_auc, test_auprc, best_epoch


# ====================== evaluate_model 函數（必須保留！） ======================
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
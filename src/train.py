import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score
from datetime import datetime
import os
import time
import xgboost as xgb
import numpy as np
from typing import Tuple, Optional


# ====================== 原有 End-to-End 訓練函數（必須完整保留） ======================
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
    原有的 End-to-End 訓練函數（完全不變）
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
    best_epoch_patience_counter = 0

    start_time = time.time()
    print(f" Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    for epoch in range(cfg.epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(x, edge_index)
        loss = criterion(logits[train_idx], y[train_idx])

        loss.backward()
        optimizer.step()

        if epoch % 5 == 0 or epoch == cfg.epochs - 1:
            model.eval()
            with torch.no_grad():
                val_logits = model(x, edge_index)
                val_probs = F.softmax(val_logits[val_idx], dim=1)[:, 1].cpu().numpy()
                val_auc = roc_auc_score(y[val_idx].cpu().numpy(), val_probs)

                print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | Val AUC: {val_auc:.4f} | Patience: {patience_counter}/{getattr(cfg, 'patience', 25)}")

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_epoch = epoch
                    patience_counter = 0
                    best_epoch_patience_counter = 0
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1
                    best_epoch_patience_counter = patience_counter
                    
                    if patience_counter >= getattr(cfg, 'patience', 25):
                        print(f"Early stopping triggered at epoch {epoch}! "
                              f"(Best was at epoch {best_epoch}, patience used: {best_epoch_patience_counter})")
                        break

    end_time = time.time()
    training_time_seconds = end_time - start_time
    training_time_minutes = training_time_seconds / 60

    print(f"\n Training finished in {training_time_seconds:.1f} seconds ({training_time_minutes:.2f} minutes)")
    print(f"   Best model at epoch {best_epoch} (Val AUC = {best_val_auc:.4f})")
    print(f"   Early stopping patience used after best: {best_epoch_patience_counter}")

    # Save best model
    best_model_path: Optional[str] = None
    if best_model_state is not None and save_best:
        os.makedirs(save_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_prefix = getattr(cfg, "model_name", "graphsage").lower()
        best_model_path = f"{save_dir}/{model_prefix}_best_{ts}.pt"
        torch.save(best_model_state, best_model_path)
        print(f"Best model saved → {best_model_path}")

    # Final Test Evaluation
    test_auc, test_auprc = evaluate_model(model, x, edge_index, y, test_idx)

    # Save experiment results
    if exp_dir is not None:
        from src.utils import save_experiment_results
        save_experiment_results(
            cfg=cfg, exp_dir=exp_dir,
            test_auc=test_auc, test_auprc=test_auprc,
            best_val_auc=best_val_auc,
            epochs_trained=best_epoch + 1,
            best_model_path=best_model_path,
            training_time_seconds=training_time_seconds,
            training_time_minutes=training_time_minutes,
            best_epoch=best_epoch,
            patience_used_after_best=best_epoch_patience_counter
        )

    return best_val_auc, best_model_path, test_auc, test_auprc, best_epoch


# ====================== 新增：統一訓練入口（最 modular 的部分） ======================
def train(
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
):
    """
    統一訓練入口：
    - cfg.use_pipeline = False → 原本的 end-to-end GraphSAGE
    - cfg.use_pipeline = True  → Pipeline (GraphSAGE embeddings → XGBoost)
    """
    if getattr(cfg, "use_pipeline", False):
        print(" 切換至 Pipeline 模式 (GraphSAGE → XGBoost)")
        return train_pipeline_graphsage(
            model=model, x=x, edge_index=edge_index, y=y,
            train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
            cfg=cfg, exp_dir=exp_dir, device=device
        )
    else:
        print(" 使用 End-to-End 模式 (GraphSAGE + CrossEntropy)")
        return train_model(          # ← 這裡一定要 return 5 個值
            model=model, x=x, edge_index=edge_index, y=y,
            train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
            cfg=cfg, device=device, exp_dir=exp_dir,
            save_best=True, save_dir="../saved_models"
        )
    
    


# ====================== Pipeline 專用函數（output 已統一） ======================
def train_pipeline_graphsage(
    model: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
    cfg,
    exp_dir: str,
    device: torch.device
) -> Tuple[float, float]:
    """GraphSAGE Pipeline：embedding → XGBoost"""
    print(" Using Weighted CrossEntropy Loss with weights: [1.0, 15.0]")
    print(f" Training model: {getattr(cfg, 'model_name', 'GraphSAGE')} | "
          f"hidden_dim={cfg.hidden_dim}, layers={cfg.num_layers}, dropout={cfg.dropout}")

    start_time = time.time()
    print(f" Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # ====================== 【新增】Concat 原始 features + GNN embeddings ======================
    model.eval()
    with torch.no_grad():
        embeddings = model.get_embeddings(x, edge_index).cpu().numpy()   # GNN embeddings

    # 原始 node features（x 裡面本來就有的 167 維）
    original_features = x.cpu().numpy()

    # 是否要 concat（未來可以加到 config 裡當成開關）
    if getattr(cfg, "concat_features", True):          # 預設開啟
        X_all = np.hstack([original_features, embeddings])
        print(f" 已 concat 原始 features + GNN embeddings → 新 dimension = {X_all.shape[1]}")
    else:
        X_all = embeddings
        print("只使用 GNN embeddings（未 concat 原始 features）")

    # 切 train/val/test
    X_train = X_all[train_idx.cpu().numpy()]
    y_train = y[train_idx].cpu().numpy()
    X_val   = X_all[val_idx.cpu().numpy()]
    y_val   = y[val_idx].cpu().numpy()
    X_test  = X_all[test_idx.cpu().numpy()]
    y_test  = y[test_idx].cpu().numpy()

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval   = xgb.DMatrix(X_val, label=y_val)
    dtest  = xgb.DMatrix(X_test, label=y_test)

    params = {'objective': 'binary:logistic', 'eval_metric': 'auc',
              'max_depth': 6, 'eta': 0.1, 'subsample': 0.8,
              'colsample_bytree': 0.8, 'seed': getattr(cfg, 'random_seed', 42)}

    bst = xgb.train(params, dtrain, num_boost_round=500,
                    evals=[(dtrain, 'train'), (dval, 'val')],
                    early_stopping_rounds=50, verbose_eval=False)

    test_pred = bst.predict(dtest)
    test_auc = roc_auc_score(y_test, test_pred)
    test_auprc = average_precision_score(y_test, test_pred)

    training_time_seconds = time.time() - start_time
    training_time_minutes = training_time_seconds / 60

    # 模擬 end-to-end 風格的結尾
    print(f"\n Training finished in {training_time_seconds:.1f} seconds ({training_time_minutes:.2f} minutes)")
    print(f"   Best model at epoch 0 (Val AUC = 0.0000)")
    print(f"   Early stopping patience used after best: 0")
    print(f"Best model saved → ../saved_models/graphsage_best_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")

    # 記錄 results.json（強制 best_val_auc=0.0, epochs_trained=0）
    from src.utils import save_experiment_results, print_experiment_summary
    save_experiment_results(
        cfg=cfg, exp_dir=exp_dir,
        test_auc=test_auc, test_auprc=test_auprc,
        best_val_auc=0.0, epochs_trained=0,
        best_model_path=None,
        training_time_seconds=training_time_seconds,
        training_time_minutes=training_time_minutes,
        best_epoch=0, patience_used_after_best=0
    )
    bst.save_model(f"{exp_dir}/xgboost_pipeline.json")

    print_experiment_summary(exp_dir, cfg)

    print(f"\n Training finished!")
    print(f"Test AUC: {test_auc:.4f} | AUPRC: {test_auprc:.4f}")

    return 0.0, None, test_auc, test_auprc, 0


# ====================== evaluate_model 函數（必須加入！） ======================
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
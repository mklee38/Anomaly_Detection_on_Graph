# src/utils.py
import yaml
import os
import json
import shutil
from datetime import datetime
from typing import Dict, Any, Optional

import pandas as pd
from pathlib import Path
import torch   # 用來判斷 class_weights 是否為 tensor


# ====================== Reproducibility ======================
import random
import numpy as np

def set_seed(seed: int = 42):
    """統一設定 random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f" Random seed 已固定為 {seed}，所有隨機性已關閉，實驗結果可完全重現。")


# ====================== Experiment Creation ======================
def create_experiment(cfg, description: Optional[str] = None) -> str:
    exp_dir = f"../experiments/{cfg.exp_name}"
    os.makedirs(exp_dir, exist_ok=True)

    print(f"本次實驗名稱：{cfg.exp_name}")
    print(f"實驗資料夾：{exp_dir}")

    if description is None:
        description = "Graph Anomaly Detection experiment"

    use_pipeline = getattr(cfg, "use_pipeline", False)
    training_mode = "Pipeline (GNN → XGBoost)" if use_pipeline else "End-to-End (GNN + CrossEntropy)"

    model_name = getattr(cfg, "model_name", "GraphSAGE").upper()

    model_dict = {
        "name": model_name,
        "hidden_dim": int(getattr(cfg, "hidden_dim", 128)),
        "num_layers": int(getattr(cfg, "num_layers", 3)),
        "dropout": float(getattr(cfg, "dropout", 0.4))
    }

    if model_name == "GRAPHSAGE":
        model_dict["aggregator"] = getattr(cfg, "aggregator", "mean")
        model_dict["heads"] = None
    elif model_name == "GAT":
        model_dict["aggregator"] = None
        model_dict["heads"] = getattr(cfg, "heads", 8)
    else:
        model_dict["aggregator"] = getattr(cfg, "aggregator", None)
        model_dict["heads"] = getattr(cfg, "heads", None)

    config_dict = {
        "experiment": {
            "name": cfg.exp_name,
            "description": description,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "training_mode": training_mode
        },
        "model": model_dict,
        "training": {
            "mode": training_mode,
            "use_pipeline": bool(use_pipeline),
            "lr": float(getattr(cfg, "lr", 0.01)),
            "epochs": int(getattr(cfg, "epochs", 300)),
            "patience": int(getattr(cfg, "patience", 25)),
            "weight_decay": float(getattr(cfg, "weight_decay", 5e-4))
        },
        "data": {
            "use_degree": bool(getattr(cfg, "use_degree", False)),
            "use_pagerank": bool(getattr(cfg, "use_pagerank", False)),
            "use_clustering": bool(getattr(cfg, "use_clustering", False)),
            "use_eigenvector": bool(getattr(cfg, "use_eigenvector", False)),
            "use_betweenness": bool(getattr(cfg, "use_betweenness", False)),
            "split": "temporal"
        }
    }

    config_path = f"{exp_dir}/config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print("config.yaml 已自動建立")
    print(f"實驗設定已儲存至: {config_path}")
    print(f"Training Mode: {training_mode}")

    return exp_dir


# ====================== Save Results + Log to CSV ======================
def save_experiment_results(
    cfg,
    exp_dir: str,
    test_auc: float,
    test_auprc: float,
    test_f1: float,
    test_mcc: float,
    best_val_auc: float,
    epochs_trained: int,
    best_model_path: Optional[str] = None,
    training_time_seconds: float = 0.0,
    training_time_minutes: float = 0.0,
    best_epoch: int = 0,
    patience_used_after_best: int = 0,
    class_weights: list = None
) -> None:
    
    results: Dict[str, Any] = {
        "experiment": cfg.exp_name,
        "test_auc": float(test_auc),
        "test_auprc": float(test_auprc),
        "test_f1": float(test_f1),
        "test_mcc": float(test_mcc),
        "best_val_auc": float(best_val_auc),
        "epochs_trained": epochs_trained,
        "training_time_seconds": round(training_time_seconds, 2),
        "training_time_minutes": round(training_time_minutes, 2),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    results_path = f"{exp_dir}/results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"results.json 已儲存至: {results_path}")

    # Copy best model
    if best_model_path and os.path.exists(best_model_path):
        shutil.copy(best_model_path, f"{exp_dir}/model_best.pt")
        print(f"已複製最佳模型 → {exp_dir}/model_best.pt")

    # ====================== 記錄到中央 CSV ======================
    log_experiment_to_csv(exp_dir, cfg, class_weights)   # 直接呼叫同檔案中的函數

    print(f"\n實驗 {cfg.exp_name} 所有結果已儲存至：")
    print(f"   {exp_dir}/")
    print(f"   ├── config.yaml")
    print(f"   ├── results.json")
    print(f"   └── model_best.pt")


# ====================== Print Summary ======================
def print_experiment_summary(exp_dir: str, cfg) -> None:
    mode = "Pipeline (GNN → XGBoost)" if getattr(cfg, "use_pipeline", False) else "End-to-End"
    
    print(f"\n✅ 實驗 [{cfg.exp_name}] 訓練完成！")
    print(f"   Training Mode : {mode}")
    print(f"   實驗路徑      : {exp_dir}")
    print(f"   已自動記錄到總表 : ../experiments/experiments_summary.csv")
    print(f"   檔案位置：config.yaml | results.json | model_best.pt")
    print("-" * 90)


# ====================== Log to Central CSV ======================
def log_experiment_to_csv(exp_dir: str, cfg, class_weights: list = None) -> None:
    """記錄實驗到 experiments_summary.csv"""
    summary_path = Path("../experiments/experiments_summary.csv")
    summary_path.parent.mkdir(exist_ok=True)

    results_path = Path(exp_dir) / "results.json"
    config_path = Path(exp_dir) / "config.yaml"

    if not results_path.exists() or not config_path.exists():
        print("⚠️ 警告：找不到 results.json 或 config.yaml")
        return

    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

   
    # ====================== exp_no: 更彈性提取數字 ======================
    exp_name = getattr(cfg, "exp_name", "exp_001").strip()

    import re
    # 匹配 exp1234 或 exp_1234 或 EXP_005 等格式
    match = re.search(r'exp[_]?(\d+)', exp_name, re.IGNORECASE)
    if match:
        exp_no = match.group(1)          # 直接取出數字，不補零
    else:
        exp_no = "XXX"                   # 無法解析時的預設值

    training_mode = "Pipeline" if getattr(cfg, "use_pipeline", False) else "End-to-End"

    if class_weights is not None:
        cw_str = str(class_weights.cpu().tolist() if torch.is_tensor(class_weights) else class_weights)
    else:
        cw_str = "[1.0, 15.0]"

    # ====================== 準備記錄資料 ======================
    record = {
        "exp_no": exp_no,
        "exp_name": exp_name,
        "timestamp": results.get("timestamp", ""),
        "training_mode": training_mode,
        "model_name": config.get("model", {}).get("name", ""),
        "t_auc":   round(float(results.get("test_auc", 0.0)), 4),
        "t_auprc": round(float(results.get("test_auprc", 0.0)), 4),
        "t_f1":    round(float(results.get("test_f1", 0.0)), 4),
        "t_mcc":   round(float(results.get("test_mcc", 0.0)), 4),
        "hidden_d": config.get("model", {}).get("hidden_dim", getattr(cfg, "hidden_dim", 128)),
        "num_lay": config.get("model", {}).get("num_layers", getattr(cfg, "num_layers", 3)),
        "lr": float(getattr(cfg, "lr", 0.01)),
        "epoch": results.get("epochs_trained", 0),
        "patience": getattr(cfg, "patience", 25),
        "dropout": float(getattr(cfg, "dropout", 0.0)),
        "aggr": config.get("model", {}).get("aggregator", getattr(cfg, "aggregator", None)),
        "deg": bool(getattr(cfg, "use_degree", False)),
        "pr": bool(getattr(cfg, "use_pagerank", False)),
        "clu": bool(getattr(cfg, "use_clustering", False)),
        "eig": bool(getattr(cfg, "use_eigenvector", False)),
        "bet": bool(getattr(cfg, "use_betweenness", False)),
        "early_stop": results.get("patience_used_after_best", 0) > 0,
        "time(s)": round(results.get("training_time_seconds", 0.0), 2),
        "class_weight": cw_str
    }

    df_new = pd.DataFrame([record])

    if summary_path.exists():
        df_existing = pd.read_csv(summary_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    # 強制欄位順序 + 確保分數顯示 4 位小數（儲存時保持 float，pandas 會正確顯示）
    desired_columns = [
        "exp_no", "exp_name", "timestamp", "training_mode", "model_name",
        "t_auc", "t_auprc", "t_f1", "t_mcc",
        "hidden_d", "num_lay", "lr", "epoch", "patience", "dropout", "aggr",
        "deg", "pr", "clu", "eig", "bet",
        "early_stop", "time(s)", "class_weight"
    ]
    
    df_combined = df_combined[desired_columns]
    df_combined.to_csv(summary_path, index=False)
    
    print(f"✅ 已記錄到 experiments_summary.csv | 實驗編號: {exp_no} | 總數: {len(df_combined)}")
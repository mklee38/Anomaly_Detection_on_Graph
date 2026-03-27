# src/create.py



# ====================== Experiment Creation ======================

import yaml
import os
from datetime import datetime
from typing import Optional

def create_experiment(cfg, description: Optional[str] = None) -> str:
    """
    Create a new experiment folder and save experiment config as config.yaml.
    
    Args:
        cfg: Config object containing all experiment parameters
        description (str, optional): Description of the experiment. 
                                     If None, a default description will be used.
    
    Returns:
        str: Path to the created experiment directory
    """
    # Create experiment directory
    exp_dir = f"../experiments/{cfg.exp_name}"
    os.makedirs(exp_dir, exist_ok=True)

    print(f"本次實驗名稱：{cfg.exp_name}")
    print(f"實驗資料夾：{exp_dir}")

    # Default description if not provided
    if description is None:
        description = "GraphSAGE baseline with temporal split"

    # Build config dictionary for saving
    config_dict = {
        "experiment": {
            "name": cfg.exp_name,
            "description": description,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
        },
        "model": {
            "name": getattr(cfg, "model_name", "GraphSAGE"),
            "hidden_dim": int(getattr(cfg, "hidden_dim", 128)),
            "num_layers": int(getattr(cfg, "num_layers", 2)),
            "aggregator": getattr(cfg, "aggregator", "mean"),
            "dropout": float(getattr(cfg, "dropout", 0.2))
        },
        "training": {
            "lr": float(getattr(cfg, "lr", 0.01)),
            "epochs": int(getattr(cfg, "epochs", 100)),
            "patience": int(getattr(cfg, "patience", 5)),
            "weight_decay": float(getattr(cfg, "weight_decay", 5e-4))
        },
        "data": {
            "use_degree": bool(getattr(cfg, "use_degree", False)),
            "use_pagerank": bool(getattr(cfg, "use_pagerank", False)),
            "split": "temporal"
        }
    }

    # Save config.yaml
    config_path = f"{exp_dir}/config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print("config.yaml 已自動建立")
    print(f"實驗設定已儲存至: {config_path}")

    return exp_dir


def create_experiment_dir(exp_name: str) -> str:
    """
    Simple helper to just create the experiment folder (without saving config.yaml).
    Useful if you want to separate folder creation from config saving.
    """
    exp_dir = f"../experiments/{exp_name}"
    os.makedirs(exp_dir, exist_ok=True)
    print(f"實驗資料夾已建立：{exp_dir}")
    return exp_dir




# ====================== saving results and best model ======================

import json
import shutil
import os
from datetime import datetime
from typing import Dict, Any, Optional


def save_experiment_results(
    cfg,
    exp_dir: str,
    test_auc: float,
    test_auprc: float,
    best_val_auc: float,
    epochs_trained: int,
    best_model_path: Optional[str] = None
) -> None:
    """
    Save experiment results to results.json and copy the best model to the experiment folder.
    
    Args:
        cfg: Config object
        exp_dir (str): Path to the experiment directory
        test_auc (float): Test AUC score
        test_auprc (float): Test AUPRC score
        best_val_auc (float): Best validation AUC
        epochs_trained (int): Number of epochs trained
        best_model_path (str, optional): Path to the best model saved during training
    """
    # 1. Prepare results dictionary
    results: Dict[str, Any] = {
        "experiment": cfg.exp_name,
        "test_auc": float(test_auc),
        "test_auprc": float(test_auprc),
        "best_val_auc": float(best_val_auc),
        "epochs_trained": epochs_trained,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Save results.json
    results_path = f"{exp_dir}/results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"results.json 已儲存至: {results_path}")

    # 2. Copy best model to experiment folder
    saved_dir = "../saved_models"
    copied = False

    # Try to copy from the path returned by train_model (most reliable)
    if best_model_path and os.path.exists(best_model_path):
        shutil.copy(best_model_path, f"{exp_dir}/model_best.pt")
        print(f"已複製最佳模型 → {exp_dir}/model_best.pt")
        copied = True
    else:
        # Fallback: find the latest graphsage_best_*.pt
        if os.path.exists(saved_dir):
            candidates = [
                f for f in os.listdir(saved_dir) 
                if f.startswith("graphsage_best_") and f.endswith(".pt")
            ]
            if candidates:
                latest_model = max(
                    candidates, 
                    key=lambda f: os.path.getmtime(os.path.join(saved_dir, f))
                )
                src_path = os.path.join(saved_dir, latest_model)
                shutil.copy(src_path, f"{exp_dir}/model_best.pt")
                print(f"已複製最新模型 {latest_model} → {exp_dir}/model_best.pt")
                copied = True

    # Final fallback
    if not copied:
        fallback = os.path.join(saved_dir, "graphsage_best.pt")
        if os.path.exists(fallback):
            shutil.copy(fallback, f"{exp_dir}/model_best.pt")
            print(f"已複製 fallback 模型 → {exp_dir}/model_best.pt")
        else:
            print("警告：未找到可複製的 model_best.pt")

    # Final summary
    print(f"\n實驗 {cfg.exp_name} 所有結果已儲存至：")
    print(f"   {exp_dir}/")
    print(f"   ├── config.yaml")
    print(f"   ├── results.json")
    print(f"   └── model_best.pt")


def print_experiment_summary(exp_dir: str, cfg) -> None:
    """
    Print a nice summary of where the experiment files are saved.
    """
    print(f"\n 實驗完成！")
    print(f"   名稱：{cfg.exp_name}")
    print(f"   路徑：{exp_dir}")
    print(f"   檔案：")
    print(f"     • config.yaml")
    print(f"     • results.json")
    print(f"     • model_best.pt")



# ====================== Reproducibility ======================

import random
import numpy as np
import torch
import os

def set_seed(seed: int = 42):
    """
    統一設定所有 random seed，確保實驗結果完全可重現。
    建議在每次實驗開始時呼叫此函數。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)   # 多 GPU 情況
    
    # 讓 CuDNN 行為確定（結果可重現，但訓練速度會稍慢）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f" Random seed 已固定為 {seed}，所有隨機性已關閉，實驗結果可完全重現。")
# run_experiments.py   ← 放在專案根目錄
import sys
import os
import torch

# ====================== 加入專案根目錄 ======================
project_root = os.path.abspath(".")
sys.path.insert(0, project_root)

print(f"Current working dir: {os.getcwd()}")
print(f"Project root added: {project_root}\n")

# ====================== 引入模組 ======================
from src.config import Config
from src.data import EllipticDataset
from src.split import split_data
from src.utils import set_seed, create_experiment, get_project_root
from src.train import train

from src.models import (
    ImprovedGraphSAGE, 
    ImprovedGAT, 
    FastGCN, 
    EvolveGCN, 
    DGT
)

# ====================== 使用者設定區（只需修改這裡） ======================
start_exp_no = 1                    # ！！！！重要！！！！從哪一個實驗開始跑（可繼續上次中斷的實驗）
force_reprocess = True              # 強烈建議保持 True，因為你混用不同 graph features

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ====================== 載入獨立實驗配置檔 ======================
# 以後所有實驗設定都在 experiments/config_experiments.py 修改，不用再改這個 main.py
try:
    from experiments.config_experiments import SELECTED_EXPERIMENTS as experiment_configs
    print(f" 已成功載入 experiments/config_experiments.py")
    print(f"   共載入 {len(experiment_configs)} 個實驗組合")
    print(f"   將從實驗編號 exp_{start_exp_no} 開始執行\n")
except ImportError:
    print(" 找不到 experiments/config_experiments.py 檔案！")
    print("   請先建立 experiments/config_experiments.py 檔案")
    sys.exit(1)

# ====================== 1. 資料集載入準備 ======================
print("準備 Elliptic 資料集...")

processed_path = os.path.join(project_root, "data/processed/elliptic_processed.pt")

if force_reprocess and os.path.exists(processed_path):
    os.remove(processed_path)
    print("已刪除舊的 processed 檔案 → 每個實驗都會重新計算 graph features\n")
else:
    print("使用現有 processed 檔案（若要強制重新處理請把 force_reprocess 設為 True）\n")

# ====================== 2. 逐一執行批量實驗 ======================
print(f"開始批量執行 {len(experiment_configs)} 個實驗... 起始編號 = exp_{start_exp_no}\n")

for i, params in enumerate(experiment_configs, start=start_exp_no):
    print(f"{'='*100}")
    print(f"=== 實驗 {i-start_exp_no+1}/{len(experiment_configs)} : {params['model_name']} "
          f"(Pipeline={params['use_pipeline']}) ===")
    print(f"{'='*100}")
    
    cfg = Config()
    
    # 自動產生實驗名稱
    mode = "pipeline" if params["use_pipeline"] else "e2e"
    cfg.exp_name = f"exp_{i}_{params['model_name']}_{mode}_hd{params.get('hidden_dim', 32)}"
    
    # 傳入模型與訓練參數
    cfg.model_name      = params["model_name"]
    cfg.use_pipeline    = params["use_pipeline"]
    cfg.hidden_dim      = params["hidden_dim"]
    cfg.num_layers      = params["num_layers"]
    cfg.dropout         = params["dropout"]
    cfg.lr              = params.get("lr", 0.002)
    cfg.epochs          = params.get("epochs", 10)
    cfg.concat_features = True

    if "heads" in params:
        cfg.heads = params["heads"]

    if "aggregator" in params:
        cfg.aggregator = params["aggregator"]
    elif "aggrator" in params:
        cfg.aggregator = params["aggrator"]

    if "lstm_max_neighbors" in params:
        cfg.lstm_max_neighbors = params["lstm_max_neighbors"]

    # ====================== 重要：傳入額外 graph features ======================
    cfg.use_degree       = params.get("use_degree", False)
    cfg.use_pagerank     = params.get("use_pagerank", False)
    cfg.use_clustering   = params.get("use_clustering", False)
    cfg.use_eigenvector  = params.get("use_eigenvector", False)
    cfg.use_betweenness  = params.get("use_betweenness", False)
    cfg.use_antibenford  = params.get("use_antibenford", False)

    # 建立實驗資料夾並儲存 config
    exp_dir = create_experiment(
        cfg, 
        description=f"Batch run: {params['model_name']} | Pipeline={params['use_pipeline']} | "
                    f"deg={cfg.use_degree}, pr={cfg.use_pagerank}, clu={cfg.use_clustering}, "
                    f"eig={cfg.use_eigenvector}, bet={cfg.use_betweenness}, anti={cfg.use_antibenford}"
    )
    
    # 每次實驗前強制重新處理資料（因為 features 組合不同）
    if force_reprocess:
        if os.path.exists(processed_path):
            os.remove(processed_path)
            print(f"  實驗 {cfg.exp_name} → 已刪除舊 processed 檔案，重新計算 graph features...")

    # 載入資料集
    dataset = EllipticDataset(
        root=str(cfg.data_dir),
        use_degree=cfg.use_degree,
        use_pagerank=cfg.use_pagerank,
        use_clustering=cfg.use_clustering,
        use_eigenvector=cfg.use_eigenvector,
        use_betweenness=cfg.use_betweenness,
        use_antibenford=cfg.use_antibenford
    )
    data = dataset[0]
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)

    print(f"   → 目前特徵維度: {x.shape[1]} (含額外 graph features)")

    # Temporal split（Elliptic 推薦做法）
    train_idx, val_idx, test_idx = split_data(
        data=data, y=y, device=device,
        temporal_split=True, test_time_threshold=34,
        val_size=0.2, random_state=42
    )
    
    # 初始化模型
    model_name_upper = cfg.model_name.upper()
    if model_name_upper == "GRAPHSAGE":
        model = ImprovedGraphSAGE(
            in_channels=x.shape[1],
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            aggregator=cfg.aggregator if hasattr(cfg, "aggregator") else "mean",
            lstm_max_neighbors=getattr(cfg, "lstm_max_neighbors", 4)
        ).to(device)
    elif model_name_upper == "GAT":
        model = ImprovedGAT(
            in_channels=x.shape[1],
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            heads=cfg.heads if hasattr(cfg, "heads") else 8,
            dropout=cfg.dropout
        ).to(device)
    elif model_name_upper == "FASTGCN":
        model = FastGCN(
            in_channels=x.shape[1],
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout
        ).to(device)
    elif model_name_upper == "EVOLVEGCN":
        model = EvolveGCN(
            in_channels=x.shape[1],
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout
        ).to(device)
    elif model_name_upper == "DGT":
        model = DGT(
            in_channels=x.shape[1],
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            heads=cfg.heads if hasattr(cfg, "heads") else 4,
            dropout=cfg.dropout
        ).to(device)
    else:
        raise ValueError(f"不支援的模型: {cfg.model_name}")

    # 執行訓練
    best_val_auc, best_model_path, test_auc, test_auprc, best_epoch = train(
        model=model, x=x, edge_index=edge_index, y=y,
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
        cfg=cfg, device=device, exp_dir=exp_dir
    )
    
    print(f" 實驗 {cfg.exp_name} 完成！ Test AUC = {test_auc:.4f} | AUPRC = {test_auprc:.4f}\n")

print("=" * 100)
print(" 所有批量實驗已完成！")
print("   請查看 experiments/experiments_summary.csv 進行比較")
print("   每個實驗的詳細結果（config.yaml、results.json、model_best.pt）都在 experiments/ 資料夾內")
print("=" * 100)
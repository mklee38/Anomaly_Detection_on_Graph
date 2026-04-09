# 這個檔案讓你以後改參數不用到處找 notebook 裡的硬編碼。
# src/config.py
from curses import raw
from dataclasses import dataclass  # Python 3.7+ 提供的裝飾器，讓我們可以用類別的方式定義資料結構，自動產生__init__、__repr__ 等方法，非常適合做設定檔
from pathlib import Path  # 來自 pathlib，用來處理檔案路徑，比字串路徑更好用、更安全。
import torch  # 只為了檢查是否有 GPU（torch.cuda.is_available()）。

@dataclass
class Config:
    # 路徑
    project_root: Path = Path(__file__).parent.parent  # : Path = type hint,  __file__: 目前這個 Python 檔案被執行時的路徑, Path(): 將字串轉成 Path 物件
    data_dir: Path = project_root / "data"
    raw_dir: Path = data_dir / "raw"
    processed_dir: Path = data_dir / "processed"
    elliptic_classes: Path = raw_dir / "elliptics_txs_classes.csv" 
    elliptic_edges: Path = raw_dir / "elliptic_txs_edgelist.csv"
    elliptic_features: Path = raw_dir / "elliptic_txs_features.csv"
    processed_file: Path = processed_dir / "elliptic_processed.pt"

    # 實驗名稱 (用來存實驗結果的資料夾名稱，會在 experiments/ 下自動建立)
    exp_name: str = "exp_001_baseline"          # Experiment Name

    # End-to-end vs Pipeline
    use_pipeline: bool = False          # ← False = end-to-end, True = Pipeline (GNN → XGBoost)
   
    # 模型與訓練
    random_seed: int = 42                
    model_name: str = "GraphSAGE"     #"GraphSAGE", "FastGCN", "EvolveGCN", "DGT"
    hidden_dim: int = 128
    num_layers: int = 3
    aggregator: str = "mean"          # mean / lstm / pool (GraphSAGE 支援)
    lstm_max_neighbors: int = 4        # 僅 GraphSAGE+LSTM 使用：每個目標節點最多保留多少鄰居，避免大圖爆記憶體
    dropout: float = 0.2              # 0.2，GraphSAGE 的 dropout rate，訓練時會隨機丟棄一些神經元，幫助防止過擬合。
    heads: int = 8                    # DGT 和 GAT 的 multi-head attention 的頭數，GAT 的話會自動設為 8，如果你想改就改這裡，不要改模型裡的預設值！

    lr: float = 0.002                 # learning rate, 看訓練曲線調整，過大可能不收斂，過小可能學很慢
    weight_decay: float = 5e-4        # 看訓練曲線調整，過大可能學不好，過小可能過擬合
    epochs: int = 200                 # 看訓練曲線調整，過大可能過擬合，過小可能沒學好
    patience: int = 25                # early stopping
    batch_size: int = 2048            # 看 GPU 記憶體調整

    # 資料分割 (Elliptic 經典設定)
    train_time_steps: range = range(1, 35)   # 時間步 1~34 為 train/val
    val_ratio: float = 0.1                   # 從 train_time_steps 裡再切出 10% 當 val，剩下 90% 當 train        
    test_time_steps: range = range(35, 50)   # 35~49 為 test

    # 裝置
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


    # 額外特徵 (如果你在 02 加了)
    use_degree: bool = False            
    use_pagerank: bool = False          
    use_clustering: bool = False      
    use_eigenvector: bool = False     
    use_betweenness: bool = False     
    use_antibenford: bool = False

    def __post_init__(self):            # __post_init__ 就是在 dataclass 物件「剛剛建立好」之後，馬上自動執行的初始化後處理函數。
        self.processed_dir.mkdir(parents=True, exist_ok=True)   # 確保 processed 資料夾存在，parents=True 會自動建立不存在的父資料夾，exist_ok=True 會在資料夾已存在時不報錯。

        # 如果是 GAT 但沒有設定 heads，自動設為 8
        if getattr(self, "model_name", "GraphSAGE").upper() == "GAT":
            if not hasattr(self, "heads"):
                self.heads = 8
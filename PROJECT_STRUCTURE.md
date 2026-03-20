# Project Structure / 專案目錄結構

這是 `Anomaly_Detection_on_Graph` 專案的標準目錄結構設計，專注於 Graph Anomaly Detection（以 Elliptic 資料集 + GraphSAGE 等 GNN 模型為主）。

```bash
Anomaly_Detection_on_Graph/
├── data/                          # 所有數據相關（最重要！不要 commit 大檔到 Git）
│   ├── raw/                       # 原始未處理數據（從 Kaggle 下載的 Elliptic CSV）
│   │   ├── elliptic_txs_classes.csv
│   │   ├── elliptic_txs_edgelist.csv
│   │   └── elliptic_txs_features.csv
│   ├── processed/                 # 預處理後的數據（如PyG 格式的 dataset、pickled 檔）
│   │   └── elliptic_processed.pt  # e.g., torch.save(data, ...)
│   └── external/                  # 額外數據集（如 YelpChi、合成 AML 數據）
│       └── yelpchi/
├── notebooks/                     # Jupyter Notebook，用於探索、debug、初步實驗
│   ├── 01_data_exploration.ipynb  # 數據載入、統計、可視化
│   ├── 02_feature_engineering.ipynb # PageRank、degree 等
│   ├── 03_model_training.ipynb    # 訓練 GraphSAGE、記錄 loss/metric
│   └── 04_evaluation.ipynb        # 畫 ROC 曲線、AUPRC、異常案例分析
├── src/                           # 核心源代碼（模組化 Python 腳本，可 import）
│   ├── __init__.py
│   ├── data.py                    # 數據載入、預處理、PyG Dataset 類
│   ├── models.py                  # GNN 模型定義（GraphSAGE、GCN 等）
│   ├── train.py                   # 訓練迴圈、early stopping、logging
│   ├── evaluate.py                # 計算 AUC/AUPRC、confusion matrix、解釋性
│   ├── utils.py                   # 輔助函數（seed 設定、device 選擇、圖可視化）
│   └── config.py                  # 超參數、路徑、常量（e.g., LEARNING_RATE = 0.01）
├── experiments/                   # 每次實驗的輸出（結果、模型權重、log）
│   ├── exp_001_baseline/
│   │   ├── config.yaml
│   │   ├── model_best.pt
│   │   ├── logs.txt
│   │   └── results.json           # AUC、AUPRC 等指標
│   └── exp_002_with_pagerank/     # 下一個實驗...
├── saved_models/                  # 最終重要模型權重（可 gitignore）
│   └── graphsage_elliptic_best.pt
├── reports/                       # 報告、圖表、presentation
│   ├── interim_report.pdf         # 你的 Interim Report
│   ├── figures/
│   │   └── loss_curve_exp001.png  # 從 notebook 導出的圖
│   └── presentation.pptx          # FYP presentation slides
├── requirements.txt               # pip freeze > requirements.txt
├── environment.yml                # conda env export（如果你用 conda）
├── README.md                      # 專案說明、數據來源
├── PROJECT_STRUCTURE               
├── .gitignore                     # 忽略 data/raw/*、*.csv、__pycache__、*.pt 等
└── main.py                        # 可選：一鍵運行訓練的入口腳本
    # e.g. python main.py --config config.yaml
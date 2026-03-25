# 04_evaluation.ipynb

## 1. 載入最佳模型 & 資料
- 從 experiments/exp_001_baseline/model_best.pt 載入
- 用 src/data.py（之後會建）或直接用 notebook 的 dataloader 載 test set

## 2. 基本指標計算
- 在 test set 上 inference → 得到所有節點的 anomaly score（通常是 reconstruction error 或 classification logit）
- 算 AUC-ROC, AUPRC, F1 (threshold tuning), Precision@K (K=100,500,...)

## 3. 畫圖（報告用）
- Loss curve（從 training log 讀）
- ROC / PR curve
- Anomaly score 分布 histogram（licit vs illicit）
- t-SNE / UMAP 投影（embedding 可視化，看 illicit 是否分離）

## 4. 錯誤案例分析（很重要，FYP 加分項）
- 找出 Top-50 highest score 但其實是 licit 的 FP 案例
- 找出 lowest score 但其實是 illicit 的 FN 案例
- 看這些交易的特徵、鄰居結構、時間步 → 寫 3–5 句觀察

## 5. 下一步實驗想法（可留白）
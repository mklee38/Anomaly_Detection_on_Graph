# ANOMALY_DETECTION_ON_GRAPH

This repository contains code and experiments for graph anomaly detection on the Elliptic dataset.

Repository layout (top-level):

```
ANOMALY_DETECTION_ON_GRAPH/
├── data/
│   ├── external/                # external datasets (Elliptic, Ethereum, Yelp, etc.)
│   ├── processed/               # preprocessed .pt artifacts
│   └── raw/                     # original CSV / edgelist / features
├── experiments/                 # experiment runs, configs and results
├── notebooks/                   # exploration & prototyping notebooks
├── reports/                     # reports and figures
├── saved_models/                # best model checkpoints
├── src/                         # project source code
│   ├── config.py
│   ├── data.py
│   ├── evaluate.py
│   ├── models.py
│   ├── split.py
│   ├── train.py
│   ├── utils.py
│   └── __init__.py
├── environment.yml              # optional conda environment spec
├── requirements.txt             # pip install requirements
├── main.py                      # batch-run experiments
├── NEXT_STEP.md                 # development notes
└── README.md
```

Quick start
1. Create and activate a Python environment (recommended Python 3.10+).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Configure experiments in `experiments/config_experiments.py` (the file already contains example groups).

3. Run the batch runner (adjust `start_exp_no` and `force_reprocess` in `main.py` as needed):

```bash
python main.py
```

Notes
- The pipeline mode (`use_pipeline=True`) extracts GNN embeddings and trains an XGBoost classifier.
- `GraphSAGE` is the only model that uses the `aggregator` option; other models will record `N/A` in outputs.
- If you run into memory issues with LSTM aggregation, set `lstm_max_neighbors` in the experiment config to a small value (e.g., 4).

For development notes and next steps see `NEXT_STEP.md`.

```

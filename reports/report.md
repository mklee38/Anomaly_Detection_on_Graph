


### 4.2 Experimental Results - GraphSAGE Pipeline vs End-to-End

| Model Variant                          | Features                          | Mode       | Test AUC   | Test AUPRC |
|----------------------------------------|-----------------------------------|------------|------------|------------|
| GraphSAGE                              | Degree + PageRank                 | End-to-End | 0.9283     | 0.7271     |
| GraphSAGE + XGBoost                    | Degree + PageRank                 | Pipeline   | 0.9321     | 0.7985     |
| **GraphSAGE + XGBoost (concat)**       | **Degree + PageRank**             | **Pipeline** | **0.9321** | **0.7985** |
| **GraphSAGE + XGBoost (Best)**         | **Degree + PageRank + Eigenvector** | **Pipeline** | **0.9396** | **0.7990** |



The experimental results in Table 4.2 demonstrate the superiority of the proposed GraphSAGE Pipeline approach. By concatenating the original node features with GNN-learned embeddings, the Pipeline model achieved a substantial improvement in both Test AUC and AUPRC. The best performance was obtained when Eigenvector Centrality was additionally included, reaching **Test AUC = 0.9396** and **Test AUPRC = 0.7990**. Compared to the traditional End-to-End GraphSAGE, the Pipeline framework consistently delivered higher AUPRC, confirming its effectiveness for illicit account detection in large-scale transaction graphs.




### 4.2 Experimental Results - GraphSAGE Pipeline (Temporal Split)

| Exp.No | Experiment Name                          | Type     | Test AUC   | Test AUPRC | F1     | MCC    | hid_dim | #layer | lr    | epochs | patience | dropout | aggr  |
|--------|------------------------------------------|----------|------------|------------|--------|--------|---------|--------|-------|--------|----------|---------|-------|
| test   | sage_pipeline_baseline                   | Pipeline | 0.7412     | 0.2140     | NA     | NA     | 32      | 3      | 0.002 | 10     | 25       | 0.45    | mean  |
| 001    | sage_pipeline_concat                     | Pipeline | 0.9335     | 0.8000     | 0.8035 | 0.7974 | 32      | 3      | 0.002 | 10     | 25       | 0.45    | mean  |
| 002    | sage_pipeline_concat_deg_pr              | Pipeline | 0.9321     | 0.7985     | 0.8035 | 0.8035 | 32      | 3      | 0.002 | 10     | 25       | 0.45    | mean  |
| 003    | sage_pipeline_concat_deg_pr_clu          | Pipeline | 0.9368     | 0.7987     | 0.8004 | 0.8002 | 32      | 3      | 0.002 | 10     | 25       | 0.45    | mean  |
| 004    | sage_pipeline_concat_deg_pr_clu_eig      | Pipeline | 0.9313     | 0.7953     | 0.8057 | 0.7953 | 32      | 3      | 0.002 | 10     | 25       | 0.45    | mean  |
| **005**    | **sage_pipeline_concat_deg_pr_eig**      | **Pipeline** | **0.9409** | **0.8031** | 0.8017 | 0.8017 | 32      | 3      | 0.002 | 10     | 25       | 0.45    | mean  |
| 006    | sage_pipeline_concat_deg_pr_clu_eig      | Pipeline | 0.9371     | 0.8015     | 0.8033 | 0.7984 | 32      | 3      | 0.002 | 10     | 25       | 0.45    | mean  |
| 007    | sage_pipeline_concat_clu_eig             | Pipeline | 0.9352     | 0.7993     | 0.8033 | 0.7984 | 32      | 3      | 0.002 | 10     | 25       | 0.45    | mean  |
| 008    | sage_pipeline_concat_deg_pr              | Pipeline | 0.9341     | 0.7985     | **0.8157** | **0.8130** | 128     | 4      | 0.002 | 10     | 25       | 0.30    | mean  |
| 009    | sage_pipeline_concat_deg_pr              | Pipeline | 0.9365     | 0.7988     | 0.8130 | 0.8130 | 128     | 4      | 0.002 | 10     | 25       | 0.30    | mean  |
| 010    | sage_pipeline_concat_deg_pr              | Pipeline | 0.9365     | 0.7988     | 0.8130 | 0.8130 | 128     | 4      | 0.002 | 10     | 25       | 0.30    | mean  |

**Note**: All experiments use temporal split (train: time steps 1–34, test: 35–49). **Best performance** is highlighted in bold.
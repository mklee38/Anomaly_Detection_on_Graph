


### 4.2 Experimental Results - GraphSAGE Pipeline vs End-to-End

| Model Variant                          | Features                          | Mode       | Test AUC   | Test AUPRC |
|----------------------------------------|-----------------------------------|------------|------------|------------|
| GraphSAGE                              | Degree + PageRank                 | End-to-End | 0.9283     | 0.7271     |
| GraphSAGE + XGBoost                    | Degree + PageRank                 | Pipeline   | 0.9321     | 0.7985     |
| **GraphSAGE + XGBoost (concat)**       | **Degree + PageRank**             | **Pipeline** | **0.9321** | **0.7985** |
| **GraphSAGE + XGBoost (Best)**         | **Degree + PageRank + Eigenvector** | **Pipeline** | **0.9396** | **0.7990** |



The experimental results in Table 4.2 demonstrate the superiority of the proposed GraphSAGE Pipeline approach. By concatenating the original node features with GNN-learned embeddings, the Pipeline model achieved a substantial improvement in both Test AUC and AUPRC. The best performance was obtained when Eigenvector Centrality was additionally included, reaching **Test AUC = 0.9396** and **Test AUPRC = 0.7990**. Compared to the traditional End-to-End GraphSAGE, the Pipeline framework consistently delivered higher AUPRC, confirming its effectiveness for illicit account detection in large-scale transaction graphs.
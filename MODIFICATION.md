修改1：数据拆分 - 两阶段拆分：
- 先拆 80%/20% → train_val 和 test
- 再将 train_val 拆 80%/20% → train 和 val
- 最终比例：Train ~64% | Val ~16% | Test 20%

修改2：训练循环 - 每10个epoch执行验证：
- 计算训练loss、验证loss、验证AUC
- 实现early stopping：验证AUC连续5次没有改善，就停止训练
- 自动保存和恢复最佳模型

修改3：最终评估 - 分别报告两个数据集的指标：
- 验证集的 AUC-ROC 和 AUPRC
- 测试集的 AUC-ROC 和 AUPRC（用于最终模型性能评估）

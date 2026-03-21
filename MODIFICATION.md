(1) 添加了验证集和early stopping机制。修改内容如下：
    
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



(2) 添加了loss曲线绘制功能, 修改内容如下：
    
    修改1：训练循环改进 - 记录loss数据：
    - 每5个epoch记录一次training loss（避免数据过多）
    - 保存到 train_losses 和 epochs_list 列表
    - 训练完成后打印总epoch数

    新增：Loss 可视化cell - 画出两个图表：
    - 左圖：Training Loss 曲线（带数据点标记）
    - 右圖：收敛趋势图
    - 显示统计信息：初始loss、最终loss、下降幅度和百分比

    现在运行训练cell后，可以直接看到loss是否平滑收敛。如果曲线抖动明显或不下降，说明可能需要调整学习率或网络结构。
#!/bin/bash
# ========================================
# 运行训练脚本
# ========================================

set -e

# 激活环境
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate triplet

# 进入项目目录
cd /root/triplet_deploy/triplet_network

echo "=========================================="
echo "开始训练"
echo "=========================================="
echo "Margin: 0.5"
echo "Epochs: 100"
echo "数据集: /root/triplet_deploy/data"
echo "=========================================="

# 运行训练
python train.py

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
echo "模型保存位置: checkpoints/best_model.pth"
echo "训练曲线: checkpoints/training_curves.png"
echo "=========================================="

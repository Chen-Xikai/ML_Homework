#!/bin/bash
# ========================================
# 运行消融实验脚本
# ========================================

set -e

# 激活环境
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate triplet

# 进入项目目录
cd /root/triplet_deploy/triplet_network

echo "=========================================="
echo "开始消融实验"
echo "=========================================="
echo "Margin值: 0.2, 0.5, 1.0, 1.5, 2.0"
echo "每个实验: 5 epochs"
echo "数据集: /root/triplet_deploy/data"
echo "=========================================="

# 运行消融实验
python ablation_margin.py

echo ""
echo "=========================================="
echo "消融实验完成！"
echo "=========================================="
echo "结果文件: ablation_margin_results.csv"
echo "对比图表: ablation_margin_comparison.png"
echo "=========================================="

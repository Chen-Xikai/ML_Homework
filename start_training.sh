#!/bin/bash
set -e

echo "=== 步骤1: 修改 ablation_margin.py ==="
sed -i "s/'num_epochs': 5/'num_epochs': 10/" /root/triplet_deploy/triplet_network/ablation_margin.py
echo "验证修改:"
grep "num_epochs" /root/triplet_deploy/triplet_network/ablation_margin.py

echo ""
echo "=== 步骤2: 启动训练任务 ==="

# 先杀掉之前的训练进程
pkill -f "train.py" 2>/dev/null || true
sleep 1

# GPU 0: 完整训练 (100 epochs)
nohup env CUDA_VISIBLE_DEVICES=0 /root/triplet_deploy/venv/bin/python /root/triplet_deploy/triplet_network/train.py > /root/triplet_deploy/train_gpu0.log 2>&1 &
echo "train.py 已启动, PID: $!"

# GPU 1: 消融实验 (5 x 10 epochs)
nohup env CUDA_VISIBLE_DEVICES=1 /root/triplet_deploy/venv/bin/python /root/triplet_deploy/triplet_network/ablation_margin.py > /root/triplet_deploy/ablation_gpu1.log 2>&1 &
echo "ablation_margin.py 已启动, PID: $!"

echo ""
echo "=== 步骤3: 验证进程 ==="
sleep 2
ps aux | grep python | grep -v grep

echo ""
echo "两个任务已同时启动!"
echo "- train.py -> GPU 0 -> train_gpu0.log"
echo "- ablation_margin.py -> GPU 1 -> ablation_gpu1.log"

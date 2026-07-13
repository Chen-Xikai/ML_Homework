#!/bin/bash
cd /root/siamese_project

echo "=== Installing dependencies ==="
pip install torch torchvision numpy pillow scikit-learn matplotlib --break-system-packages -q 2>&1 | tail -3

echo "=== Checking environment ==="
python3 -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('GPU count:', torch.cuda.device_count())
"

echo "=== Starting training ==="
nohup python3 cloud_train.py > train_log.txt 2>&1 &
echo "Training PID: $!"
echo "Monitor: tail -f /root/siamese_project/train_log.txt"

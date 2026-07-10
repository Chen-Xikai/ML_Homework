#!/bin/bash
set -e
VENV=/root/triplet_deploy/venv

echo "[1/4] 升级pip..."
$VENV/bin/pip install --upgrade pip -q
echo "pip升级完成"

echo "[2/4] 安装PyTorch (CUDA 11.8)..."
$VENV/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
echo "PyTorch安装完成"

echo "[3/4] 安装其他依赖..."
$VENV/bin/pip install matplotlib pillow numpy -q
echo "其他依赖安装完成"

echo "[4/4] 验证环境..."
$VENV/bin/python -c 'import torch; print("PyTorch:", torch.__version__); print("CUDA:", torch.cuda.is_available())'
echo "环境配置完成!"

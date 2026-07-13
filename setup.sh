#!/bin/bash
# ========================================
# Triplet Network 环境配置脚本
# 适用于 Ubuntu 20.04/22.04 LTS
# ========================================

set -e

echo "=========================================="
echo "Triplet Network 环境配置"
echo "=========================================="

# 1. 更新系统
echo ""
echo "[1/5] 更新系统包..."
sudo apt update
sudo apt upgrade -y

# 2. 安装 NVIDIA 驱动
echo ""
echo "[2/5] 安装 NVIDIA 驱动..."
sudo apt install -y nvidia-driver-470
echo "NVIDIA 驱动安装完成，请运行 'sudo reboot' 重启服务器"
echo "重启后重新运行此脚本继续安装"
echo ""
read -p "是否现在重启？(y/n): " reboot_choice
if [ "$reboot_choice" = "y" ]; then
    sudo reboot
fi

# 3. 安装 Miniconda
echo ""
echo "[3/5] 安装 Miniconda..."
if [ ! -d "$HOME/miniconda3" ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p $HOME/miniconda3
    rm /tmp/miniconda.sh
    echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
fi
source ~/.bashrc

# 4. 创建 Python 环境
echo ""
echo "[4/5] 创建 Python 环境..."
$HOME/miniconda3/bin/conda create -n triplet python=3.11 -y
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate triplet

# 5. 安装 PyTorch 和依赖
echo ""
echo "[5/5] 安装 PyTorch 和依赖..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install matplotlib pillow numpy

echo ""
echo "=========================================="
echo "环境配置完成！"
echo "=========================================="
echo ""
echo "使用方法："
echo "  conda activate triplet"
echo "  cd /root/triplet_deploy/triplet_network"
echo "  python train.py"
echo ""
echo "或使用快捷脚本："
echo "  bash run_train.sh"
echo "=========================================="

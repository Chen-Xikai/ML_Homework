#!/bin/bash
# ========================================
# 下载 Omniglot 数据集
# ========================================

set -e

DATA_DIR="/root/triplet_deploy/data"

echo "=========================================="
echo "下载 Omniglot 数据集"
echo "=========================================="

# 创建数据目录
mkdir -p $DATA_DIR
cd $DATA_DIR

# 下载数据集
echo "正在下载 Omniglot 数据集..."
wget -q https://github.com/brendenlake/omniglot/raw/master/python/omniglot_train.zip -O omniglot_train.zip
wget -q https://github.com/brendenlake/omniglot/raw/master/python/omniglot_eval.zip -O omniglot_eval.zip

# 解压
echo "正在解压..."
unzip -q omniglot_train.zip
unzip -q omniglot_eval.zip

# 重命名目录
mv images_background images_background 2>/dev/null || true
mv images_evaluation images_evaluation 2>/dev/null || true

# 清理压缩包
rm -f omniglot_train.zip omniglot_eval.zip

echo ""
echo "=========================================="
echo "数据集下载完成！"
echo "=========================================="
echo "数据位置: $DATA_DIR"
echo "目录结构:"
ls -la $DATA_DIR
echo "=========================================="

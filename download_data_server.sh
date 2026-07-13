#!/bin/bash
set -e

echo "正在下载Omniglot数据集..."
cd /root/triplet_deploy/data

wget -q https://github.com/brendenlake/omniglot/raw/master/python/omniglot_train.zip -O omniglot_train.zip
wget -q https://github.com/brendenlake/omniglot/raw/master/python/omniglot_eval.zip -O omniglot_eval.zip

echo "正在解压..."
unzip -q -o omniglot_train.zip
unzip -q -o omniglot_eval.zip

rm -f omniglot_train.zip omniglot_eval.zip

echo "数据集下载完成!"
ls -la /root/triplet_deploy/data/

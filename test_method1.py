"""
test_method1.py - 方法1测试脚本

使用训练好的特征进行K=1最近邻分类测试。
"""

import os
import sys

import torch
from tqdm import tqdm

from config import Config


def knn_classify(val_features, val_labels, test_features, test_labels, k=1):
    """
    K最近邻分类
    
    参数:
        val_features: (N_val, D) 验证集特征
        val_labels: (N_val,) 验证集标签
        test_features: (N_test, D) 测试集特征
        test_labels: (N_test,) 测试集标签
        k: K值
    
    返回:
        accuracy: 分类准确率
        predictions: 预测标签
    """
    correct = 0
    total = len(test_labels)
    predictions = []
    
    for i in tqdm(range(total), desc="KNN分类", leave=False):
        # 计算与所有验证集特征的L2距离
        distances = torch.norm(val_features - test_features[i], dim=1)
        
        # 找最近的k个
        _, topk_indices = distances.topk(k, largest=False)
        topk_labels = val_labels[topk_indices]
        
        # 多数投票
        pred = topk_labels.mode().values.item()
        predictions.append(pred)
        
        if pred == test_labels[i].item():
            correct += 1
    
    accuracy = correct / total
    return accuracy, predictions


def test_method1(config):
    """方法1测试流程"""
    print("\n" + "=" * 60)
    print("[方法1] 最近邻分类测试")
    print("=" * 60)
    
    # 检查特征文件是否存在
    features_path = os.path.join(config.CHECKPOINT_DIR, "method1_features.pth")
    if not os.path.exists(features_path):
        print(f"[FAIL] 特征文件不存在: {features_path}")
        print("  请先运行 train_method1.py")
        return None
    
    # 加载特征
    print("\n加载特征...")
    data = torch.load(features_path)
    val_features = data["val_features"]
    val_labels = data["val_labels"]
    test_features = data["test_features"]
    test_labels = data["test_labels"]
    
    print(f"  验证集特征: {val_features.shape}")
    print(f"  测试集特征: {test_features.shape}")
    
    # KNN分类
    print("\n执行K=1最近邻分类...")
    accuracy, predictions = knn_classify(
        val_features, val_labels,
        test_features, test_labels,
        k=config.KNN_K
    )
    
    print(f"\n测试结果:")
    print(f"  方法: K={config.KNN_K} 最近邻")
    print(f"  测试准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # 保存结果
    result_path = os.path.join(config.RESULT_DIR, "method1_result.txt")
    with open(result_path, "w") as f:
        f.write(f"方法: K={config.KNN_K} 最近邻\n")
        f.write(f"测试准确率: {accuracy:.4f}\n")
        f.write(f"验证集样本数: {len(val_labels)}\n")
        f.write(f"测试集样本数: {len(test_labels)}\n")
    print(f"\n结果已保存: {result_path}")
    
    return accuracy


if __name__ == "__main__":
    config = Config()
    config.print_config()
    test_method1(config)

"""
test_method2.py - 方法2测试脚本

加载训练好的分类模型，在测试集上评估。
"""

import os
import sys

import torch
from tqdm import tqdm

from config import Config
from dataset import load_data_method2, get_data_loader
from model import ClassificationNet


@torch.no_grad()
def evaluate(model, loader, device):
    """评估"""
    model.eval()
    correct = 0
    total = 0
    
    for imgs, labels in tqdm(loader, desc="评估", leave=False):
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        outputs = model(imgs)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    accuracy = correct / total
    return accuracy


def test_method2(config):
    """方法2测试流程"""
    print("\n" + "=" * 60)
    print("[方法2] 端到端分类测试")
    print("=" * 60)
    
    device = torch.device(config.DEVICE)
    
    # 检查模型文件是否存在
    model_path = os.path.join(config.CHECKPOINT_DIR, "method2_model.pth")
    if not os.path.exists(model_path):
        print(f"[FAIL] 模型文件不存在: {model_path}")
        print("  请先运行 train_method2.py")
        return None
    
    # 加载数据
    print("\n加载数据...")
    (train_ds, val_ds, test_ds), char_to_idx = load_data_method2(config.DATA_DIR, config.RANDOM_SEED)
    test_loader = get_data_loader(test_ds, config.BATCH_SIZE, shuffle=False)
    
    # 更新类别数
    config.NUM_CLASSES = len(char_to_idx)
    
    # 加载模型
    print("\n加载模型...")
    model = ClassificationNet(config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    
    # 评估
    print("\n在测试集上评估...")
    test_acc = evaluate(model, test_loader, device)
    
    print(f"\n测试结果:")
    print(f"  方法: 端到端分类")
    print(f"  测试准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # 保存结果
    result_path = os.path.join(config.RESULT_DIR, "method2_result.txt")
    with open(result_path, "w") as f:
        f.write(f"方法: 端到端分类\n")
        f.write(f"测试准确率: {test_acc:.4f}\n")
        f.write(f"测试集样本数: {len(test_ds)}\n")
    print(f"\n结果已保存: {result_path}")
    
    return test_acc


if __name__ == "__main__":
    config = Config()
    config.print_config()
    test_method2(config)

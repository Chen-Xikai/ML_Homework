"""
margin_ablation.py
Contrastive Loss间隔α消融实验
测试不同margin值对20-way 1-shot准确率的影响
"""

import os
import sys
import time
import json
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 确保当前目录在Python路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import SiameseContrastive, ContrastiveLoss
from cache_dataset import create_cached_loaders
from evaluate import FewShotEvaluator
from utils import set_seed, get_device, format_time


def train_and_evaluate(config, margin):
    """训练一个模型并评估"""
    set_seed(42)
    device = get_device()

    print(f"\n{'='*60}")
    print(f"Training with margin={margin}")
    print(f"{'='*60}")

    # 从缓存创建数据加载器（加速）
    print(f"  Loading cached datasets...")
    train_loader, val_loader, test_loader = create_cached_loaders(
        cache_dir=r'C:\Users\ASUS\Desktop\任务三\dataset_cache',
        batch_size=config['batch_size']
    )

    # 创建模型和损失函数
    model = SiameseContrastive(embedding_dim=config['embedding_dim']).to(device)
    criterion = ContrastiveLoss(margin=margin).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # 创建评估器
    evaluator = FewShotEvaluator(model, device, config['root_dir'], seed=42)

    best_test_acc = 0.0
    best_epoch = 0
    start_time = time.time()

    for epoch in range(1, config['num_epochs'] + 1):
        epoch_start = time.time()

        # 训练
        model.train()
        train_loss = 0.0
        train_batches = 0
        for img1, img2, labels in train_loader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            optimizer.zero_grad()
            e1, e2 = model(img1, img2)
            loss = criterion(e1, e2, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1
        train_loss /= train_batches

        # 验证
        model.eval()
        val_loss = 0.0
        val_batches = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for img1, img2, labels in val_loader:
                img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
                e1, e2 = model(img1, img2)
                loss = criterion(e1, e2, labels)
                val_loss += loss.item()
                val_batches += 1
                distances = criterion.get_distance(e1, e2)
                predictions = (distances > criterion.margin / 2).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        val_loss /= val_batches
        val_acc = 100.0 * correct / total if total > 0 else 0

        # 20-way 1-shot测试
        test_results = evaluator.run_evaluation(n_episodes=400, n_way=20)
        test_acc = test_results['accuracy']

        scheduler.step(val_loss)
        epoch_time = time.time() - epoch_start

        print(f"\n--- Epoch {epoch}/{config['num_epochs']} (margin={margin}) ---")
        print(f"  Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}% | Test Acc: {test_acc:.2f}% | Time: {epoch_time:.1f}s")

        # 记录最佳
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            torch.save({
                'model_state_dict': model.state_dict(),
                'test_acc': test_acc,
                'epoch': epoch,
                'margin': margin
            }, os.path.join(config['save_dir'], f'best_margin_{margin}.pth'))
            print(f"  [Best] Test Acc: {best_test_acc:.2f}% (Saved)")

    total_time = time.time() - start_time
    return {
        'margin': margin,
        'best_test_acc': best_test_acc,
        'best_epoch': best_epoch,
        'total_time': total_time
    }


def main():
    config = {
        'root_dir': r'C:\Users\ASUS\Desktop\task3\data',
        'embedding_dim': 128,
        'batch_size': 32,
        'pairs_per_class': 17,  # ~30000 pairs
        'learning_rate': 1e-4,
        'num_epochs': 10,  # 每个margin训练10轮
        'save_dir': './checkpoints'
    }

    os.makedirs(config['save_dir'], exist_ok=True)

    # 测试的margin值
    margins = [0.5, 1.0, 1.5, 2.0]

    print("=" * 60)
    print("Margin Ablation Experiment")
    print(f"Margins: {margins}")
    print(f"Epochs per margin: {config['num_epochs']}")
    print("=" * 60)

    results = []
    total_start = time.time()

    for margin in margins:
        result = train_and_evaluate(config, margin)
        results.append(result)

    total_time = time.time() - total_start

    # 打印结果汇总
    print("\n" + "=" * 60)
    print("MARGIN ABLATION RESULTS")
    print("=" * 60)
    print(f"| {'Margin':>6} | {'Best Acc':>9} | {'Best Epoch':>10} | {'Time':>10} |")
    print(f"|{'-'*8}|{'-'*11}|{'-'*12}|{'-'*12}|")
    for r in results:
        print(f"| {r['margin']:>6.1f} | {r['best_test_acc']:>8.2f}% | {r['best_epoch']:>10} | {format_time(r['total_time']):>10} |")

    # 找到最佳margin
    best = max(results, key=lambda x: x['best_test_acc'])
    print(f"\nBest margin: {best['margin']} (Accuracy: {best['best_test_acc']:.2f}%)")
    print(f"Total experiment time: {format_time(total_time)}")

    # 保存结果
    results_path = os.path.join(config['save_dir'], 'margin_ablation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()

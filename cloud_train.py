"""
cloud_train.py
云服务器训练脚本
margin=0.5, 200 epochs, early_stopping_patience=20
"""

import os
import sys
import time
import json
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import SiameseContrastive, ContrastiveLoss
from cache_dataset import create_cached_loaders
from evaluate import FewShotEvaluator
from utils import set_seed, get_device, format_time


class EarlyStopping:
    def __init__(self, patience=20, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            print(f"  EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.early_stop


def train_model(config):
    set_seed(42)
    device = get_device()

    print(f"\n{'='*60}")
    print(f"Cloud Training - margin={config['margin']}")
    print(f"Max epochs: {config['num_epochs']}, Early stop: {config['early_stopping_patience']}")
    print(f"{'='*60}")

    # 加载缓存数据
    print("\nLoading cached datasets...")
    train_loader, val_loader, test_loader = create_cached_loaders(
        cache_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset_cache'),
        batch_size=config['batch_size']
    )

    # 创建模型
    model = SiameseContrastive(embedding_dim=config['embedding_dim']).to(device)
    criterion = ContrastiveLoss(margin=config['margin']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 创建评估器
    evaluator = FewShotEvaluator(model, device, config['root_dir'], seed=42)

    # 早停
    early_stopping = EarlyStopping(patience=config['early_stopping_patience'])

    # 历史记录
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'test_acc': []}
    best_test_acc = 0.0
    best_epoch = 0
    start_time = time.time()

    for epoch in range(1, config['num_epochs'] + 1):
        epoch_start = time.time()

        # 训练
        model.train()
        train_loss = 0.0
        for img1, img2, labels in train_loader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            optimizer.zero_grad()
            e1, e2 = model(img1, img2)
            loss = criterion(e1, e2, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # 验证
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for img1, img2, labels in val_loader:
                img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
                e1, e2 = model(img1, img2)
                loss = criterion(e1, e2, labels)
                val_loss += loss.item()
                distances = criterion.get_distance(e1, e2)
                predictions = (distances > criterion.margin / 2).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        val_loss /= len(val_loader)
        val_acc = 100.0 * correct / total if total > 0 else 0

        # 20-way 1-shot测试
        test_results = evaluator.run_evaluation(n_episodes=400, n_way=20)
        test_acc = test_results['accuracy']

        scheduler.step(val_loss)
        epoch_time = time.time() - epoch_start

        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['test_acc'].append(test_acc)

        # 打印进度
        total_elapsed = time.time() - start_time
        eta = total_elapsed / epoch * (config['num_epochs'] - epoch)
        print(f"\nEpoch {epoch}/{config['num_epochs']} ({epoch_time:.0f}s, ETA: {format_time(eta)})")
        print(f"  Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}% | Test Acc: {test_acc:.2f}%")

        # 保存最佳模型
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            torch.save({
                'model_state_dict': model.state_dict(),
                'test_acc': test_acc,
                'epoch': epoch,
                'margin': config['margin'],
                'config': config
            }, os.path.join(config['save_dir'], 'best_model.pth'))
            print(f"  [Best] Test Acc: {best_test_acc:.2f}% (Saved)")

        # 保存最新模型
        torch.save({
            'model_state_dict': model.state_dict(),
            'test_acc': test_acc,
            'epoch': epoch,
            'margin': config['margin']
        }, os.path.join(config['save_dir'], 'latest_model.pth'))

        # 早停检查
        if early_stopping(test_acc):
            print(f"\nEarly stopping at Epoch {epoch}")
            break

    total_time = time.time() - start_time

    # 保存历史
    with open(os.path.join(config['save_dir'], 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # 最终结果
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"Best Test Acc: {best_test_acc:.2f}% at Epoch {best_epoch}")
    print(f"Total Time: {format_time(total_time)}")
    print(f"Total Epochs: {epoch}")
    print(f"{'='*60}")

    return model, history


def main():
    # 自动检测数据路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    config = {
        'root_dir': os.path.join(script_dir, 'data'),
        'embedding_dim': 128,
        'margin': 0.5,
        'batch_size': 32,
        'pairs_per_class': 17,
        'learning_rate': 1e-4,
        'num_epochs': 200,
        'early_stopping_patience': 20,
        'save_dir': os.path.join(script_dir, 'checkpoints')
    }

    os.makedirs(config['save_dir'], exist_ok=True)

    print("=" * 60)
    print("Cloud Training Script")
    print(f"Config: margin={config['margin']}, epochs={config['num_epochs']}, "
          f"early_stop={config['early_stopping_patience']}")
    print("=" * 60)

    model, history = train_model(config)


if __name__ == "__main__":
    main()

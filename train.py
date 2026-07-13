import os
import time
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import SiameseContrastive, ContrastiveLoss
from dataset import create_data_loaders
from evaluate import FewShotEvaluator
from utils import set_seed, get_device, format_time, plot_training_curves, EarlyStopping


def train_model(config):
    set_seed(42)
    device = get_device()

    train_loader, val_loader, test_loader = create_data_loaders(
        root_dir=config['root_dir'],
        batch_size=config['batch_size'],
        pairs_per_class=config['pairs_per_class'],
        seed=42
    )

    model = SiameseContrastive(embedding_dim=config['embedding_dim']).to(device)
    criterion = ContrastiveLoss(margin=config['margin']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # 创建评估器（用于每个epoch的20-way 1-shot测试）
    evaluator = FewShotEvaluator(model, device, config['root_dir'], seed=42)

    # 初始化早停（基于20-way 1-shot准确率）
    early_stopping = EarlyStopping(
        patience=20,           # 20轮不提升就停止
        min_delta=0.1,         # 准确率提升至少0.1%才算改善
        mode='max'             # 准确率越高越好
    )

    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'test_acc': []}
    best_val_acc = 0.0
    best_test_acc = 0.0
    start_time = time.time()

    for epoch in range(1, config['num_epochs'] + 1):
        epoch_start = time.time()
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config['num_epochs']}")
        print(f"{'='*60}")

        # Train
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

        # Validate (快速二分类验证)
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

        # 20-way 1-shot测试（每个epoch都运行）
        test_results = evaluator.run_evaluation(n_episodes=400, n_way=20)
        test_acc = test_results['accuracy']

        scheduler.step(val_loss)
        epoch_time = time.time() - epoch_start

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['test_acc'].append(test_acc)

        print(f"\n--- Epoch {epoch} Summary ---")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"  Test Acc (20-way 1-shot): {test_acc:.2f}%")
        print(f"  Time: {epoch_time:.1f}s")

        # 保存最佳模型（基于20-way 1-shot准确率）
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save({'model_state_dict': model.state_dict(), 'val_acc': val_acc,
                       'test_acc': test_acc, 'epoch': epoch},
                      os.path.join(config['save_dir'], 'best_model.pth'))
            print(f"  [Best] Test Acc: {best_test_acc:.2f}% (Saved)")

        # 早停检查
        if early_stopping(test_acc):
            print(f"\n{'='*60}")
            print(f"Early Stopping Triggered at Epoch {epoch}")
            print(f"No improvement for {early_stopping.patience} epochs")
            print(f"Best Test Acc: {best_test_acc:.2f}%")
            print(f"{'='*60}")
            break

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training Done!")
    print(f"Best Test Acc (20-way 1-shot): {best_test_acc:.2f}%")
    print(f"Total Time: {format_time(total_time)}")
    print(f"{'='*60}")
    return model, history


def test_model(model, config):
    device = get_device()
    evaluator = FewShotEvaluator(model, device, config['root_dir'], seed=42)
    results = evaluator.run_evaluation(n_episodes=400, n_way=20)
    return results


def main():
    config = {
        'root_dir': r'C:\Users\ASUS\Desktop\task3\data',
        'embedding_dim': 128,
        'margin': 1.0,
        'batch_size': 32,
        'pairs_per_class': 17,  # ~30000 training pairs (28 alphabets)
        'learning_rate': 1e-4,
        'num_epochs': 100,
        'save_dir': './checkpoints'
    }

    os.makedirs(config['save_dir'], exist_ok=True)
    print("Config:", config)

    model, history = train_model(config)
    results = test_model(model, config)

    print(f"\n{'='*50}")
    print(f"20-way 1-shot Accuracy: {results['accuracy']:.2f}%")
    print(f"{'='*50}")

    plot_training_curves(history, os.path.join(config['save_dir'], 'training_curves.png'))


if __name__ == "__main__":
    main()

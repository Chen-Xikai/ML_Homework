"""
train_method1.py - 方法1训练脚本

训练分类网络，然后提取特征用于最近邻分类。
数据划分：原始one-shot划分（30字母表训练，10验证，10测试）
"""

import os
import sys
import time
import csv
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config import Config
from dataset import load_data_method2, get_data_loader
from model import ClassificationNet, init_weights


class EarlyStopping:
    """早停管理器"""
    
    def __init__(self, patience=20, max_epochs=200):
        self.patience = patience
        self.max_epochs = max_epochs
        self.best_acc = 0.0
        self.best_epoch = 0
        self.counter = 0
        self.best_model_state = None
        self.stop_reason = ""
    
    def step(self, val_acc, epoch, model):
        should_stop = False
        
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            self.best_epoch = epoch
            self.counter = 0
            self.best_model_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            should_stop = True
            self.stop_reason = f"早停：连续{self.patience}轮未提升（最优epoch {self.best_epoch}，准确率 {self.best_acc:.4f}）"
        elif epoch >= self.max_epochs:
            should_stop = True
            self.stop_reason = f"达到最大训练轮数 {self.max_epochs}"
        
        return should_stop, self.stop_reason
    
    def restore_best(self, model):
        """恢复历史最优模型"""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for imgs, labels in tqdm(loader, desc="训练", leave=False):
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * labels.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(model, loader, device):
    """验证"""
    model.eval()
    correct = 0
    total = 0
    
    for imgs, labels in tqdm(loader, desc="验证", leave=False):
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        outputs = model(imgs)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    accuracy = correct / total
    return accuracy


def train_method1(config):
    """方法1训练流程"""
    print("\n" + "=" * 60)
    print("[方法1] 训练分类网络 + 最近邻测试")
    print("  数据划分：按字符12:4:4划分（与方法2相同）")
    print("  区别：测试时使用K=1最近邻而非直接分类")
    print("=" * 60)
    
    config.ensure_dirs()
    device = torch.device(config.DEVICE)
    
    # 加载数据 - 使用与方法2相同的按字符12:4:4划分
    # 这样验证集和测试集共享相同类别，KNN才能有效工作
    (train_ds, val_ds, test_ds), char_to_idx = load_data_method2(config.DATA_DIR, config.RANDOM_SEED)
    train_loader = get_data_loader(train_ds, config.BATCH_SIZE, shuffle=True)
    val_loader = get_data_loader(val_ds, config.BATCH_SIZE, shuffle=False)
    
    # 更新类别数
    config.NUM_CLASSES = len(char_to_idx)
    
    # 构建模型
    model = ClassificationNet(config)
    init_weights(model, config)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {total_params:,}")
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    
    # 早停
    early_stopping = EarlyStopping(patience=config.PATIENCE, max_epochs=config.MAX_EPOCHS)
    
    # 训练日志
    log_path = os.path.join(config.LOG_DIR, "method1_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_acc", "lr"])
    
    # 训练循环
    print("\n开始训练...")
    start_time = time.time()
    
    for epoch in range(1, config.MAX_EPOCHS + 1):
        epoch_start = time.time()
        
        # 学习率衰减
        current_lr = config.LR * (config.LR_DECAY ** (epoch - 1))
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        
        # 训练
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # 验证
        val_acc = validate(model, val_loader, device)
        
        # 记录日志
        epoch_time = time.time() - epoch_start
        log_line = [epoch, f"{train_loss:.6f}", f"{train_acc:.4f}", f"{val_acc:.4f}", f"{current_lr:.6f}"]
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(log_line)
            f.flush()
        
        # 打印
        print(f"  Epoch {epoch:3d}/{config.MAX_EPOCHS} | "
              f"Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Acc: {val_acc:.4f} | LR: {current_lr:.6f} | Time: {epoch_time:.1f}s",
              flush=True)
        
        # 早停（基于验证准确率）
        should_stop, stop_reason = early_stopping.step(val_acc, epoch, model)
        if should_stop:
            print(f"\n  {stop_reason}")
            break
    
    total_time = time.time() - start_time
    print(f"\n训练完成，耗时 {total_time:.1f}秒")
    print(f"最优验证准确率: {early_stopping.best_acc:.4f}")
    
    # 恢复最优模型
    early_stopping.restore_best(model)
    
    # 保存模型
    model_path = os.path.join(config.CHECKPOINT_DIR, "method1_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存: {model_path}")
    
    # 保存特征用于最近邻测试
    print("\n提取特征用于最近邻测试...")
    extract_and_save_features(model, val_ds, test_ds, config, device)
    
    return early_stopping.best_acc, 0.0


@torch.no_grad()
def extract_and_save_features(model, val_ds, test_ds, config, device):
    """提取并保存特征"""
    model.eval()
    
    val_loader = get_data_loader(val_ds, config.BATCH_SIZE, shuffle=False)
    test_loader = get_data_loader(test_ds, config.BATCH_SIZE, shuffle=False)
    
    # 提取验证集特征
    val_features = []
    val_labels = []
    for imgs, labels in tqdm(val_loader, desc="提取验证集特征", leave=False):
        imgs = imgs.to(device)
        features = model.extract_features(imgs)
        val_features.append(features.cpu())
        val_labels.append(labels)
    
    val_features = torch.cat(val_features, dim=0)
    val_labels = torch.cat(val_labels, dim=0)
    
    # 提取测试集特征
    test_features = []
    test_labels = []
    for imgs, labels in tqdm(test_loader, desc="提取测试集特征", leave=False):
        imgs = imgs.to(device)
        features = model.extract_features(imgs)
        test_features.append(features.cpu())
        test_labels.append(labels)
    
    test_features = torch.cat(test_features, dim=0)
    test_labels = torch.cat(test_labels, dim=0)
    
    # 保存
    features_path = os.path.join(config.CHECKPOINT_DIR, "method1_features.pth")
    torch.save({
        "val_features": val_features,
        "val_labels": val_labels,
        "test_features": test_features,
        "test_labels": test_labels,
    }, features_path)
    print(f"特征已保存: {features_path}")
    print(f"  验证集特征: {val_features.shape}")
    print(f"  测试集特征: {test_features.shape}")


if __name__ == "__main__":
    config = Config()
    config.print_config()
    train_method1(config)

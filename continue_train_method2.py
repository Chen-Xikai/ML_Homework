"""
continue_train_method2.py - 方法2续训脚本

从已保存的method2_model.pth继续训练100 epochs。
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
from model import ClassificationNet


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


def continue_train_method2(config):
    """方法2续训流程"""
    RESUME_EPOCHS = 100
    START_EPOCH = 101
    MAX_EPOCH = START_EPOCH + RESUME_EPOCHS - 1  # 200

    print("\n" + "=" * 60)
    print("[方法2] 端到端分类续训 (epoch 101-200)")
    print("=" * 60)
    
    config.ensure_dirs()
    device = torch.device(config.DEVICE)
    
    # 加载数据
    (train_ds, val_ds, test_ds), char_to_idx = load_data_method2(config.DATA_DIR, config.RANDOM_SEED)
    train_loader = get_data_loader(train_ds, config.BATCH_SIZE, shuffle=True)
    val_loader = get_data_loader(val_ds, config.BATCH_SIZE, shuffle=False)
    test_loader = get_data_loader(test_ds, config.BATCH_SIZE, shuffle=False)
    
    config.NUM_CLASSES = len(char_to_idx)
    print(f"  最终类别数: {config.NUM_CLASSES}")
    
    # 构建并加载模型
    model = ClassificationNet(config)
    model_path = os.path.join(config.CHECKPOINT_DIR, "method2_model.pth")
    if not os.path.exists(model_path):
        print(f"  [ERROR] 模型文件不存在: {model_path}")
        return
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    print(f"  已加载模型: {model_path}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {total_params:,}")
    
    # 优化器（lr从衰减后位置开始）
    initial_lr = config.LR * (config.LR_DECAY ** (START_EPOCH - 1))
    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    
    # 早停
    early_stopping = EarlyStopping(patience=config.PATIENCE, max_epochs=MAX_EPOCH)
    
    # 续训日志（追加到已有文件）
    log_path = os.path.join(config.LOG_DIR, "method2_log.csv")
    
    # 续训循环
    print(f"\n开始续训 (epoch {START_EPOCH}-{MAX_EPOCH})...")
    start_time = time.time()
    
    for epoch in range(START_EPOCH, MAX_EPOCH + 1):
        epoch_start = time.time()
        
        # 学习率衰减（从START_EPOCH继续）
        current_lr = config.LR * (config.LR_DECAY ** (epoch - 1))
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        
        # 训练
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # 验证
        val_acc = validate(model, val_loader, device)
        
        # 记录日志（追加）
        epoch_time = time.time() - epoch_start
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.6f}", f"{train_acc:.4f}", f"{val_acc:.4f}", f"{current_lr:.6f}"])
            f.flush()
        
        # 打印
        print(f"  Epoch {epoch:3d}/{MAX_EPOCH} | "
              f"Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Acc: {val_acc:.4f} | LR: {current_lr:.6f} | "
              f"Time: {epoch_time:.1f}s",
              flush=True)
        
        # 早停
        should_stop, stop_reason = early_stopping.step(val_acc, epoch, model)
        if should_stop:
            print(f"\n  {stop_reason}")
            break
    
    total_time = time.time() - start_time
    print(f"\n续训完成，耗时 {total_time:.1f}秒")
    print(f"最优验证准确率: {early_stopping.best_acc:.4f}")
    
    # 恢复最优模型
    early_stopping.restore_best(model)
    
    # 在测试集上评估
    print("\n在测试集上评估...")
    test_acc = evaluate(model, test_loader, device)
    print(f"测试集准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # 保存模型
    model_resume_path = os.path.join(config.CHECKPOINT_DIR, "method2_model_resume.pth")
    torch.save(model.state_dict(), model_resume_path)
    print(f"模型已保存: {model_resume_path}")
    
    # 保存结果
    result_path = os.path.join(config.RESULT_DIR, "method2_resume_result.txt")
    with open(result_path, "w") as f:
        f.write(f"续训阶段: epoch {START_EPOCH}-{MAX_EPOCH}\n")
        f.write(f"验证集准确率: {early_stopping.best_acc:.4f}\n")
        f.write(f"测试集准确率: {test_acc:.4f}\n")
        f.write(f"最优轮数: {early_stopping.best_epoch}\n")
        f.write(f"续训耗时: {total_time:.1f}秒\n")
    
    return early_stopping.best_acc, test_acc


if __name__ == "__main__":
    config = Config()
    config.print_config()
    continue_train_method2(config)

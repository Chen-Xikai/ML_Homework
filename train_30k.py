"""
train_30k.py —— 单次训练脚本

执行一次完整的训练流程（论文原版早停机制）。

训练配置:
  - 30k 训练对，每 epoch 从字符池重新随机采样（扩大认知范围）
  - 无畸变（num_transforms=0）
  - 优化器: AdamW
  - 损失函数: Focal Loss（γ=2.0，内部自带 sigmoid）
  - LR 热身 5 轮 + 每 epoch 衰减 1%
  - 早停: 验证集 320 次单样本准确率连续 20 轮未提升
  - 最大 200 轮

返回训练结果字典供 main.py 使用。
"""

import os
import sys
import time
import csv
import copy
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from affine_augmentation import augment_batch_on_gpu


# ======================== 早停管理器 ========================

class EarlyStopping:
    """
    论文原版早停管理器（与 93% 无关）。

    判断标准:
      1. 验证集单样本准确率连续 patience=20 轮未提升 → 停止
      2. epoch 达到 max_epochs=200 → 强制停止

    停止时自动恢复历史最优模型权重。
    """

    def __init__(self, patience: int = 20, max_epochs: int = 200):
        self.patience = patience
        self.max_epochs = max_epochs
        self.best_acc = 0.0
        self.best_epoch = 0
        self.counter = 0
        self.best_model_state = None
        self.stop_reason = ""

    def step(self, val_acc: float, epoch: int,
             model: nn.Module, save_path: str) -> Tuple[bool, str]:
        """
        每个 epoch 结束后调用。

        参数:
            val_acc: 当前 epoch 的验证集单样本准确率
            epoch: 当前轮次
            model: 当前模型
            save_path: 最优模型的保存路径

        返回:
            (should_stop, stop_reason)
        """
        should_stop = False

        if val_acc > self.best_acc:
            # 有提升：保存最优模型，计数器归零
            self.best_acc = val_acc
            self.best_epoch = epoch
            self.counter = 0
            self.best_model_state = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), save_path)
        else:
            # 未提升：计数器 +1
            self.counter += 1

        # 判断条件 1：连续 patience 轮未提升
        if self.counter >= self.patience:
            should_stop = True
            self.stop_reason = f"早停：连续{self.patience}轮未提升（最优 epoch {self.best_epoch}，准确率 {self.best_acc:.4f}）"

        # 判断条件 2：达到最大轮数（只在未触发条件1时设置reason）
        elif epoch >= self.max_epochs:
            should_stop = True
            self.stop_reason = f"达到最大训练轮数 {self.max_epochs}"

        return should_stop, self.stop_reason

    def restore_best(self, model: nn.Module):
        """恢复历史最优模型权重"""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


# ======================== Focal Loss ========================

class FocalLoss(nn.Module):
    """
    Focal Loss: 自动给困难样本升权、容易样本降权。

    公式: FL = -α * (1 - p_t)^γ * log(p_t)

    γ=0 时退化为标准交叉熵，γ 越大困难样本权重越高。
    BCEWithLogitsLoss 内部做 sigmoid，数值更稳定。
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        """logits: 原始分数 (B,)，targets: 0/1 标签 (B,)"""
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


# ======================== 优化器构建 ========================

def build_optimizer(model: nn.Module,
                    config,
                    lr: float,
                    l2_lambda: float):
    """
    构建 AdamW 优化器。

    AdamW = Adam + 解耦权重衰减，收敛更快、对 lr 不敏感。
    """

    return optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=config.adam_betas,
        weight_decay=l2_lambda,
    )


# ======================== 单 epoch 训练 ========================

def train_one_epoch(model: nn.Module,
                    loader,
                    optimizer: optim.Optimizer,
                    criterion: nn.Module,
                    epoch: int,
                    config) -> Tuple[float, float, float]:
    """
    执行一个 epoch 的训练。

    返回:
        (avg_loss, avg_acc, elapsed_time_sec)
    """
    model.train()
    device = next(model.parameters()).device

    total_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()

    pbar = tqdm(loader, desc=f"Epoch {epoch:3d}", unit="batch", leave=False)
    for imgs1, imgs2, labels in pbar:
        imgs1 = imgs1.to(device, non_blocking=True)
        imgs2 = imgs2.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if config.num_transforms > 0:
            # GPU 批量仿射增强：1 原始 + N 增强 = (1+N) 倍数据
            imgs1, imgs2, labels = augment_batch_on_gpu(
                imgs1, imgs2, labels, config, num_copies=config.num_transforms,
            )
            # 梯度累积：分子 batch 处理，避免 FC 层 batch 过大
            B_orig = config.batch_size
            sub_total = imgs1.size(0)
            n_sub = sub_total // B_orig
            sub_loss_sum = 0.0
            sub_correct = 0
            optimizer.zero_grad()
            for j in range(n_sub):
                s = j * B_orig
                e = s + B_orig
                sub_out = model(imgs1[s:e], imgs2[s:e]).squeeze()
                sub_labels = labels[s:e]
                sub_loss = criterion(sub_out, sub_labels)
                (sub_loss / n_sub).backward()
                sub_loss_sum += sub_loss.item() * B_orig
                sub_preds = (sub_out >= 0.0).float()
                sub_correct += (sub_preds == sub_labels).sum().item()
            optimizer.step()
            total_loss += sub_loss_sum
            correct += sub_correct
            total += sub_total
            pbar.set_postfix({
                "loss": f"{sub_loss_sum/sub_total:.4f}",
                "acc": f"{correct/total:.3f}",
            })
        else:
            # 无增强：直接前向
            outputs = model(imgs1, imgs2).squeeze()
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            preds = (outputs >= 0.0).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{correct/total:.3f}",
            })

    elapsed = time.time() - start_time
    avg_loss = total_loss / total
    avg_acc = correct / total

    return avg_loss, avg_acc, elapsed


# ======================== 验证 ========================

@torch.no_grad()
def validate_binary(model: nn.Module, loader, config) -> float:
    """
    验证集 10k 对二分类准确率测试（仅作训练监控参考，不用于早停）。
    """
    model.eval()
    device = next(model.parameters()).device

    correct = 0
    total = 0

    for imgs1, imgs2, labels in loader:
        imgs1 = imgs1.to(device, non_blocking=True)
        imgs2 = imgs2.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(imgs1, imgs2).squeeze()
        preds = (outputs >= 0.0).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total


@torch.no_grad()
def validate_one_shot(model: nn.Module,
                      val_trials: list,
                      config) -> float:
    """
    验证集 320 次 20-way 单样本测试 —— 早停的唯一判断依据。

    val_trials 在训练开始时随机生成一次，所有 epoch 共用，
    消除每 epoch 重新采样带来的考题波动。
    """
    model.eval()
    device = next(model.parameters()).device

    correct = 0
    total = len(val_trials)

    for test_img, support_imgs, correct_idx in val_trials:
        # test_img 复制 n_way 份，与每个候选配对
        test_tensor = torch.from_numpy(
            np.array(test_img, dtype=np.float32) / 255.0
        ).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 105, 105)

        support_tensors = torch.stack([
            torch.from_numpy(
                np.array(img, dtype=np.float32) / 255.0
            ).unsqueeze(0)
            for img in support_imgs
        ]).to(device)  # (n_way, 1, 105, 105)

        # 复制测试图为 n_way 份
        test_batch = test_tensor.repeat(config.n_way, 1, 1, 1)

        scores = model(test_batch, support_tensors).squeeze()  # (n_way,)
        pred = scores.argmax().item()

        if pred == correct_idx:
            correct += 1

    return correct / total if total > 0 else 0.0


# ======================== 困难负样本重采样 ========================

@torch.no_grad()
def score_all_pairs(model, pair_list, device, config):
    """
    用当前模型对所有训练对打分，返回 logits 列表。
    batch 推理，不上梯度。
    """
    from torch.utils.data import DataLoader
    from omniglot_dataset import OmniglotPairDataset

    model.eval()
    temp_dataset = OmniglotPairDataset(pair_list, augmentation_fn=None, return_pil=False)
    temp_loader = DataLoader(temp_dataset, batch_size=config.batch_size,
                             shuffle=False, num_workers=0, pin_memory=True)

    all_logits = []
    for imgs1, imgs2, labels in temp_loader:
        imgs1 = imgs1.to(device, non_blocking=True)
        imgs2 = imgs2.to(device, non_blocking=True)
        logits = model(imgs1, imgs2).squeeze(-1)  # (B,)
        all_logits.append(logits.cpu())

    return torch.cat(all_logits, dim=0)  # (N,)


def resample_easy_pairs(pairs, all_logits, char_index, rng, config):
    """
    找出简单的样本对并替换为新随机对，困难的对保留继续训练。

    简单标准：
      - 正样本（label=1）：logits 高 = 模型很有把握 → 简单 → 替换
      - 负样本（label=0）：logits 低 = 模型很有把握 → 简单 → 替换

    困难标准：
      - 正样本：logits 低甚至负（模型犹豫"是同类吗？"）→ 困难 → 保留
      - 负样本：logits 高甚至正（模型犹豫"是不同类吗？"）→ 困难 → 保留
    """
    from omniglot_dataset import generate_pairs

    n_total = len(pairs)
    pos_indices = [i for i, (_, _, label) in enumerate(pairs) if label == 1]
    neg_indices = [i for i, (_, _, label) in enumerate(pairs) if label == 0]

    n_pos_replace = max(1, int(len(pos_indices) * config.hard_neg_ratio))
    n_neg_replace = max(1, int(len(neg_indices) * config.hard_neg_ratio))

    # 正样本：logits 最高 = 最简单 → 替换
    pos_logits = [(i, all_logits[i].item()) for i in pos_indices]
    pos_logits.sort(key=lambda x: x[1], reverse=True)
    easy_pos_indices = {idx for idx, _ in pos_logits[:n_pos_replace]}

    # 负样本：logits 最低 = 最简单 → 替换
    neg_logits = [(i, all_logits[i].item()) for i in neg_indices]
    neg_logits.sort(key=lambda x: x[1])
    easy_neg_indices = {idx for idx, _ in neg_logits[:n_neg_replace]}

    replace_indices = easy_pos_indices | easy_neg_indices

    # 生成替换用新对
    new_pairs = list(pairs)
    replaced_pos = 0
    replaced_neg = 0

    for i in sorted(replace_indices):
        _, _, label = pairs[i]
        if label == 1:
            # 生成新的正样本对
            new_pair = generate_pairs(char_index, 1, rng)
            new_pair = [(img1, img2, 1.0) for img1, img2, l in new_pair if l == 1]
            if new_pair:
                new_pairs[i] = new_pair[0]
                replaced_pos += 1
        else:
            new_pair = generate_pairs(char_index, 1, rng)
            new_pair = [(img1, img2, 0.0) for img1, img2, l in new_pair if l == 0]
            if new_pair:
                new_pairs[i] = new_pair[0]
                replaced_neg += 1

    print(f"  [Mine] 替换正样本 {replaced_pos}/{n_pos_replace}，负样本 {replaced_neg}/{n_neg_replace}")
    return new_pairs


# ======================== 学习率 & 动量调度 ========================

def adjust_learning_rate(optimizer: optim.Optimizer, config):
    """学习率统一衰减: η ← 0.99 × η（对所有参数组）"""
    for param_group in optimizer.param_groups:
        param_group["lr"] *= config.lr_decay_rate


def warmup_lr(optimizer: optim.Optimizer,
              epoch: int,
              config,
              base_lr: float):
    """
    LR 热身: 前 warmup_epochs 轮从 base_lr/10 线性增长到 base_lr，
    防止训练初期震荡。
    """
    if epoch <= config.warmup_epochs:
        scale = 0.1 + 0.9 * (epoch - 1) / config.warmup_epochs
        for param_group in optimizer.param_groups:
            param_group["lr"] = base_lr * scale


# ======================== 日志 ========================

def setup_logging(config, trial_id: int):
    """创建 epoch 级训练日志文件，写入表头"""
    log_path = config.get_log_path(trial_id)
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "train_acc",
            "val_binary_acc", "val_one_shot_acc",
            "learning_rate",
            "is_best", "elapsed_time",
        ])
    return log_path


def log_epoch(log_path: str, epoch: int, metrics: dict):
    """追加一行 epoch 记录"""
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch,
            f"{metrics['train_loss']:.6f}",
            f"{metrics['train_acc']:.6f}",
            f"{metrics['val_binary_acc']:.6f}",
            f"{metrics['val_one_shot_acc']:.6f}",
            f"{metrics['learning_rate']:.8f}",
            metrics["is_best"],
            f"{metrics['elapsed_time']:.2f}",
        ])


# ======================== 训练主函数 ========================

def train(config,
          trial_id: int,
          lr: float,
          l2_lambda: float,
          train_char_index: dict,
          val_char_index: dict,
          val_drawer_pool: set) -> Dict:
    """
    执行一次完整训练。

    参数:
        config: 配置实例
        trial_id: Optuna trial ID
        lr: 学习率（Optuna 建议）
        l2_lambda: L2 正则系数（Optuna 建议）
        train_char_index: 训练集字符索引
        val_char_index: 验证集字符索引
        val_drawer_pool: 验证集可用书写者集合

    返回:
        {
            "model_path":    最优模型文件路径,
            "best_val_acc":  训练中验证单样本最高准确率,
            "epochs_trained": 实际训练轮数,
            "total_time":    训练总耗时（秒）,
            "stop_reason":   停止原因,
        }
    """
    import random
    from omniglot_dataset import get_train_loader, get_val_loader, generate_one_shot_trials
    from siamese_model import SiameseNetwork, initialize_model, count_parameters

    device = torch.device(config.device)
    print(f"\n{'='*50}")
    print(f"[Start]  Trial {trial_id}: 开始训练")
    print(f"   超参数: lr={lr:.6f}, l2={l2_lambda:.6f}")
    print(f"   优化器: AdamW, Focal Loss (γ={config.focal_gamma})")
    print(f"   设备: {device}")
    print(f"{'='*50}")

    # ---- 准备数据 ----
    from omniglot_dataset import get_val_loader as gvl
    gvl._val_char_index = val_char_index

    val_loader = get_val_loader(config)
    print(f"  训练样本对: 每 epoch 从训练集字符池随机生成 {config.train_pairs:,} 对")

    # 训练开始时生成一组固定的验证 trial（所有 epoch 共用，消除采样波动）
    val_seed = config.random_seed + 1000
    val_rng = random.Random(val_seed)
    val_trials = generate_one_shot_trials(
        val_char_index, val_drawer_pool,
        config.n_way, config.val_one_shot_trials, val_rng,
    )
    print(f"  验证集固定 trial: {len(val_trials)} 次（seed={val_seed}）")

    # ---- 构建模型 ----
    model = SiameseNetwork(config)
    initialize_model(model, config)
    model = model.to(device)
    info = count_parameters(model)
    print(f"  模型参数量: {info['trainable']:,}")

    # ---- 优化器 ----
    optimizer = build_optimizer(model, config, lr, l2_lambda)

    # ---- 损失函数 ----
    # Focal Loss: 自动聚焦困难样本，内部自带 sigmoid（替代 BCELoss）
    criterion = FocalLoss(gamma=config.focal_gamma, alpha=config.focal_alpha)

    # ---- 早停 ----
    early_stopping = EarlyStopping(
        patience=config.early_stop_patience,
        max_epochs=config.max_epochs,
    )
    save_path = config.get_checkpoint_path(trial_id)

    # ---- 日志 ----
    log_path = setup_logging(config, trial_id)

    # ---- 训练循环 ----
    best_val_acc = 0.0
    total_start = time.time()
    stop_reason = ""
    epochs_trained = 0
    rng = random.Random(config.random_seed + 1)

    for epoch in range(1, config.max_epochs + 1):
        epoch_start = time.time()

        # 每 epoch 从训练集字符池重新随机采样 30k 对（扩大认知范围，防过拟合）
        from omniglot_dataset import OmniglotPairDataset, generate_pairs
        epoch_pairs = generate_pairs(train_char_index, config.train_pairs,
                                     random.Random(config.random_seed + epoch))
        train_loader = torch.utils.data.DataLoader(
            OmniglotPairDataset(epoch_pairs, augmentation_fn=None, return_pil=False),
            batch_size=config.batch_size, shuffle=True,
            num_workers=config.num_workers, pin_memory=config.pin_memory,
        )

        # 训练
        train_loss, train_acc, _ = train_one_epoch(
            model, train_loader, optimizer, criterion, epoch, config,
        )

        # LR 热身（前 N 轮逐渐增加到 base_lr）
        warmup_lr(optimizer, epoch, config, lr)

        # 学习率衰减
        adjust_learning_rate(optimizer, config)

        # 验证：二分类
        val_binary_acc = validate_binary(model, val_loader, config)

        # 验证：固定 320 次单样本（早停依据，所有 epoch 同一组题）
        val_one_shot_acc = validate_one_shot(model, val_trials, config)

        # 更新最优
        is_best = 1 if val_one_shot_acc > best_val_acc else 0
        if val_one_shot_acc > best_val_acc:
            best_val_acc = val_one_shot_acc

        # 日志记录
        epoch_elapsed = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]
        log_epoch(log_path, epoch, {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_binary_acc": val_binary_acc,
            "val_one_shot_acc": val_one_shot_acc,
            "learning_rate": current_lr,
            "is_best": is_best,
            "elapsed_time": epoch_elapsed,
        })

        # 终端输出
        best_marker = " *" if is_best else ""
        print(
            f"Epoch {epoch:3d}/{config.max_epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.3f} | "
            f"Val Bin: {val_binary_acc:.3f} | "
            f"Val 1-Shot: {val_one_shot_acc:.4f}{best_marker} | "
            f"LR: {current_lr:.6f}"
        )

        # 早停判断
        should_stop, stop_reason = early_stopping.step(
            val_one_shot_acc, epoch, model, save_path,
        )
        epochs_trained = epoch

        if should_stop:
            print(f"\n[STOP]  {stop_reason}")
            # 恢复最优模型
            early_stopping.restore_best(model)
            break

    total_time = time.time() - total_start
    hours = total_time / 3600

    print(f"\n{'='*50}")
    print(f"[OK]  Trial {trial_id} 训练完成")
    print(f"   验证最佳单样本准确率: {best_val_acc:.4f}")
    print(f"   训练轮数:             {epochs_trained}")
    print(f"   停止原因:             {stop_reason}")
    print(f"   总耗时:               {hours:.1f} 小时 ({total_time:.0f} 秒)")
    print(f"   模型保存至:           {save_path}")
    print(f"{'='*50}")

    return {
        "model_path": save_path,
        "best_val_acc": best_val_acc,
        "epochs_trained": epochs_trained,
        "total_time": total_time,
        "stop_reason": stop_reason,
    }


if __name__ == "__main__":
    # 独立运行单次训练（用于调试）
    sys.path.insert(0, ".")
    from config import Config
    from omniglot_dataset import (
        download_omniglot, load_raw_data,
        split_alphabets_drawers, build_char_index,
    )

    config = Config()
    config.ensure_dirs()

    print("独立训练模式（调试用）")
    download_omniglot(config.data_dir)
    raw_data = load_raw_data(config.data_dir)
    train_chars, val_chars, test_chars = split_alphabets_drawers(raw_data, config)
    train_ci = build_char_index(train_chars)
    val_ci = build_char_index(val_chars)
    val_drawers = set(d for _, _, d, _ in val_chars)

    result = train(
        config,
        trial_id=0,
        lr=0.01,
        l2_lambda=0.0005,
        train_char_index=train_ci,
        val_char_index=val_ci,
        val_drawer_pool=val_drawers,
    )
    print(f"\n最终结果: {result}")

"""
test_one_shot.py —— 评估集单样本测试脚本

加载训练完成的最优模型权重，在完全没见过的评估集上执行
400 次 20-way 单样本分类测试，输出最终准确率。
对标论文 Table 2 的 92.0%。

论文测试流程（Section 4.3）:
  1. 从评估集中随机选 1 种字母 → 随机选 20 个字符
  2. 从评估集的 4 个书写者中随机选 2 人 A 和 B
  3. A 的 20 张图当候选集，B 的 20 张图当测试图
  4. 每张测试图 vs 20 张候选 → 20 次推理 → 取最高分为预测
  5. 重复 2 次（不同书写者组合），共 40 次/字母 × 10 字母 = 400 次
"""

import os
import sys
import csv
import time

import torch
import numpy as np
from tqdm import tqdm

from siamese_model import SiameseNetwork, count_parameters


def load_model(model_path: str, config) -> SiameseNetwork:
    """
    加载指定路径的模型权重。

    返回:
        SiameseNetwork（已设为 eval 模式，已移至 GPU/CPU）
    """
    device = torch.device(config.device)
    model = SiameseNetwork(config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def run_single_trial(model: SiameseNetwork,
                     test_img: torch.Tensor,
                     support_imgs: torch.Tensor,
                     n_way: int,
                     device: torch.device) -> int:
    """
    执行 1 次 20-way 单样本判断。

    参数:
        model: 孪生网络
        test_img: (1, 1, 105, 105) 测试图
        support_imgs: (n_way, 1, 105, 105) 候选图
        n_way: 候选类别数
        device: 计算设备

    返回:
        得分最高的候选索引 (0 ~ n_way-1)
    """
    # 将测试图复制 n_way 份
    test_batch = test_img.repeat(n_way, 1, 1, 1).to(device)
    support_batch = support_imgs.to(device)

    scores = model(test_batch, support_batch).squeeze()
    return scores.argmax().item()


@torch.no_grad()
def run_full_test(model: SiameseNetwork,
                  test_loader,
                  config) -> tuple:
    """
    执行全部 400 次 20-way 单样本测试。

    每个 test_loader 的 item 包含:
      test_tensor: (1, 1, 105, 105)
      support_tensors: (n_way, 1, 105, 105)
      correct_idx: int

    返回:
        (test_accuracy, detail_list)
        detail_list: [{"trial_id":..., "alphabet":..., "correct":..., "accuracy_per_trial":...}, ...]
    """
    device = next(model.parameters()).device
    correct_total = 0
    total_samples = 0
    detail_list = []

    pbar = tqdm(test_loader, desc="评估集测试", unit="trial")

    for episode_idx, (test_tensor, support_tensors, correct_idx) in enumerate(pbar):
        # 处理 batch 维度
        if test_tensor.dim() == 5:
            test_tensor = test_tensor.squeeze(0)
        if support_tensors.dim() == 5:
            support_tensors = support_tensors.squeeze(0)

        n_way = support_tensors.size(0)

        # 每次 trial 只有 1 次分类决策: 1 张测试图 vs n_way 张候选图
        pred = run_single_trial(
            model, test_tensor, support_tensors, n_way, device,
        )
        if isinstance(correct_idx, torch.Tensor):
            ci = correct_idx.item()
        else:
            ci = correct_idx

        is_correct = 1 if pred == ci else 0
        if is_correct:
            correct_total += 1
        total_samples += 1

        detail_list.append({
            "trial_id": episode_idx + 1,
            "alphabet": "",
            "correct": is_correct,
            "accuracy_per_trial": float(is_correct),
        })

        pbar.set_postfix({
            "acc": f"{correct_total/total_samples:.3f}" if total_samples > 0 else "N/A",
        })

    test_accuracy = correct_total / total_samples if total_samples > 0 else 0.0
    return test_accuracy, detail_list


def save_test_detail(detail_list: list, config, trial_id: int):
    """将测试明细写入 CSV"""
    path = config.get_detail_path()
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "trial_id", "alphabet", "correct", "accuracy_per_trial",
        ])
        for d in detail_list:
            writer.writerow([
                d["trial_id"],
                d.get("alphabet", ""),
                d["correct"],
                f"{d['accuracy_per_trial']:.4f}",
            ])
    print(f"[OK]  测试明细已保存到 {path}")


def test(model_path: str, config, trial_id: int, test_loader, alphabet_meta=None) -> tuple:
    """
    对外接口：加载模型 → 执行测试 → 返回准确率 + 明细。

    参数:
        model_path: 模型权重文件路径
        config: 配置实例
        trial_id: 当前 trial ID
        test_loader: 测试 DataLoader
        alphabet_meta: 字母表元数据列表（可选）

    返回:
        (test_accuracy, detail_list)
    """
    print(f"\n[Test]  评估集 400 次 20-way 单样本测试...")
    start = time.time()

    # 加载模型
    model = load_model(model_path, config)
    info = count_parameters(model)
    print(f"  模型参数量: {info['trainable']:,}")

    # 执行测试
    test_acc, detail_list = run_full_test(model, test_loader, config)

    # 填入字母表信息（如果有）
    if alphabet_meta:
        for i, alpha in enumerate(alphabet_meta):
            if i < len(detail_list):
                detail_list[i]["alphabet"] = alpha

    elapsed = time.time() - start
    print(f"  耗时: {elapsed:.1f} 秒")
    print(f"  评估集单样本准确率: {test_acc:.4f} ({test_acc*100:.1f}%)")
    print(f"  论文基准 (Table 2): 92.0%")
    print(f"  与论文差距: {test_acc - 0.92:+.4f}")

    return test_acc, detail_list


def run_test_episodes_direct(model, trials, config):
    """
    直接使用 trial 列表进行测试（不需要 DataLoader）。
    用于 main.py 中已有 trials 的情况。

    参数:
        model: 已加载的模型
        trials: [(test_img, [candidate_imgs], correct_idx), ...]
        config: 配置实例

    返回:
        (test_accuracy, detail_list)
    """
    device = next(model.parameters()).device
    correct_total = 0
    total_samples = 0
    detail_list = []

    pbar = tqdm(enumerate(trials), total=len(trials), desc="评估集测试", unit="trial")

    for episode_idx, (test_img, support_imgs, correct_idx) in pbar:
        n_way = len(support_imgs)

        test_tensor = torch.from_numpy(
            np.array(test_img, dtype=np.float32) / 255.0
        ).unsqueeze(0).unsqueeze(0)  # (1, 1, 105, 105)

        support_tensors = torch.stack([
            torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).unsqueeze(0)
            for img in support_imgs
        ])  # (n_way, 1, 105, 105)

        # 每次 trial 只有 1 次分类决策: 1 张测试图 vs n_way 张候选图
        pred = run_single_trial(
            model, test_tensor, support_tensors, n_way, device,
        )
        if isinstance(correct_idx, torch.Tensor):
            ci = correct_idx.item()
        else:
            ci = correct_idx

        is_correct = 1 if pred == ci else 0
        if is_correct:
            correct_total += 1
        total_samples += 1

        detail_list.append({
            "trial_id": episode_idx + 1,
            "alphabet": "",
            "correct": is_correct,
            "accuracy_per_trial": float(is_correct),
        })

        pbar.set_postfix({
            "acc": f"{correct_total/total_samples:.3f}" if total_samples > 0 else "N/A",
        })

    test_accuracy = correct_total / total_samples if total_samples > 0 else 0.0
    return test_accuracy, detail_list


if __name__ == "__main__":
    # 独立测试模式
    sys.path.insert(0, ".")
    from config import Config
    from omniglot_dataset import (
        download_omniglot, load_raw_data,
        split_alphabets_drawers, build_char_index,
        generate_one_shot_trials,
    )
    import random

    config = Config()
    config.ensure_dirs()

    print("独立测试模式")
    model_path = config.get_checkpoint_path(0)

    if not os.path.exists(model_path):
        print(f"[FAIL]  模型文件不存在: {model_path}")
        print("  请先运行 train_90k.py")
        sys.exit(1)

    # 加载数据
    download_omniglot(config.data_dir)
    raw_data = load_raw_data(config.data_dir)
    train_chars, val_chars, test_chars = split_alphabets_drawers(raw_data, config)
    test_ci = build_char_index(test_chars)
    test_drawers = set(d for _, _, d, _ in test_chars)

    # 生成测试任务
    rng = random.Random(config.random_seed + 200)
    trials = generate_one_shot_trials(
        test_ci, test_drawers, config.n_way,
        config.test_one_shot_trials, rng,
    )

    # 测试
    model = load_model(model_path, config)
    test_acc, detail_list = run_test_episodes_direct(model, trials, config)

    print(f"\n最终评估集准确率: {test_acc:.4f} ({test_acc*100:.1f}%)")
    print(f"论文基准: 92.0%")

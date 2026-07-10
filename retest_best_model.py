"""
retest_best_model.py - 用正确流程重新测试最优模型

正确的20-way one-shot评估流程:
1. 每个字母表有4个书写者
2. 将4个书写者随机分成2+2的两队
3. 支撑队(2人): 随机选1人提供20个字符的支撑图
4. 查询队(2人): 每人各提供查询图
5. 每字母表: 2查询者 × 20字符 = 40次测试
6. 10字母表 × 40次 = 400次测试
"""

import sys
import os
import random
import time

import torch
import numpy as np

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from omniglot_dataset import (
    download_omniglot, load_raw_data,
    split_alphabets_drawers, build_char_index,
    generate_one_shot_trials,
)
from siamese_model import SiameseNetwork, count_parameters
from test_one_shot import load_model, run_test_episodes_direct


def main():
    config = Config()
    config.ensure_dirs()
    
    print("=" * 60)
    print("重新测试最优模型（正确评估流程）")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n[1/4] 加载数据...")
    download_omniglot(config.data_dir)
    raw_data = load_raw_data(config.data_dir)
    train_chars, val_chars, test_chars = split_alphabets_drawers(raw_data, config)
    
    # 构建测试集字符索引
    test_char_index = build_char_index(test_chars)
    test_drawer_pool = set(d for _, _, d, _ in test_chars)
    
    print(f"  测试集字母表数: {len(test_char_index)}")
    print(f"  测试集书写者: {sorted(test_drawer_pool)}")
    
    # 2. 生成测试任务（正确流程）
    print("\n[2/4] 生成测试任务（正确流程）...")
    test_seed = config.random_seed + 200
    test_rng = random.Random(test_seed)
    
    test_trials, test_meta = generate_one_shot_trials(
        test_char_index, test_drawer_pool,
        config.n_way, config.test_one_shot_trials, test_rng,
        return_meta=True,
    )
    
    print(f"  生成测试任务数: {len(test_trials)}")
    print(f"  测试种子: {test_seed}")
    
    # 统计每个字母表的测试次数
    from collections import Counter
    alpha_counts = Counter(test_meta)
    print(f"\n  每个字母表的测试次数:")
    for alpha, count in sorted(alpha_counts.items()):
        print(f"    {alpha}: {count}次")
    
    # 3. 加载最优模型
    print("\n[3/4] 加载最优模型...")
    best_trial_id = 1  # Trial 1 是历史最优
    model_path = config.get_checkpoint_path(best_trial_id)
    
    if not os.path.exists(model_path):
        print(f"  [FAIL] 模型文件不存在: {model_path}")
        return
    
    model = load_model(model_path, config)
    info = count_parameters(model)
    print(f"  模型路径: {model_path}")
    print(f"  模型参数量: {info['trainable']:,}")
    
    # 4. 执行测试
    print("\n[4/4] 执行测试...")
    start_time = time.time()
    
    test_acc, detail_list = run_test_episodes_direct(model, test_trials, config)
    
    elapsed = time.time() - start_time
    
    # 填入字母表信息
    for i, alpha in enumerate(test_meta):
        if i < len(detail_list):
            detail_list[i]["alphabet"] = alpha
    
    # 5. 输出结果
    print("\n" + "=" * 60)
    print("测试结果")
    print("=" * 60)
    print(f"  评估流程: 正确的20-way one-shot流程")
    print(f"  测试任务数: {len(test_trials)}")
    print(f"  测试准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  论文基准: 92.0%")
    print(f"  与论文差距: {test_acc - 0.92:+.4f}")
    print(f"  测试耗时: {elapsed:.1f}秒")
    
    # 对比旧结果
    old_acc = 0.865000  # Trial 1 旧测试准确率
    print(f"\n  旧测试准确率（错误流程）: {old_acc*100:.2f}%")
    print(f"  新测试准确率（正确流程）: {test_acc*100:.2f}%")
    print(f"  差异: {(test_acc - old_acc)*100:+.2f}%")
    
    # 按字母表统计准确率
    print(f"\n  按字母表统计准确率:")
    alpha_correct = {}
    alpha_total = {}
    for d in detail_list:
        alpha = d["alphabet"]
        if alpha not in alpha_correct:
            alpha_correct[alpha] = 0
            alpha_total[alpha] = 0
        alpha_total[alpha] += 1
        if d["correct"]:
            alpha_correct[alpha] += 1
    
    for alpha in sorted(alpha_correct.keys()):
        acc = alpha_correct[alpha] / alpha_total[alpha] if alpha_total[alpha] > 0 else 0
        print(f"    {alpha}: {acc*100:.2f}% ({alpha_correct[alpha]}/{alpha_total[alpha]})")
    
    print("=" * 60)
    
    return test_acc, detail_list


if __name__ == "__main__":
    main()

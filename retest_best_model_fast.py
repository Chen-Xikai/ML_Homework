"""
retest_best_model_fast.py - 快速重新测试最优模型
"""

import sys
import os
import random
import time
import pickle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from collections import Counter
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

def main():
    config = Config()
    
    print("=" * 60)
    print("快速重新测试最优模型（正确评估流程）")
    print("=" * 60)
    
    # 1. 从缓存加载数据
    print("\n[1/4] 从缓存加载数据...")
    start = time.time()
    
    if not os.path.exists(config.dataset_cache):
        print(f"  [FAIL] 缓存文件不存在: {config.dataset_cache}")
        return
    
    with open(config.dataset_cache, "rb") as f:
        cache_data = pickle.load(f)
    
    # 加载原始数据
    from omniglot_dataset import load_raw_data
    raw_data = load_raw_data(config.data_dir)
    
    # 重建test_chars
    test_chars = []
    for alphabet, char_id, drawer_id in cache_data["test"]:
        key = (alphabet, char_id, drawer_id)
        if key in raw_data:
            test_chars.append((alphabet, char_id, drawer_id, raw_data[key]))
    
    print(f"  加载耗时: {time.time()-start:.1f}秒")
    print(f"  测试集图片: {len(test_chars)}")
    
    # 2. 构建字符索引
    from omniglot_dataset import build_char_index
    test_char_index = build_char_index(test_chars)
    test_drawer_pool = set(d for _, _, d, _ in test_chars)
    
    print(f"  测试集字母表数: {len(test_char_index)}")
    print(f"  测试集书写者: {sorted(test_drawer_pool)}")
    
    # 3. 生成测试任务
    print("\n[2/4] 生成测试任务（正确流程）...")
    from omniglot_dataset import generate_one_shot_trials
    
    test_seed = config.random_seed + 200
    test_rng = random.Random(test_seed)
    
    test_trials, test_meta = generate_one_shot_trials(
        test_char_index, test_drawer_pool,
        config.n_way, config.test_one_shot_trials, test_rng,
        return_meta=True,
    )
    
    print(f"  生成测试任务数: {len(test_trials)}")
    
    # 统计每个字母表的测试次数
    alpha_counts = Counter(test_meta)
    print(f"\n  每个字母表的测试次数:")
    for alpha, count in sorted(alpha_counts.items()):
        print(f"    {alpha}: {count}次")
    
    # 4. 加载模型并测试
    print("\n[3/4] 加载最优模型...")
    best_trial_id = 1
    model_path = config.get_checkpoint_path(best_trial_id)
    
    from siamese_model import SiameseNetwork, count_parameters
    from test_one_shot import load_model, run_test_episodes_direct
    
    model = load_model(model_path, config)
    info = count_parameters(model)
    print(f"  模型参数量: {info['trainable']:,}")
    
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
    old_acc = 0.865000
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


if __name__ == "__main__":
    main()

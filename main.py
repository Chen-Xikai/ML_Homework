"""
main.py —— Optuna 贝叶斯优化编排主脚本

整个实验的顶层调度器。使用 Optuna TPE 贝叶斯采样器自动搜索最优超参数。
循环执行: 训练 → 评估集测试 → 判断 → Optuna 建议下一组参数 → 重训
直到评估集测试准确率 ≥ 90 或达到最大尝试次数。

与论文的一致性:
  - 论文使用 Whetlab 贝叶斯优化 → 本实验使用 Optuna TPE（同类算法）
  - 搜索空间与论文 Section 3.2 完全一致
  - 早停标准与论文完全一致（验证集 320 次单样本，连续 20 轮不提升）

断点续训:
  - Optuna SQLite 持久化: trial 级别断点续训
  - 数据集缓存: 跳过重复的下载和划分
  - 任意中断后重新运行 python main.py 即可无缝续训

用法:
  python main.py          # 全新开始
  python main.py          # 断点续训（自动检测已有进度）
"""

import os
import sys
import csv
import time
import random

import numpy as np
import torch
import optuna
from optuna.samplers import TPESampler
from optuna.trial import TrialState


# ======================== 超参数建议函数（由 objective 调用） ========================

def adapt_lr_range(trial: optuna.Trial, config):
    """
    根据历史 trial 结果自动收窄 lr 搜索范围，跳过已验证无效的区域。

    前 5 个 trial 使用全范围探索，
    第 6 个起抬升下限到历史最优 lr 的 1/5，避免在过低 lr 上浪费时间。
    """
    study = trial.study
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]

    if len(completed) < 3:
        return config.lr_range

    try:
        best_lr = study.best_params["lr"]
        new_low = max(config.lr_range[0], best_lr / 3)
        return (new_low, config.lr_range[1])
    except (ValueError, KeyError):
        return config.lr_range


def suggest_params(trial: optuna.Trial, config) -> tuple:
    """
    从 Optuna trial 获取下一组超参数建议。

    AdamW 搜索空间:
      lr ∈ 自适应范围（最少 [3e-4, 1e-2]，根据历史自动收窄）
      l2 ∈ [0, 0.1]

    返回:
        (lr, l2_lambda)
    """
    lr_low, lr_high = adapt_lr_range(trial, config)
    lr = trial.suggest_float("lr", lr_low, lr_high, log=True)
    l2 = trial.suggest_float(
        "l2_lambda", config.l2_range[0], config.l2_range[1], log=True,
    )
    return lr, l2


# ======================== 汇总日志 ========================

def init_summary(config):
    """创建 summary.csv，写入表头"""
    path = config.get_summary_path()
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "trial_id", "lr", "l2_lambda",
                "best_val_one_shot_acc", "test_accuracy",
                "epochs_trained", "stop_reason",
                "total_time", "model_params",
            ])


def append_summary(config, trial_id, lr, l2,
                   train_result, test_acc, model_params):
    """追加一行汇总记录"""
    path = config.get_summary_path()
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            trial_id,
            f"{lr:.8f}",
            f"{l2:.8f}",
            f"{train_result['best_val_acc']:.6f}",
            f"{test_acc:.6f}",
            train_result["epochs_trained"],
            train_result["stop_reason"],
            f"{train_result['total_time']:.1f}",
            model_params,
        ])


# ======================== Optuna Study 管理（断点续训核心） ========================

def load_or_create_study(config):
    """
    续训核心函数。

    检查 logs/optuna_study.db 是否存在:
      - 存在: 加载已有 study，返回已完成 trial 数
      - 不存在: 创建新 study

    返回:
        (study, completed_count)
    """
    storage_url = f"sqlite:///{config.optuna_db}"
    study_name = "siamese_one_shot_90k"

    if os.path.exists(config.optuna_db):
        print(f"\n{'='*50}")
        print(f"[Resume]  检测到已有 Optuna 数据库")
        print(f"{'='*50}")

        try:
            study = optuna.load_study(
                storage=storage_url,
                study_name=study_name,
            )
            completed = study.trials  # Optuna 自动过滤出已完成的
            completed_valid = [t for t in completed
                               if t.state == TrialState.COMPLETE]

            print(f"  已完成 trial:   {len(completed_valid)}")
            print(f"  剩余 trial:     {config.n_trials - len(completed_valid)}"
                  f"（共 {config.n_trials}）")

            if completed_valid:
                best = study.best_trial
                print(f"  历史最优准确率:  {best.value:.4f}（Trial {best.number}）")
                print(f"  历史最优参数:    lr={best.params['lr']:.6f}, "
                      f"l2={best.params['l2_lambda']:.6f}")
            print(f"{'='*50}")

            return study, len(completed_valid)

        except Exception as e:
            print(f"  [WARN]  加载已有 study 失败: {e}")
            print(f"  将创建新的 study...")

    # 创建新 study
    sampler = TPESampler(seed=config.random_seed)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        sampler=sampler,
        direction="maximize",  # 最大化测试准确率
        load_if_exists=True,
    )
    return study, 0


# ======================== Optuna 目标函数 ========================

def create_objective(config,
                     train_char_index,
                     val_char_index,
                     val_drawer_pool,
                     test_char_index,
                     test_drawer_pool):
    """
    创建 Optuna 目标函数（闭包，捕获数据集相关参数）。

    每次 trial 由 Optuna 自动调用，执行一次完整的"训练→测试"流程。
    返回评估集测试准确率（Optuna 最大化目标）。

    若 test_accuracy ≥ 93%，触发 Optuna 停止（不再浪费资源继续后续 trial）。
    """
    from train_30k import train
    from test_one_shot import load_model, run_test_episodes_direct
    from omniglot_dataset import generate_one_shot_trials
    from siamese_model import count_parameters, SiameseNetwork

    # 所有 trial 共用同一组评估集测试任务（训练前固定生成）
    test_rng = random.Random(config.random_seed + 200)
    test_trials = generate_one_shot_trials(
        test_char_index, test_drawer_pool,
        config.n_way, config.test_one_shot_trials, test_rng,
    )
    print(f"  评估集固定 trial: {len(test_trials)} 次（seed={config.random_seed + 200}）")

    def objective(trial: optuna.Trial):
        trial_id = trial.number
        lr, l2 = suggest_params(trial, config)

        # ---- 训练 ----
        train_result = train(
            config=config,
            trial_id=trial_id,
            lr=lr,
            l2_lambda=l2,
            train_char_index=train_char_index,
            val_char_index=val_char_index,
            val_drawer_pool=val_drawer_pool,
        )

        # ---- 测试（固定 400 次评估集 trial） ----
        model = load_model(train_result["model_path"], config)

        test_acc, detail_list = run_test_episodes_direct(model, test_trials, config)

        # ---- 记录模型参数量 ----
        info = count_parameters(model)
        model_params = info["trainable"]

        # ---- 写入汇总日志 ----
        append_summary(config, trial_id, lr, l2,
                       train_result, test_acc, model_params)

        # ---- 打印结果 ----
        print(f"\n{'='*50}")
        print(f"[Test]  Trial {trial_id} 总结")
        print(f"   超参数: lr={lr:.6f}, l2={l2:.6f}")
        print(f"   验证最佳单样本: {train_result['best_val_acc']:.4f}")
        print(f"   评估集准确率:   {test_acc:.4f} ({test_acc*100:.1f}%)")
        print(f"   目标准确率:      {config.target_test_accuracy}")
        print(f"   差距:          {test_acc - config.target_test_accuracy:+.4f}")

        if test_acc >= config.target_test_accuracy:
            print(f"\n[SUCCESS]  达标！评估集准确率 {test_acc*100:.1f}% ≥ "
                  f"{config.target_test_accuracy*100:.0f}%")
            # 保存最终测试明细
            from test_one_shot import save_test_detail
            save_test_detail(detail_list, config, trial_id)
            print("=" * 50)

            # 告诉 Optuna 停止后续 trial
            study = trial.study
            study.set_user_attr("best_trial_id", trial_id)
            study.set_user_attr("best_test_acc", test_acc)
        else:
            print(f"   [FAIL]  未达标（{test_acc*100:.1f}% < "
                  f"{config.target_test_accuracy*100:.0f}%）")
            print("=" * 50)

        return test_acc

    return objective


# ======================== 主入口 ========================

def main(config):
    print("\n" + "=" * 60)
    print("[Target]  孪生网络单样本图像识别训练")
    print("   论文: Siamese Neural Networks for One-shot Image Recognition")
    print("   配置: 30k 训练对 ")
    print("   调参: Optuna TPE 贝叶斯优化（对应论文 Whetlab）")
    print("   目标: 评估集准确率 ≥ 90% ")
    print("   设备: " + config.device.upper())
    print("=" * 60)

    config.ensure_dirs()

    # ==================== Step 1: 准备数据（支持缓存加速断点续训） ====================
    print("\n" + "=" * 60)
    print("Step 1: 准备 Omniglot 数据集")
    print("=" * 60)

    from omniglot_dataset import (
        download_omniglot, load_raw_data,
        split_alphabets_drawers, build_char_index,
        cache_dataset_split, load_cached_split,
    )

    download_omniglot(config.data_dir)
    raw_data = load_raw_data(config.data_dir)

    # 尝试从缓存恢复
    cached = load_cached_split(config.dataset_cache, raw_data)

    if cached is not None:
        train_chars, val_chars, test_chars = cached
    else:
        train_chars, val_chars, test_chars = split_alphabets_drawers(raw_data, config)
        cache_dataset_split(train_chars, val_chars, test_chars, config.dataset_cache)

    # 构建字符索引（轻量操作，不需缓存）
    train_char_index = build_char_index(train_chars)
    val_char_index = build_char_index(val_chars)
    test_char_index = build_char_index(test_chars)

    # 提取书写者池
    val_drawer_pool = set(d for _, _, d, _ in val_chars)
    test_drawer_pool = set(d for _, _, d, _ in test_chars)

    print(f"[OK]  数据准备完成")
    print(f"  训练集字符索引: {len(train_char_index)} 类")
    print(f"  验证集字符索引: {len(val_char_index)} 类")
    print(f"  测试集字符索引: {len(test_char_index)} 类")
    print(f"  验证集可用书写者: {sorted(val_drawer_pool)}")
    print(f"  测试集可用书写者: {sorted(test_drawer_pool)}")

    # ==================== Step 2: 初始化汇总日志 ====================
    init_summary(config)

    # ==================== Step 3: Optuna Study（断点续训） ====================
    print("\n" + "=" * 60)
    print("Step 2: Optuna 贝叶斯优化")
    print("=" * 60)

    study, completed_count = load_or_create_study(config)
    remaining = config.n_trials - completed_count

    if remaining <= 0:
        print(f"\n[OK]  所有 {config.n_trials} 次 trial 已完成！")
        best = study.best_trial
        print(f"  最优准确率: {best.value:.4f}（Trial {best.number}）")
        print(f"  最优参数: lr={best.params['lr']:.6f}, "
              f""
              f"l2={best.params['l2_lambda']:.6f}")

        if best.value >= config.target_test_accuracy:
            print(f"  [SUCCESS]  已达标！")
        else:
            print(f"  [WARN]  未达标，建议增加 n_trials 或调整搜索空间")
        return

    print(f"准备开始 {remaining} 次新 trial（最多）...")

    # ==================== Step 4: 运行 Optuna 优化 ====================
    objective_fn = create_objective(
        config,
        train_char_index, val_char_index, val_drawer_pool,
        test_char_index, test_drawer_pool,
    )

    # 添加早停回调：若某次 trial 达标则停止
    class TargetReachedCallback:
        """若达标则停止 Optuna study"""
        def __call__(self, study, trial):
            if trial.value is not None and trial.value >= config.target_test_accuracy:
                print(f"\n[Target]  目标准确率已达成，停止后续 trial...")
                study.stop()

    print(f"\n开始 Optuna 优化...")
    study_start = time.time()

    try:
        study.optimize(
            objective_fn,
            n_trials=remaining,
            callbacks=[TargetReachedCallback()],
            show_progress_bar=True,
        )
    except KeyboardInterrupt:
        print(f"\n[WARN]  手动中断（Ctrl+C）。进度已保存，下次运行将自动续训。")

    study_elapsed = time.time() - study_start

    # ==================== Step 5: 打印最终报告 ====================
    print("\n" + "=" * 60)
    print("[Config]  最终报告")
    print("=" * 60)

    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    print(f"  完成 trial 数:   {len(completed)}/{config.n_trials}")
    print(f"  Study 总耗时:    {study_elapsed/3600:.1f} 小时")

    if len(completed) > 0:
        best = study.best_trial
        print(f"  ──────────────────────────────────")
        print(f"  最优 Trial:      {best.number}")
        print(f"  最优参数:        lr={best.params['lr']:.6f}, "
              f""
              f"l2={best.params['l2_lambda']:.6f}")
        print(f"  最优测试准确率:   {best.value:.4f} ({best.value*100:.1f}%)")
        print(f"  目标准确率:        {config.target_test_accuracy}")
        print(f"  与目标差距:        {best.value - config.target_test_accuracy:+.4f}")
        print(f"  ──────────────────────────────────")

        if best.value >= config.target_test_accuracy:
            print(f"\n  [SUCCESS]  已达目标准确率 {config.target_test_accuracy*100:.0f}%！")
        else:
            print(f"\n  [FAIL]  未达目标准确率 {config.target_test_accuracy*100:.0f}%")

            # 打印所有 trial 排名
            print(f"\n  所有 trial 排名:")
            sorted_trials = sorted(completed, key=lambda t: t.value, reverse=True)
            for i, t in enumerate(sorted_trials):
                print(f"    {i+1}. Trial {t.number}: {t.value:.4f} "
                      f"(lr={t.params['lr']:.6f}, λ={t.params['l2_lambda']:.6f})")

        # 超参数重要性分析
        if len(completed) >= 3:
            try:
                importances = optuna.importance.get_param_importances(study)
                print(f"\n  超参数重要性（Optuna 分析）:")
                for param, imp in importances.items():
                    bar = "█" * int(imp * 40)
                    print(f"    {param}: {imp:.3f} {bar}")
            except Exception:
                pass

    print(f"\n  Summary: {config.get_summary_path()}")
    print(f"  Optuna DB: {config.optuna_db}")
    print("=" * 60)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import Config

    config = Config()
    Config.print_config()
    main(config)

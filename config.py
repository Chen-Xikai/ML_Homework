"""
config.py —— 全配置集中管理
所有超参数、文件路径、训练设置、Optuna 搜索空间集中定义于此
其他模块统一引用，修改参数只需改这一个文件
"""

import os
import torch


class Config:
    """全局配置类，按功能分组存放所有配置项"""

    # ======================== 项目根路径 ========================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # ======================== 路径配置 ========================
    data_dir: str = os.path.join(BASE_DIR, "omniglot_data")
    checkpoint_dir: str = os.path.join(BASE_DIR, "checkpoints")
    log_dir: str = os.path.join(BASE_DIR, "logs")
    dataset_cache: str = os.path.join(checkpoint_dir, "dataset_cache.pkl")
    optuna_db: str = os.path.join(log_dir, "optuna_study.db")

    # ======================== 数据集配置 ========================
    # 字母表划分（与论文表1完全一致）
    train_alphabets: int = 30        # 训练用字母表数（取自背景集）
    train_drawers: int = 12          # 训练用书写者数
    val_alphabets: int = 10          # 验证用字母表数（背景集剩余）
    val_drawers: int = 4             # 验证用书写者数
    test_alphabets: int = 10         # 测试用字母表数（评估集全部）
    test_drawers: int = 4            # 测试用书写者数（评估集专属）

    # 样本对数量
    train_pairs: int = 30000         # 训练样本对（正负1:1均衡）
    val_pairs: int = 10000           # 验证集二分类样本对
    val_one_shot_trials: int = 320   # 验证集单样本测试次数（早停依据）
    test_one_shot_trials: int = 400  # 评估集单样本测试次数

    # 单样本分类设置
    n_way: int = 20                  # 单样本分类的类别数
    img_size: int = 105              # 输入图片尺寸（像素）

    # ======================== 仿射畸变配置 ========================
    # 与论文 Section 3.2 完全一致
    num_transforms: int = 0          # 每个样本对生成的增强版本数（0=无畸变）
    theta_range: tuple = (-10.0, 10.0)         # 旋转角度 θ，单位：度
    rhox_range: tuple = (-0.3, 0.3)            # 水平错切 ρx
    rhoy_range: tuple = (-0.3, 0.3)            # 垂直错切 ρy
    sx_range: tuple = (0.8, 1.2)               # 水平缩放
    sy_range: tuple = (0.8, 1.2)               # 垂直缩放
    tx_range: tuple = (-2.0, 2.0)              # 水平平移，单位：像素
    ty_range: tuple = (-2.0, 2.0)              # 垂直平移，单位：像素
    prob_apply: float = 0.5                     # 每个参数独立生效的概率

    # ======================== 模型架构配置 ========================
    # 卷积层配置列表，每项格式: (输出通道数, 卷积核大小)
    # 对应论文 Figure 4 及表1架构
    conv_layers: list = [
        (64, 10),    # Conv1: 1→64, 10×10核
        (128, 7),    # Conv2: 64→128, 7×7核
        (128, 4),    # Conv3: 128→128, 4×4核
        (256, 4),    # Conv4: 128→256, 4×4核
    ]
    pool_size: int = 2              # 池化核大小和步长
    fc_dim: int = 4096              # 全连接层输出维度
    use_batchnorm: bool = True      # 是否在全连接层后接 BatchNorm
    use_sigmoid: bool = False      # 输出层：False=原始分数，FocalLoss 内部自带 sigmoid

    # ======================== 权重初始化配置 ========================
    # 与论文 Section 3.2 完全一致
    conv_weight_mean: float = 0.0
    conv_weight_std: float = 0.01       # 标准差 10⁻²
    conv_bias_mean: float = 0.5
    conv_bias_std: float = 0.01
    fc_weight_mean: float = 0.0
    fc_weight_std: float = 0.2          # 标准差 2×10⁻¹（比卷积层更宽）
    fc_bias_mean: float = 0.5
    fc_bias_std: float = 0.01
    bn_gamma_init: float = 1.0          # BatchNorm γ 初始值
    bn_beta_init: float = 0.0           # BatchNorm β 初始值

    # ======================== 训练超参数 ========================
    batch_size: int = 128
    max_epochs: int = 400
    early_stop_patience: int = 40       # 验证单样本准确率连续40轮未提升则停止
    lr_decay_rate: float = 0.995        # 每 epoch 学习率 ×0.995
    optimizer_type: str = "adamw"
    # lr / l2_lambda 由 Optuna 每次 trial 动态建议，此处不设固定值

    # AdamW 配置
    adam_betas: tuple = (0.9, 0.999)

    # LR 热身
    warmup_epochs: int = 5              # 前 N 个 epoch 从 lr/10 线性增长到 lr

    # Focal Loss
    focal_gamma: float = 2.0            # 聚焦强度，越大越聚焦困难样本
    focal_alpha: float = 1.0            # 类别权重（1.0=正负等权）

    # 困难负样本重采样
    hard_neg_mining: bool = True        # 开启困难负样本在线重采样
    hard_neg_interval: int = 5          # 每隔 N 个 epoch 重采样一次
    hard_neg_ratio: float = 0.3         # 替换比例（30% 的最困难负样本被替换）

    # ======================== Optuna 搜索空间配置 ========================
    lr_range: tuple = (5e-3, 1.5e-2)       # 高初始 lr 区间
    l2_range: tuple = (1e-6, 1e-1)         # 论文: λ ∈ [0, 0.1]（1e-6 近似为 0）
    n_trials: int = 10                      # Optuna 最多尝试次数
    target_test_accuracy: float = 0.90      # 评估集 one-shot 准确率达标线
    optuna_sampler: str = "TPE"             # TPE 贝叶斯采样器（与 Whetlab 同类算法）

    # ======================== 系统配置 ========================
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0                    # Windows 多进程 pickle PIL 图极慢，用单进程
    random_seed: int = 42                   # 全局随机种子
    pin_memory: bool = True                 # 加速 CPU → GPU 数据传输

    # ======================== 运行时自动计算的路径 ========================
    @classmethod
    def get_checkpoint_path(cls, trial_id: int) -> str:
        """获取某次 trial 的模型保存路径"""
        return os.path.join(cls.checkpoint_dir, f"attempt_{trial_id}_best.pth")

    @classmethod
    def get_log_path(cls, trial_id: int) -> str:
        """获取某次 trial 的训练日志路径"""
        return os.path.join(cls.log_dir, f"attempt_{trial_id}_log.csv")

    @classmethod
    def get_summary_path(cls) -> str:
        return os.path.join(cls.log_dir, "summary.csv")

    @classmethod
    def get_detail_path(cls) -> str:
        return os.path.join(cls.log_dir, "final_test_detail.csv")

    @classmethod
    def ensure_dirs(cls):
        """确保所有需要的目录存在"""
        os.makedirs(cls.checkpoint_dir, exist_ok=True)
        os.makedirs(cls.log_dir, exist_ok=True)

    @classmethod
    def print_config(cls):
        """打印关键配置信息"""
        print("=" * 60)
        print("[Config]  训练配置摘要")
        print("=" * 60)
        print(f"  设备:           {cls.device.upper()}")
        print(f"  模型输入尺寸:    {cls.img_size}×{cls.img_size}")
        print(f"  训练样本对:     {cls.train_pairs:,}（原始）")
        print(f"  有效样本对:     {cls.train_pairs * (cls.num_transforms + 1):,}（含增强）")
        print(f"  验证样本对:     {cls.val_pairs:,}")
        print(f"  验证单样本次数:  {cls.val_one_shot_trials}")
        print(f"  测试单样本次数:  {cls.test_one_shot_trials}")
        print(f"  N-way:          {cls.n_way}")
        print(f"  字母表划分:     训练{cls.train_alphabets} / 验证{cls.val_alphabets} / 测试{cls.test_alphabets}")
        print(f"  书写者划分:     训练{cls.train_drawers} / 验证{cls.val_drawers} / 测试{cls.test_drawers}")
        print(f"  Batch Size:     {cls.batch_size}")
        print(f"  最大 Epoch:     {cls.max_epochs}")
        print(f"  早停 Patience:   {cls.early_stop_patience}")
        print(f"  目标准确率:     {cls.target_test_accuracy}")
        print(f"  Optuna Trials:  {cls.n_trials}")
        print(f"  Optuna 算法:     {cls.optuna_sampler}")
        print(f"  随机种子:       {cls.random_seed}")
        print("=" * 60)


if __name__ == "__main__":
    Config.print_config()

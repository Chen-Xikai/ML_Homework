"""
config.py - 分类网络对比实验配置
"""

import os
import torch


class Config:
    """全局配置类"""
    
    # ======================== 路径配置 ========================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "..", "omniglot_data")
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
    LOG_DIR = os.path.join(BASE_DIR, "logs")
    RESULT_DIR = os.path.join(BASE_DIR, "results")
    
    # ======================== 数据集配置 ========================
    # Omniglot总类别数
    NUM_CLASSES = 1623
    
    # 图像配置
    IMG_SIZE = 105
    IMG_CHANNELS = 1
    
    # 方法1：原始one-shot划分
    METHOD1_TRAIN_ALPHABETS = 30
    METHOD1_VAL_ALPHABETS = 10
    METHOD1_TEST_ALPHABETS = 10
    METHOD1_TRAIN_DRAWERS = 12
    METHOD1_VAL_DRAWERS = 4
    METHOD1_TEST_DRAWERS = 4
    
    # 方法2：按字符12:4:4划分
    METHOD2_TRAIN_RATIO = 12
    METHOD2_VAL_RATIO = 4
    METHOD2_TEST_RATIO = 4
    
    # ======================== 网络架构 ========================
    # 与孪生网络特征提取器相同的CNN架构
    CONV_LAYERS = [(64, 10), (128, 7), (128, 4), (256, 4)]
    POOL_SIZE = 2
    FC_DIM = 4096
    USE_BATCHNORM = True
    
    # MLP分类头
    HIDDEN_DIM = 1024
    DROPOUT = 0.5
    
    # ======================== 训练配置 ========================
    BATCH_SIZE = 128
    MAX_EPOCHS = 100
    LR = 0.001
    LR_DECAY = 0.995
    WEIGHT_DECAY = 1e-4
    WARMUP_EPOCHS = 5
    
    # 早停
    PATIENCE = 20
    
    # 随机种子
    RANDOM_SEED = 42
    
    # ======================== 测试配置 ========================
    KNN_K = 1
    
    # ======================== 设备 ========================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    @classmethod
    def ensure_dirs(cls):
        """确保所有目录存在"""
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        os.makedirs(cls.RESULT_DIR, exist_ok=True)
    
    @classmethod
    def print_config(cls):
        """打印配置"""
        print("=" * 60)
        print("分类网络对比实验配置")
        print("=" * 60)
        print(f"  设备: {cls.DEVICE}")
        print(f"  数据目录: {cls.DATA_DIR}")
        print(f"  总类别数: {cls.NUM_CLASSES}")
        print(f"  图像尺寸: {cls.IMG_SIZE}x{cls.IMG_SIZE}")
        print(f"  CNN架构: {cls.CONV_LAYERS}")
        print(f"  FC维度: {cls.FC_DIM}")
        print(f"  MLP隐藏层: {cls.HIDDEN_DIM}")
        print(f"  批大小: {cls.BATCH_SIZE}")
        print(f"  最大轮数: {cls.MAX_EPOCHS}")
        print(f"  学习率: {cls.LR}")
        print(f"  KNN K值: {cls.KNN_K}")
        print("=" * 60)


if __name__ == "__main__":
    Config.print_config()

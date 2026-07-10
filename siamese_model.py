"""
siamese_model.py —— 孪生网络模型定义

卷积孪生网络架构，完全按论文 Figure 4 和 Section 3.1 搭建:
  - 4 层卷积（含 3 次 2×2 最大池化）
  - 展平后接 4096 维全连接层 + BatchNorm + ReLU（特征提取分支）
  - 双分支共享权重 → 逐元素 L1 距离 → FC(4096→1) → sigmoid

论文原文（Section 3.1）:
  p = σ(Σ_j α_j |h1_j - h2_j|)   —— 加权 L1 距离 + sigmoid
  这里 α_j 由最后一个 FC 层的权重隐式学习

权重初始化（Section 3.2）:
  - 卷积层权重: N(0, 10⁻²), 偏置: N(0.5, 10⁻²)
  - 全连接层权重: N(0, 2×10⁻¹), 偏置: N(0.5, 10⁻²)
"""

import torch
import torch.nn as nn


class ConvFeatureExtractor(nn.Module):
    """
    单个 CNN 特征提取分支（孪生网络的"一半"）。

    架构（对应论文 Figure 4 红色框之前的部分）:
        Input: (B, 1, 105, 105)
        Conv1: 1→64, 10×10核, stride=1 → 96×96 → ReLU → MaxPool(2×2) → 48×48
        Conv2: 64→128, 7×7核, stride=1 → 42×42 → ReLU → MaxPool(2×2) → 21×21
        Conv3: 128→128, 4×4核, stride=1 → 18×18 → ReLU → MaxPool(2×2) → 9×9
        Conv4: 128→256, 4×4核, stride=1 → 6×6 → ReLU（无池化）
        Flatten: 256×6×6 = 9216
        FC: 9216 → 4096 → BatchNorm1d → ReLU
        Output: (B, 4096)
    """

    def __init__(self, config):
        super(ConvFeatureExtractor, self).__init__()

        # 构建卷积层序列
        conv_blocks = []
        in_channels = 1  # 输入为单通道灰度图

        for idx, (out_channels, kernel_size) in enumerate(config.conv_layers):
            # 卷积 + ReLU
            conv_blocks.append(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=1)
            )
            conv_blocks.append(nn.ReLU(inplace=True))

            # 前 3 层卷积后接 MaxPool
            # (Conv4 即 idx=3 不接池化)
            if idx < 3:
                conv_blocks.append(
                    nn.MaxPool2d(kernel_size=config.pool_size, stride=config.pool_size)
                )

            in_channels = out_channels

        self.conv_layers = nn.Sequential(*conv_blocks)

        # 全连接层: 展平后 256×6×6 = 9216 → 4096
        self.flatten_dim = config.conv_layers[-1][0] * 6 * 6  # 256×6×6 = 9216
        self.fc = nn.Linear(self.flatten_dim, config.fc_dim)

        # BatchNorm + ReLU（在 FC 之后）
        if config.use_batchnorm:
            self.bn = nn.BatchNorm1d(config.fc_dim)
        else:
            self.bn = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: (B, 1, 105, 105) 输入图像张量
        返回:
            (B, 4096) 特征向量
        """
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)       # (B, 9216)
        x = self.fc(x)                   # (B, 4096)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SiameseNetwork(nn.Module):
    """
    完整孪生网络。

    两个分支共享同一个 ConvFeatureExtractor（权重绑定），
    分别提取两张图片的特征 → 计算逐元素 L1 距离 → FC 输出 → sigmoid。

    前向过程:
        ① feat1 = self.extractor(img1)   → (B, 4096)
        ② feat2 = self.extractor(img2)   → (B, 4096)   （同一个 extractor！）
        ③ diff = |feat1 - feat2|         → (B, 4096)
        ④ out = self.output_fc(diff)     → (B, 1)
        ⑤ prob = sigmoid(out)            → (B, 1) ∈ [0, 1]
    """

    def __init__(self, config):
        super(SiameseNetwork, self).__init__()

        # 共享的特征提取分支（只创建一次，两个输入共用）
        self.extractor = ConvFeatureExtractor(config)

        # 输出层: 4096 维差值 → 1 维相似度分数
        # 这一层的权重 α_j 相当于论文中的加权 L1 距离权重
        self.output_fc = nn.Linear(config.fc_dim, 1)

        if config.use_sigmoid:
            self.sigmoid = nn.Sigmoid()
        else:
            self.sigmoid = nn.Identity()

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        参数:
            img1: (B, 1, 105, 105) 第一张图
            img2: (B, 1, 105, 105) 第二张图
        返回:
            (B, 1) 相似度概率 ∈ [0, 1]，1=同类，0=不同类
        """
        feat1 = self.extractor(img1)
        feat2 = self.extractor(img2)

        # 逐元素 L1 距离
        diff = torch.abs(feat1 - feat2)

        # 全连接输出 + sigmoid
        out = self.output_fc(diff)
        prob = self.sigmoid(out)

        return prob


# ======================== 权重初始化函数 ========================

def init_weights_conv(module: nn.Module, config):
    """对卷积层应用论文指定的初始化"""
    if isinstance(module, nn.Conv2d):
        nn.init.normal_(module.weight,
                        mean=config.conv_weight_mean,
                        std=config.conv_weight_std)
        nn.init.normal_(module.bias,
                        mean=config.conv_bias_mean,
                        std=config.conv_bias_std)


def init_weights_fc(module: nn.Module, config):
    """对全连接层应用论文指定的初始化"""
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight,
                        mean=config.fc_weight_mean,
                        std=config.fc_weight_std)
        nn.init.normal_(module.bias,
                        mean=config.fc_bias_mean,
                        std=config.fc_bias_std)


def init_weights_bn(module: nn.Module, config):
    """对 BatchNorm 层初始化"""
    if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, config.bn_gamma_init)
        nn.init.constant_(module.bias, config.bn_beta_init)


def initialize_model(model: nn.Module, config):
    """
    递归遍历模型所有子模块，按类型调用对应的初始化函数。

    卷积层 → init_weights_conv
    全连接层 → init_weights_fc
    BatchNorm 层 → init_weights_bn
    """
    for module in model.modules():
        init_weights_conv(module, config)
        init_weights_fc(module, config)
        init_weights_bn(module, config)


def count_parameters(model: nn.Module) -> dict:
    """
    统计模型参数量。

    返回:
        {"total": 总参数, "trainable": 可训练参数, "details": 各层明细}
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    details = {}
    for name, param in model.named_parameters():
        details[name] = param.numel()

    return {"total": total, "trainable": trainable, "details": details}


if __name__ == "__main__":
    # 自测：构建模型 + 前向传播 + 参数统计
    import sys
    sys.path.insert(0, ".")
    from config import Config

    config = Config()

    print("=" * 50)
    print("模型自测")
    print("=" * 50)

    # 1. 构建特征提取器
    extractor = ConvFeatureExtractor(config)
    x = torch.randn(4, 1, 105, 105)
    feat = extractor(x)
    print(f"[OK]  ConvFeatureExtractor: 输入 {x.shape} → 输出 {feat.shape}")
    assert feat.shape == (4, 4096), f"期望 (4, 4096)，实际 {feat.shape}"

    # 2. 构建完整孪生网络
    model = SiameseNetwork(config)
    initialize_model(model, config)
    img1 = torch.randn(4, 1, 105, 105)
    img2 = torch.randn(4, 1, 105, 105)
    output = model(img1, img2)
    print(f"[OK]  SiameseNetwork: 输入 {img1.shape} + {img2.shape} → 输出 {output.shape}")
    assert output.shape == (4, 1), f"期望 (4, 1)，实际 {output.shape}"
    assert torch.all(output >= 0) and torch.all(output <= 1), \
        "sigmoid 输出应在 [0, 1] 范围内"

    # 3. 统计参数量
    info = count_parameters(model)
    print(f"[OK]  总参数量:    {info['total']:,}")
    print(f"[OK]  可训练参数:  {info['trainable']:,}")
    print(f"  - 特征提取器: {sum(count_parameters(extractor)['details'].values()):,}")

    # 4. 验证权重共享（两个相同输入必须产生相同特征）
    with torch.no_grad():
        f1 = model.extractor(img1)
        f2 = model.extractor(img1)  # 相同输入
        diff = (f1 - f2).abs().max().item()
        print(f"[OK]  权重共享验证: feat(img1) 两次调用最大差异 = {diff:.10f}")

    print("\n[SUCCESS]  所有模型自测通过！")

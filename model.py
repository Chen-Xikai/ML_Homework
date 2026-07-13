"""
model.py - 分类网络模型

包含：
1. ConvFeatureExtractor：4层CNN特征提取器（与孪生网络相同）
2. ClassificationNet：分类网络（CNN + MLP分类头）
"""

import torch
import torch.nn as nn


class ConvFeatureExtractor(nn.Module):
    """
    4层CNN特征提取器（与孪生网络相同）
    
    架构：
        Input: (B, 1, 105, 105)
        Conv1: 1→64, 10×10核 → ReLU → MaxPool(2×2) → 48×48
        Conv2: 64→128, 7×7核 → ReLU → MaxPool(2×2) → 21×21
        Conv3: 128→128, 4×4核 → ReLU → MaxPool(2×2) → 9×9
        Conv4: 128→256, 4×4核 → ReLU（无池化） → 6×6
        Flatten: 256×6×6 = 9216
        FC: 9216 → 4096 → BatchNorm → ReLU
        Output: (B, 4096)
    """
    
    def __init__(self, config):
        super(ConvFeatureExtractor, self).__init__()
        
        # 构建卷积层
        conv_blocks = []
        in_channels = 1
        
        for idx, (out_channels, kernel_size) in enumerate(config.CONV_LAYERS):
            conv_blocks.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride=1))
            conv_blocks.append(nn.ReLU(inplace=True))
            
            if idx < 3:  # 前3层接MaxPool
                conv_blocks.append(nn.MaxPool2d(kernel_size=config.POOL_SIZE, stride=config.POOL_SIZE))
            
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*conv_blocks)
        
        # 全连接层
        self.flatten_dim = config.CONV_LAYERS[-1][0] * 6 * 6  # 256×6×6 = 9216
        self.fc = nn.Linear(self.flatten_dim, config.FC_DIM)
        
        # BatchNorm + ReLU
        if config.USE_BATCHNORM:
            self.bn = nn.BatchNorm1d(config.FC_DIM)
        else:
            self.bn = nn.Identity()
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """
        参数:
            x: (B, 1, 105, 105)
        返回:
            (B, 4096) 特征向量
        """
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ClassificationNet(nn.Module):
    """
    分类网络：CNN特征提取器 + MLP分类头
    
    架构：
        ConvFeatureExtractor → 4096维特征
        MLP: 4096 → 1024 → ReLU → Dropout → num_classes
    """
    
    def __init__(self, config):
        super(ClassificationNet, self).__init__()
        
        # 特征提取器
        self.feature_extractor = ConvFeatureExtractor(config)
        
        # MLP分类头
        self.classifier = nn.Sequential(
            nn.Linear(config.FC_DIM, config.HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM, config.NUM_CLASSES)
        )
    
    def forward(self, x):
        """
        参数:
            x: (B, 1, 105, 105)
        返回:
            (B, num_classes) 类别 logits
        """
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits
    
    def extract_features(self, x):
        """
        提取中间特征（用于最近邻）
        
        参数:
            x: (B, 1, 105, 105)
        返回:
            (B, 4096) 特征向量
        """
        return self.feature_extractor(x)


def init_weights(model, config):
    """初始化权重"""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                nn.init.normal_(m.bias, mean=0.5, std=0.01)
        elif isinstance(m, nn.Linear):
            if m.out_features == config.FC_DIM:
                nn.init.normal_(m.weight, mean=0.0, std=0.2)
            else:
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                nn.init.normal_(m.bias, mean=0.5, std=0.01)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


if __name__ == "__main__":
    from config import Config
    
    config = Config()
    model = ClassificationNet(config)
    
    # 测试前向传播
    x = torch.randn(4, 1, 105, 105)
    logits = model(x)
    features = model.extract_features(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出logits形状: {logits.shape}")
    print(f"特征向量形状: {features.shape}")
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

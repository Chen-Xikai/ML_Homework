"""
dataset.py - 数据集加载模块

支持两种数据划分方式：
1. 方法1：原始one-shot划分（30字母表训练，10验证，10测试）
2. 方法2：按字符12:4:4划分
"""

import os
import pickle
import random
import numpy as np
from PIL import Image
from collections import defaultdict
from typing import Tuple, Dict, List

import torch
from torch.utils.data import Dataset, DataLoader


def load_raw_data(data_dir: str) -> Dict[tuple, Image.Image]:
    """
    加载Omniglot原始数据
    
    返回:
        {(alphabet_name, char_id, drawer_id): PIL_Image}
    """
    raw_data = {}
    
    for split_name in ["images_background", "images_evaluation"]:
        split_dir = os.path.join(data_dir, split_name)
        if not os.path.exists(split_dir):
            continue
        
        for alphabet_name in os.listdir(split_dir):
            alpha_dir = os.path.join(split_dir, alphabet_name)
            if not os.path.isdir(alpha_dir):
                continue
            
            for char_dir_name in os.listdir(alpha_dir):
                char_path = os.path.join(alpha_dir, char_dir_name)
                if not os.path.isdir(char_path):
                    continue
                
                try:
                    char_id = int(char_dir_name.replace("character", ""))
                except ValueError:
                    char_id = char_dir_name
                
                for img_filename in os.listdir(char_path):
                    if not img_filename.endswith(".png"):
                        continue
                    
                    try:
                        drawer_id = int(img_filename.split("_")[-1].replace(".png", ""))
                    except ValueError:
                        drawer_id = img_filename
                    
                    img_path = os.path.join(char_path, img_filename)
                    img = Image.open(img_path).convert("L")
                    if img.size != (105, 105):
                        img = img.resize((105, 105), Image.BILINEAR)
                    
                    raw_data[(alphabet_name, char_id, drawer_id)] = img
    
    return raw_data


def split_alphabets_drawers(raw_data: dict, seed: int = 42):
    """
    论文原始划分：50字母表 → 40背景+10评估
    背景集 → 30训练 + 10验证
    评估集 → 10测试
    书写者 → 12训练 + 4验证 + 4测试
    
    返回:
        train_chars, val_chars, test_chars
    """
    all_alphabets = sorted(set(k[0] for k in raw_data.keys()))
    all_drawers = sorted(set(k[2] for k in raw_data.keys()))
    
    rng = random.Random(seed)
    
    # 划分字母表
    shuffled_alphabets = list(all_alphabets)
    rng.shuffle(shuffled_alphabets)
    
    background_alphabets = set(shuffled_alphabets[:40])
    evaluation_alphabets = set(shuffled_alphabets[40:50])
    
    bg_list = sorted(background_alphabets)
    rng.shuffle(bg_list)
    train_alphabets = set(bg_list[:30])
    val_alphabets = set(bg_list[30:40])
    
    # 划分书写者
    shuffled_drawers = list(all_drawers)
    rng.shuffle(shuffled_drawers)
    train_drawers = set(shuffled_drawers[:12])
    val_drawers = set(shuffled_drawers[12:16])
    test_drawers = set(shuffled_drawers[16:20])
    
    # 归类
    train_chars = []
    val_chars = []
    test_chars = []
    
    for (alphabet, char_id, drawer_id), img in raw_data.items():
        entry = (alphabet, char_id, drawer_id, img)
        
        if alphabet in train_alphabets and drawer_id in train_drawers:
            train_chars.append(entry)
        elif alphabet in val_alphabets and drawer_id in val_drawers:
            val_chars.append(entry)
        elif alphabet in evaluation_alphabets and drawer_id in test_drawers:
            test_chars.append(entry)
    
    return train_chars, val_chars, test_chars


def split_per_character(raw_data: dict, seed: int = 42):
    """
    按字符划分：每个字符的20个书写者按12:4:4划分
    
    返回:
        train_chars, val_chars, test_chars
    """
    # 按字符分组
    char_groups = defaultdict(list)
    for (alphabet, char_id, drawer_id), img in raw_data.items():
        char_groups[(alphabet, char_id)].append((drawer_id, img))
    
    rng = random.Random(seed)
    
    train_chars = []
    val_chars = []
    test_chars = []
    
    for char_key, drawers_imgs in char_groups.items():
        alphabet, char_id = char_key
        
        # 按书写者ID排序
        drawers_imgs.sort(key=lambda x: x[0])
        
        # 随机打乱
        rng.shuffle(drawers_imgs)
        
        # 按12:4:4划分
        train = drawers_imgs[:12]
        val = drawers_imgs[12:16]
        test = drawers_imgs[16:20]
        
        for drawer_id, img in train:
            train_chars.append((alphabet, char_id, drawer_id, img))
        for drawer_id, img in val:
            val_chars.append((alphabet, char_id, drawer_id, img))
        for drawer_id, img in test:
            test_chars.append((alphabet, char_id, drawer_id, img))
    
    return train_chars, val_chars, test_chars


class ClassificationDataset(Dataset):
    """分类数据集"""
    
    def __init__(self, char_list: list, char_to_idx: dict, transform=None):
        self.samples = []
        self.transform = transform
        
        for alphabet, char_id, drawer_id, img in char_list:
            char_key = (alphabet, char_id)
            if char_key in char_to_idx:
                self.samples.append((img, char_to_idx[char_key]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img, label = self.samples[idx]
        img_np = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).unsqueeze(0)
        
        if self.transform:
            img_tensor = self.transform(img_tensor)
        
        return img_tensor, label


def load_data_method1(data_dir: str, seed: int = 42):
    """
    方法1：原始one-shot划分
    
    返回:
        (train_dataset, val_dataset, test_dataset), char_to_idx
    """
    print("[方法1] 加载数据（原始one-shot划分）...")
    
    raw_data = load_raw_data(data_dir)
    train_chars, val_chars, test_chars = split_alphabets_drawers(raw_data, seed)
    
    # 构建字符到索引的映射
    all_chars = set()
    for chars in [train_chars, val_chars, test_chars]:
        for alphabet, char_id, _, _ in chars:
            all_chars.add((alphabet, char_id))
    
    char_to_idx = {char: idx for idx, char in enumerate(sorted(all_chars))}
    
    print(f"  总类别数: {len(char_to_idx)}")
    print(f"  训练集: {len(train_chars)} 张")
    print(f"  验证集: {len(val_chars)} 张")
    print(f"  测试集: {len(test_chars)} 张")
    
    train_dataset = ClassificationDataset(train_chars, char_to_idx)
    val_dataset = ClassificationDataset(val_chars, char_to_idx)
    test_dataset = ClassificationDataset(test_chars, char_to_idx)
    
    return (train_dataset, val_dataset, test_dataset), char_to_idx


def load_data_method2(data_dir: str, seed: int = 42):
    """
    方法2：按字符12:4:4划分
    
    返回:
        (train_dataset, val_dataset, test_dataset), char_to_idx
    """
    cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "dataset_cache.pkl")
    
    if os.path.exists(cache_path):
        print("[方法2] 从缓存加载数据...")
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        train_chars, val_chars, test_chars = cache["train_chars"], cache["val_chars"], cache["test_chars"]
        char_to_idx = cache["char_to_idx"]
    else:
        print("[方法2] 加载数据（按字符12:4:4划分）...")
        raw_data = load_raw_data(data_dir)
        train_chars, val_chars, test_chars = split_per_character(raw_data, seed)
        
        all_chars = set()
        for chars in [train_chars, val_chars, test_chars]:
            for alphabet, char_id, _, _ in chars:
                all_chars.add((alphabet, char_id))
        
        char_to_idx = {char: idx for idx, char in enumerate(sorted(all_chars))}
        
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump({"train_chars": train_chars, "val_chars": val_chars, "test_chars": test_chars, "char_to_idx": char_to_idx}, f)
        print(f"  缓存已保存: {cache_path}")
    
    print(f"  总类别数: {len(char_to_idx)}")
    print(f"  训练集: {len(train_chars)} 张")
    print(f"  验证集: {len(val_chars)} 张")
    print(f"  测试集: {len(test_chars)} 张")
    
    train_dataset = ClassificationDataset(train_chars, char_to_idx)
    val_dataset = ClassificationDataset(val_chars, char_to_idx)
    test_dataset = ClassificationDataset(test_chars, char_to_idx)
    
    return (train_dataset, val_dataset, test_dataset), char_to_idx


def get_data_loader(dataset: Dataset, batch_size: int, shuffle: bool = True):
    """创建DataLoader"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
    )


if __name__ == "__main__":
    from config import Config
    
    config = Config()
    
    # 测试方法1
    print("\n" + "=" * 60)
    (train_ds, val_ds, test_ds), char_to_idx = load_data_method1(config.DATA_DIR)
    print(f"方法1类别数: {len(char_to_idx)}")
    
    # 测试方法2
    print("\n" + "=" * 60)
    (train_ds2, val_ds2, test_ds2), char_to_idx2 = load_data_method2(config.DATA_DIR)
    print(f"方法2类别数: {len(char_to_idx2)}")

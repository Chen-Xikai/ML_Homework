"""
.py
cache_dataset预生成所有数据集的本地缓存，加速消融实验
独立运行，不影响当前训练
"""

import os
import pickle
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from collections import defaultdict

# 添加当前目录到路径
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import OmniglotDataset, get_test_alphabets, get_random_writer_split


class DatasetCacher:
    """数据集缓存生成器"""
    
    def __init__(self, root_dir, cache_dir='./dataset_cache', seed=42):
        self.root_dir = root_dir
        self.cache_dir = cache_dir
        self.seed = seed
        self.transform = transforms.Compose([
            transforms.Resize((105, 105)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        os.makedirs(cache_dir, exist_ok=True)
    
    def cache_dataset(self, split, writers, alphabets=None, pairs_per_class=17, name='train'):
        """缓存一个数据集"""
        cache_path = os.path.join(self.cache_dir, f'{name}_cache.pkl')
        
        # 检查缓存是否已存在
        if os.path.exists(cache_path):
            print(f"[{name}] Cache already exists at {cache_path}")
            # 加载并验证
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            print(f"  Cached pairs: {len(cached_data['pairs'])}")
            return cached_data
        
        print(f"\n[{name}] Generating cache...")
        start_time = time.time()
        
        # 加载数据集
        dataset = OmniglotDataset(self.root_dir, split=split)
        
        # 筛选字母表
        if alphabets is not None:
            available_chars = {}
            for char_id, images in dataset.char_to_images.items():
                alphabet = char_id.split('/')[0]
                if alphabet in alphabets:
                    available_chars[char_id] = images
        else:
            available_chars = dataset.char_to_images
        
        # 按书写者筛选图像
        writer_set = set(writers)
        filtered_chars = {}
        for char_id, images in available_chars.items():
            filtered_images = []
            for img_path in images:
                filename = os.path.basename(img_path)
                writer_id = int(filename.split('_')[-1].split('.')[0])
                if writer_id in writer_set:
                    filtered_images.append(img_path)
            if len(filtered_images) >= 2:
                filtered_chars[char_id] = filtered_images
        
        print(f"  Characters: {len(filtered_chars)}")
        
        # 预加载所有图像到内存
        image_cache = {}
        all_paths = set()
        
        # 先收集所有需要的路径
        char_list = list(filtered_chars.keys())
        for char_id in char_list:
            images = filtered_chars[char_id]
            for img_path in images:
                all_paths.add(img_path)
        
        print(f"  Loading {len(all_paths)} images...")
        
        for i, img_path in enumerate(all_paths):
            if (i + 1) % 1000 == 0:
                print(f"    Loaded {i+1}/{len(all_paths)} images...")
            img = Image.open(img_path).convert('L')
            image_cache[img_path] = self.transform(img)
        
        print(f"  All images loaded to memory")
        
        # 生成图像对
        pairs = []
        labels = []
        
        # 正样本对
        for char_id, images in filtered_chars.items():
            for _ in range(pairs_per_class):
                img1, img2 = np.random.choice(len(images), 2, replace=False)
                pairs.append((images[img1], images[img2]))
                labels.append(0)
        
        # 负样本对
        num_positive = len(labels)
        char_list = list(filtered_chars.keys())
        num_negative = 0
        while num_negative < num_positive:
            char1, char2 = np.random.choice(len(char_list), 2, replace=False)
            img1 = np.random.choice(filtered_chars[char_list[char1]])
            img2 = np.random.choice(filtered_chars[char_list[char2]])
            pairs.append((img1, img2))
            labels.append(1)
            num_negative += 1
        
        # 保存缓存
        cached_data = {
            'pairs': pairs,
            'labels': labels,
            'image_cache': image_cache,
            'filtered_chars': filtered_chars,
            'writers': writers,
            'alphabets': list(set([c.split('/')[0] for c in filtered_chars.keys()]))
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cached_data, f)
        
        elapsed = time.time() - start_time
        print(f"  Cache saved: {cache_path}")
        print(f"  Pairs: {len(pairs)}, Time: {elapsed:.1f}s")
        
        return cached_data
    
    def cache_all_datasets(self, pairs_per_class=17):
        """缓存所有数据集"""
        print("=" * 60)
        print("Generating Dataset Caches")
        print("=" * 60)
        
        # 获取书写者划分
        train_writers, val_writers, test_writers = get_random_writer_split(self.seed)
        
        # 获取字母表划分
        background_dataset = OmniglotDataset(self.root_dir, split='background')
        all_background_alphabets = list(background_dataset.alphabet_to_chars.keys())
        
        excluded_from_val = ['Braille', 'Balinese']
        available_alphabets = [a for a in all_background_alphabets if a not in excluded_from_val]
        
        np.random.seed(self.seed)
        np.random.shuffle(available_alphabets)
        
        val_alphabets = available_alphabets[:10]
        train_alphabets = available_alphabets[10:38]  # 包含Braille/Balinese
        
        test_alphabets = get_test_alphabets(self.root_dir)
        
        print(f"\nConfiguration:")
        print(f"  Train writers: {train_writers}")
        print(f"  Val writers: {val_writers}")
        print(f"  Test writers: {test_writers}")
        print(f"  Train alphabets: {len(train_alphabets)}")
        print(f"  Val alphabets: {len(val_alphabets)}")
        print(f"  Test alphabets: {len(test_alphabets)}")
        print(f"  Pairs per class: {pairs_per_class}")
        
        # 缓存训练集
        self.cache_dataset(
            split='background',
            writers=train_writers,
            alphabets=train_alphabets,
            pairs_per_class=pairs_per_class,
            name='train'
        )
        
        # 缓存验证集
        self.cache_dataset(
            split='background',
            writers=val_writers,
            alphabets=val_alphabets,
            pairs_per_class=pairs_per_class // 2,
            name='val'
        )
        
        # 缓存测试集
        self.cache_dataset(
            split='evaluation',
            writers=test_writers,
            alphabets=None,  # 使用所有测试集字母表
            pairs_per_class=pairs_per_class // 2,
            name='test'
        )
        
        print("\n" + "=" * 60)
        print("All caches generated successfully!")
        print("=" * 60)


class CachedDataset(Dataset):
    """使用缓存的数据集类"""
    
    def __init__(self, cache_path):
        with open(cache_path, 'rb') as f:
            self.data = pickle.load(f)
        
        self.pairs = self.data['pairs']
        self.labels = self.data['labels']
        self.image_cache = self.data['image_cache']
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img_path1, img_path2 = self.pairs[idx]
        label = self.labels[idx]
        
        img1 = self.image_cache[img_path1]
        img2 = self.image_cache[img_path2]
        
        return img1, img2, torch.tensor(label, dtype=torch.float32)


def create_cached_loaders(cache_dir='./dataset_cache', batch_size=32, num_workers=0):
    """从缓存创建数据加载器"""
    
    train_dataset = CachedDataset(os.path.join(cache_dir, 'train_cache.pkl'))
    val_dataset = CachedDataset(os.path.join(cache_dir, 'val_cache.pkl'))
    test_dataset = CachedDataset(os.path.join(cache_dir, 'test_cache.pkl'))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    
    print(f"\nCached Dataset Sizes:")
    print(f"  Train: {len(train_dataset)} pairs")
    print(f"  Val: {len(val_dataset)} pairs")
    print(f"  Test: {len(test_dataset)} pairs")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    root_dir = r'C:\Users\ASUS\Desktop\task3\data'
    
    cacher = DatasetCacher(root_dir, cache_dir='./dataset_cache', seed=42)
    cacher.cache_all_datasets(pairs_per_class=17)
    
    # 验证缓存
    print("\nVerifying caches...")
    train_loader, val_loader, test_loader = create_cached_loaders(
        cache_dir='./dataset_cache', batch_size=4
    )
    
    # 测试一个批次
    for img1, img2, labels in train_loader:
        print(f"\nSample batch:")
        print(f"  img1 shape: {img1.shape}")
        print(f"  img2 shape: {img2.shape}")
        print(f"  labels: {labels}")
        break
    
    print("\nCache generation complete!")

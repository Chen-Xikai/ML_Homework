import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def get_random_writer_split(seed=42):
    # 按顺序划分：1-12训练，13-16验证，17-20测试
    train_writers = list(range(1, 13))   # [1,2,...,12]
    val_writers = list(range(13, 17))    # [13,14,15,16]
    test_writers = list(range(17, 21))   # [17,18,19,20]
    return train_writers, val_writers, test_writers


class OmniglotDataset:
    def __init__(self, root_dir, split='background'):
        self.root_dir = root_dir
        self.data_dir = os.path.join(root_dir, f'images_{split}')
        self.alphabets = []
        self.alphabet_to_chars = {}
        self.char_to_images = {}
        self._load_data()

    def _load_data(self):
        for alphabet in sorted(os.listdir(self.data_dir)):
            alphabet_path = os.path.join(self.data_dir, alphabet)
            if not os.path.isdir(alphabet_path):
                continue
            self.alphabets.append(alphabet)
            self.alphabet_to_chars[alphabet] = []
            for char_dir in sorted(os.listdir(alphabet_path)):
                char_path = os.path.join(alphabet_path, char_dir)
                if not os.path.isdir(char_path):
                    continue
                images = [os.path.join(char_path, img)
                         for img in sorted(os.listdir(char_path))
                         if img.endswith('.png')]
                if len(images) > 0:
                    self.alphabet_to_chars[alphabet].append(char_dir)
                    self.char_to_images[f"{alphabet}/{char_dir}"] = images

    def get_all_characters(self):
        return list(self.char_to_images.keys())


class WriterSplitDataset(Dataset):
    def __init__(self, omniglot_dataset, allowed_alphabets=None,
                 allowed_writers=None, pairs_per_class=20, transform=None):
        self.omniglot = omniglot_dataset
        self.allowed_writers = set(allowed_writers) if allowed_writers else None
        self.pairs_per_class = pairs_per_class
        self.transform = transform
        self.char_images = {}
        self.chars = []
        self._organize_data(allowed_alphabets)
        self.pairs = []
        self.labels = []
        self._generate_pairs()
        # 图像缓存
        self.image_cache = {}
        self._warmup_cache()

    def _organize_data(self, allowed_alphabets):
        for char_id, images in self.omniglot.char_to_images.items():
            alphabet = char_id.split('/')[0]
            if allowed_alphabets is not None and alphabet not in allowed_alphabets:
                continue
            filtered_images = []
            for img_path in images:
                filename = os.path.basename(img_path)
                writer_id = int(filename.split('_')[-1].split('.')[0])
                if self.allowed_writers is None or writer_id in self.allowed_writers:
                    # 验证图片是否可读
                    try:
                        img = Image.open(img_path)
                        img.verify()
                        filtered_images.append(img_path)
                    except Exception:
                        continue
            if len(filtered_images) >= 2:
                self.char_images[char_id] = filtered_images
                self.chars.append(char_id)

    def _generate_pairs(self):
        for char_id, images in self.char_images.items():
            if len(images) < 2:
                continue
            for _ in range(self.pairs_per_class):
                img1, img2 = random.sample(images, 2)
                self.pairs.append((img1, img2))
                self.labels.append(0)
        num_positive = len(self.labels)
        num_negative = 0
        while num_negative < num_positive:
            char1, char2 = random.sample(self.chars, 2)
            if char1 == char2:
                continue
            img1 = random.choice(self.char_images[char1])
            img2 = random.choice(self.char_images[char2])
            self.pairs.append((img1, img2))
            self.labels.append(1)
            num_negative += 1

    def _warmup_cache(self):
        """预加载所有图像到缓存"""
        all_paths = set()
        for img1, img2 in self.pairs:
            all_paths.add(img1)
            all_paths.add(img2)
        print(f"  Caching {len(all_paths)} images...")
        skipped = 0
        for img_path in all_paths:
            try:
                img = Image.open(img_path).convert('L')
                if self.transform:
                    self.image_cache[img_path] = self.transform(img)
                else:
                    self.image_cache[img_path] = transforms.ToTensor()(img)
            except Exception:
                skipped += 1
                continue
        if skipped > 0:
            print(f"  Warning: skipped {skipped} corrupted images")
        print(f"  Cache warmup done! Cached: {len(self.image_cache)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path1, img_path2 = self.pairs[idx]
        label = self.labels[idx]
        # 从缓存读取图像
        img1 = self.image_cache[img_path1]
        img2 = self.image_cache[img_path2]
        return img1, img2, torch.tensor(label, dtype=torch.float32)


def get_data_transforms(training=True):
    return transforms.Compose([
        transforms.Resize((105, 105)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])


def create_data_loaders(root_dir, batch_size=32, pairs_per_class=20,
                       num_workers=4, seed=42):
    train_writers, val_writers, test_writers = get_random_writer_split(seed)

    print(f"\nWriters split (seed={seed}):")
    print(f"  Train: {sorted(train_writers)}")
    print(f"  Val: {sorted(val_writers)}")
    print(f"  Test: {sorted(test_writers)}")

    background_dataset = OmniglotDataset(root_dir, split='background')
    evaluation_dataset = OmniglotDataset(root_dir, split='evaluation')

    all_background_alphabets = list(background_dataset.alphabet_to_chars.keys())

    # 训练集30个（含Braille/Balinese），验证集10个（不含Braille/Balinese）
    excluded_from_val = ['Braille', 'Balinese']

    # 优先把Braille和Balinese放入训练集（2个）
    train_alphabets = [a for a in all_background_alphabets if a in excluded_from_val]
    remaining = [a for a in all_background_alphabets if a not in excluded_from_val]

    random.seed(seed)
    random.shuffle(remaining)

    # 剩余38个中取28个给训练集，10个给验证集
    train_alphabets.extend(remaining[:28])
    val_alphabets = remaining[28:]

    assert len(train_alphabets) == 30, f"train_alphabets={len(train_alphabets)}"
    assert len(val_alphabets) == 10, f"val_alphabets={len(val_alphabets)}"
    assert 'Braille' in train_alphabets
    assert 'Balinese' in train_alphabets
    assert 'Braille' not in val_alphabets
    assert 'Balinese' not in val_alphabets

    print(f"\nAlphabet split:")
    print(f"  Train: {len(train_alphabets)} alphabets (no Braille/Balinese)")
    print(f"  Val: {len(val_alphabets)} alphabets (no Braille/Balinese)")
    print(f"  Test: 10 alphabets (user specified)")

    train_dataset = WriterSplitDataset(background_dataset, allowed_alphabets=train_alphabets,
                                       allowed_writers=train_writers, pairs_per_class=pairs_per_class,
                                       transform=get_data_transforms(training=True))
    val_dataset = WriterSplitDataset(background_dataset, allowed_alphabets=val_alphabets,
                                     allowed_writers=val_writers, pairs_per_class=pairs_per_class // 2,
                                     transform=get_data_transforms(training=False))
    test_dataset = WriterSplitDataset(evaluation_dataset, allowed_alphabets=None,
                                      allowed_writers=test_writers, pairs_per_class=pairs_per_class // 2,
                                      transform=get_data_transforms(training=False))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)} pairs")
    print(f"  Val: {len(val_dataset)} pairs")
    print(f"  Test: {len(test_dataset)} pairs")
    print(f"  Total: {len(train_dataset) + len(val_dataset) + len(test_dataset)} pairs")

    return train_loader, val_loader, test_loader


def get_test_alphabets(root_dir):
    return ['Atlantean', 'Ge_ez', 'Glagolitic', 'Gurmukhi', 'Kannada',
            'Malayalam', 'Manipuri', 'Old_Church_Slavonic_(Cyrillic)',
            'Tengwar', 'Tibetan']

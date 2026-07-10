"""
omniglot_dataset.py —— Omniglot 数据集模块

功能: 下载 → 加载 → 按论文规则划分字母表和书写者 → 生成训练/验证样本对
     → 生成单样本测试任务（episode） → 封装为 PyTorch DataLoader
     → 支持数据集缓存以加速断点续训

论文数据划分（Section 4.1 & 4.2）:
  第1级（Lake 原始划分）: 50 字母表 → 40 背景集 + 10 评估集
  第2级（细粒度划分）:
    训练: 30 字母表 + 12 书写者（取自背景集）
    验证: 10 字母表 + 4 书写者（背景集剩余部分）
    测试: 10 字母表 + 4 书写者（评估集全部，与 Lake 原始划分一致）
"""

import os
import sys
import zipfile
import pickle
import random
import urllib.request
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ======================== 下载 ========================

OMNIGLOT_URLS = {
    "images_background": "https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip",
    "images_evaluation": "https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip",
}


def download_omniglot(data_dir: str) -> bool:
    """
    从 GitHub 自动下载 Omniglot 数据集的两个 zip 包并解压。

    参数:
        data_dir: 数据存放目录

    返回:
        True 表示下载成功，False 表示失败（需手动下载）
    """
    os.makedirs(data_dir, exist_ok=True)

    success = True
    for name, url in OMNIGLOT_URLS.items():
        zip_path = os.path.join(data_dir, f"{name}.zip")
        extract_dir = os.path.join(data_dir, name)

        # 已解压则跳过
        if os.path.exists(extract_dir):
            print(f"[OK]  {name} 已存在，跳过下载")
            continue

        # 已有 zip 文件则直接解压
        if not os.path.exists(zip_path):
            print(f"正在下载 {name}.zip（约 50MB）...")
            try:
                urllib.request.urlretrieve(url, zip_path)
                print(f"[OK]  下载完成")
            except Exception as e:
                print(f"[FAIL]  下载失败: {e}")
                print(f"  请手动从 {url} 下载，放入 {data_dir} 目录")
                print(f"  然后重新运行此脚本")
                success = False
                continue

        # 解压
        print(f"正在解压 {name}.zip...")
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(data_dir)
            print(f"[OK]  解压完成")
        except Exception as e:
            print(f"[FAIL]  解压失败: {e}")
            success = False

    return success


# ======================== 加载原始数据 ========================

def load_raw_data(data_dir: str) -> Dict[tuple, Image.Image]:
    """
    遍历解压后的文件夹树，加载所有 105×105 二值字符图。

    Omniglot 目录结构:
        images_background/
          Alphabet_Name/
            character01/
              0709_01.png    ← drawer 1
              0709_02.png    ← drawer 2
              ...
            character02/
              ...

    返回:
        {(alphabet_name, character_id, drawer_id): PIL_Image, ...}
    """
    raw_data = {}
    total_images = 0

    for split_name in ["images_background", "images_evaluation"]:
        split_dir = os.path.join(data_dir, split_name)
        if not os.path.exists(split_dir):
            print(f"[WARN]  目录不存在: {split_dir}，跳过")
            continue

        for alphabet_name in sorted(os.listdir(split_dir)):
            alpha_dir = os.path.join(split_dir, alphabet_name)
            if not os.path.isdir(alpha_dir):
                continue

            for char_dir_name in sorted(os.listdir(alpha_dir)):
                char_path = os.path.join(alpha_dir, char_dir_name)
                if not os.path.isdir(char_path):
                    continue

                # 提取 character_id（去掉 "character" 前缀，如 "character01" → 1）
                try:
                    char_id = int(char_dir_name.replace("character", ""))
                except ValueError:
                    char_id = char_dir_name

                for img_filename in sorted(os.listdir(char_path)):
                    if not img_filename.endswith(".png"):
                        continue

                    img_path = os.path.join(char_path, img_filename)

                    # 从文件名提取 drawer_id（如 "0709_01.png" → 1）
                    try:
                        drawer_id = int(img_filename.split("_")[-1].replace(".png", ""))
                    except ValueError:
                        drawer_id = img_filename

                    img = Image.open(img_path).copy()
                    # 确保是 105×105 的灰度图
                    if img.size != (105, 105):
                        img = img.resize((105, 105), Image.BILINEAR)
                    if img.mode != "L":
                        img = img.convert("L")

                    raw_data[(alphabet_name, char_id, drawer_id)] = img
                    total_images += 1

    print(f"[OK]  加载完成: {len(raw_data)} 张图片")
    return raw_data


# ======================== 数据划分 ========================

def split_alphabets_drawers(raw_data: dict, config):
    """
    实现论文两级划分。

    第1级: 50 字母表 → 40 背景集 + 10 评估集
    第2级:
      背景集 → 训练 30 + 验证 10
      评估集 → 测试 10

    书写者划分: 20 个 → 训练 12 + 验证 4 + 测试 4

    返回:
        train_chars: [(alphabet, char_id, drawer_id, PIL_Image), ...]
        val_chars:   [(alphabet, char_id, drawer_id, PIL_Image), ...]
        test_chars:  [(alphabet, char_id, drawer_id, PIL_Image), ...]
    """
    # 收集所有字母表名
    all_alphabets = sorted(set(k[0] for k in raw_data.keys()))
    print(f"  总字母表数: {len(all_alphabets)}")
    print(f"  总字符数:   {len(set((k[0], k[1]) for k in raw_data.keys()))}")

    # 收集所有书写者 ID
    all_drawers = sorted(set(k[2] for k in raw_data.keys()))
    print(f"  总书写者数: {len(all_drawers)}")

    # 使用固定随机种子划分字母表（保证可复现）
    rng = random.Random(config.random_seed)
    shuffled_alphabets = list(all_alphabets)
    rng.shuffle(shuffled_alphabets)

    # 40 背景 + 10 评估
    background_alphabets = set(shuffled_alphabets[:40])
    evaluation_alphabets = set(shuffled_alphabets[40:50])

    # 从背景集中取 30 训练 + 10 验证
    bg_list = sorted(background_alphabets)
    rng.shuffle(bg_list)
    train_alphabets = set(bg_list[:30])
    val_alphabets = set(bg_list[30:40])

    print(f"  训练字母表: {len(train_alphabets)}")
    print(f"  验证字母表: {len(val_alphabets)}")
    print(f"  测试字母表: {len(evaluation_alphabets)}")

    # 划分书写者（确保训练/验证/测试不重叠）
    shuffled_drawers = list(all_drawers)
    rng.shuffle(shuffled_drawers)
    train_drawers = set(shuffled_drawers[:12])
    val_drawers = set(shuffled_drawers[12:16])
    test_drawers = set(shuffled_drawers[16:20])

    print(f"  训练书写者: {len(train_drawers)}（{sorted(train_drawers)}）")
    print(f"  验证书写者: {len(val_drawers)}（{sorted(val_drawers)}）")
    print(f"  测试书写者: {len(test_drawers)}（{sorted(test_drawers)}）")

    # 按划分归类
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

    print(f"  训练集图片: {len(train_chars)}")
    print(f"  验证集图片: {len(val_chars)}")
    print(f"  测试集图片: {len(test_chars)}")

    return train_chars, val_chars, test_chars


# ======================== 数据集缓存 ========================

def cache_dataset_split(train_chars, val_chars, test_chars, cache_path: str):
    """将划分结果缓存到磁盘（加速断点续训）"""
    # 只缓存轻量信息（不含图片对象）
    cache_data = {
        "train": [(a, c, d) for a, c, d, _ in train_chars],
        "val":   [(a, c, d) for a, c, d, _ in val_chars],
        "test":  [(a, c, d) for a, c, d, _ in test_chars],
    }
    with open(cache_path, "wb") as f:
        pickle.dump(cache_data, f)
    print(f"[OK]  数据集划分已缓存到 {cache_path}")


def load_cached_split(cache_path: str, raw_data: dict):
    """
    从缓存恢复划分结果。

    返回:
        (train_chars, val_chars, test_chars) 或 None（缓存不存在时）
    """
    if not os.path.exists(cache_path):
        return None

    print(f"[OK]  检测到数据集缓存 {cache_path}，正在恢复...")
    with open(cache_path, "rb") as f:
        cache_data = pickle.load(f)

    train_chars = []
    val_chars = []
    test_chars = []

    for alphabet, char_id, drawer_id in cache_data["train"]:
        key = (alphabet, char_id, drawer_id)
        if key in raw_data:
            train_chars.append((alphabet, char_id, drawer_id, raw_data[key]))

    for alphabet, char_id, drawer_id in cache_data["val"]:
        key = (alphabet, char_id, drawer_id)
        if key in raw_data:
            val_chars.append((alphabet, char_id, drawer_id, raw_data[key]))

    for alphabet, char_id, drawer_id in cache_data["test"]:
        key = (alphabet, char_id, drawer_id)
        if key in raw_data:
            test_chars.append((alphabet, char_id, drawer_id, raw_data[key]))

    print(f"  恢复训练集: {len(train_chars)} 张")
    print(f"  恢复验证集: {len(val_chars)} 张")
    print(f"  恢复测试集: {len(test_chars)} 张")

    return train_chars, val_chars, test_chars


# ======================== 字符索引 ========================

def build_char_index(char_list: list) -> Dict[tuple, list]:
    """
    按 (alphabet, char_id) 分组，方便快速查询某字符有哪些书写者版本。

    返回:
        {(alphabet, char_id): [PIL_Image, ...], ...}
    """
    char_index = defaultdict(list)
    for alphabet, char_id, drawer_id, img in char_list:
        char_index[(alphabet, char_id)].append((drawer_id, img))
    return dict(char_index)


# ======================== 样本对生成 ========================

def generate_pairs(char_index: dict,
                   num_pairs: int,
                   rng: random.Random) -> List[Tuple[Image.Image, Image.Image, int]]:
    """
    从字符索引中随机采样指定数量的正负样本对（1:1 均衡）。

    正样本对（same）: 同一字符，取两个不同书写者的图像 → label=1
    负样本对（different）: 两个不同字符，各取一张图 → label=0

    参数:
        char_index: {(alphabet, char_id): [(drawer_id, PIL_Image), ...]}
        num_pairs: 总样本对数
        rng: 随机数生成器

    返回:
        [(img1, img2, label), ...]
    """
    # 展开所有 (alphabet, char_id) 键
    char_keys = list(char_index.keys())
    if len(char_keys) < 2:
        raise ValueError(f"字符数不足: 仅有 {len(char_keys)} 个字符，无法生成样本对")

    half = num_pairs // 2
    pairs = []

    # --- 正样本对 ---
    # 只保留有至少 2 个书写者的字符键
    valid_keys = [k for k in char_keys if len(char_index[k]) >= 2]
    if not valid_keys:
        raise ValueError(f"没有字符具备 >= 2 个书写者（共 {len(char_keys)} 个字符），无法生成正样本对")
    for _ in range(half):
        key = rng.choice(valid_keys)
        entries = char_index[key]
        e1, e2 = rng.sample(entries, 2)
        pairs.append((e1[1], e2[1], 1))

    # --- 负样本对 ---
    for _ in range(half):
        key1, key2 = rng.sample(char_keys, 2)
        img1 = rng.choice(char_index[key1])[1]
        img2 = rng.choice(char_index[key2])[1]
        pairs.append((img1, img2, 0))

    rng.shuffle(pairs)
    return pairs


# ======================== 单样本测试任务生成 ========================

def generate_one_shot_trials(char_index: dict,
                             drawer_pool: set,
                             n_way: int,
                             n_trials: int,
                             rng: random.Random,
                             return_meta: bool = False) -> list:
    """
    生成单样本分类测试任务（论文标准流程）。

    论文测试流程（Section 4.3）:
      1. 对每个字母表（10个），获取该字母表的所有书写者
      2. 将4个书写者随机分成2+2两队：支撑队(2人) + 查询队(2人)
      3. 从支撑队中随机选1人，提供20个字符的支撑图
      4. 对查询队中每个人（2人），各提供20次测试（每个字符1次）
      5. 每字母表: 2查询者 × 20字符 = 40次测试
      6. 总计: 10字母表 × 40次 = 400次测试

    参数:
        char_index: {(alphabet, char_id): [(drawer_id, PIL_Image), ...]}
        drawer_pool: 可选的书写者 ID 集合
        n_way: 每次测试的类别数（20）
        n_trials: 单样本分类决策总次数（400）
        rng: 随机数生成器
        return_meta: 是否返回字母表名称

    返回:
        若 return_meta=False:
          [(test_img, [candidate1, ..., candidateN], correct_idx), ...]
          共 n_trials 条
        若 return_meta=True:
          ([...], [alphabet_name, ...])
    """
    # 按字母表分组的字符键
    alpha_groups = defaultdict(list)
    for key in char_index.keys():
        alpha_groups[key[0]].append(key)

    # 只保留有足够字符和书写者的字母表（至少4个书写者用于2+2划分）
    valid_alphas = []
    for alpha, keys in alpha_groups.items():
        if len(keys) >= n_way:
            # 获取该字母表在drawer_pool中的所有书写者
            alpha_drawers = set()
            for key in keys:
                for d_id, _ in char_index[key]:
                    if d_id in drawer_pool:
                        alpha_drawers.add(d_id)
            # 需要至少4个书写者才能进行2+2划分
            if len(alpha_drawers) >= 4:
                valid_alphas.append(alpha)

    if not valid_alphas:
        raise ValueError(f"没有字母表同时满足: >= {n_way} 个字符 + >= 4 个书写者")

    # 如果字母表数量超过需要，随机选择
    if len(valid_alphas) > n_trials // (2 * n_way):
        rng.shuffle(valid_alphas)
        valid_alphas = valid_alphas[:n_trials // (2 * n_way)]

    trials = []
    meta = []

    # 对每个字母表生成40次测试
    for alpha in valid_alphas:
        # 获取该字母表在drawer_pool中的所有书写者
        alpha_drawers = set()
        keys = alpha_groups[alpha]
        for key in keys:
            for d_id, _ in char_index[key]:
                if d_id in drawer_pool:
                    alpha_drawers.add(d_id)

        # 确保有至少4个书写者
        if len(alpha_drawers) < 4:
            continue

        # 随机分成2+2两队
        alpha_drawers_list = sorted(alpha_drawers)
        rng.shuffle(alpha_drawers_list)
        support_team = alpha_drawers_list[:2]  # 支撑队
        query_team = alpha_drawers_list[2:4]   # 查询队

        # 随机选n_way个字符
        selected_keys = rng.sample(keys, n_way)

        # 从支撑队中随机选1人提供支撑图
        support_drawer = rng.choice(support_team)

        # 构建支撑集：支撑队该书写者的n_way张图（每类1张）
        support_imgs = []
        for key in selected_keys:
            imgs = [img for d_id, img in char_index[key] if d_id == support_drawer]
            if imgs:
                support_imgs.append(imgs[0])
            else:
                # 理论上不应该发生，因为已经验证过
                support_imgs.append(None)

        # 对每个查询书写者
        for query_drawer in query_team:
            # 对每个字符生成1次测试
            for test_idx in range(n_way):
                test_key = selected_keys[test_idx]
                imgs = [img for d_id, img in char_index[test_key] if d_id == query_drawer]
                if imgs:
                    test_img = imgs[0]
                    trials.append((test_img, list(support_imgs), test_idx))
                    if return_meta:
                        meta.append(alpha)

    # 截断或填充到n_trials
    if len(trials) > n_trials:
        trials = trials[:n_trials]
        if return_meta:
            meta = meta[:n_trials]
    elif len(trials) < n_trials:
        print(f"  [WARN] 生成了 {len(trials)} 个测试，少于目标 {n_trials}")

    if return_meta:
        return trials, meta
    return trials


# ======================== PyTorch Dataset 类 ========================

class OmniglotPairDataset(Dataset):
    """
    二分类验证任务的图片对 Dataset。
    支持可选的在线仿射增强。

    当 return_pil=True 时，返回原始 PIL Image（供 collate 函数做批量增强）；
    当 return_pil=False 时，返回 Tensor（供标准 DataLoader 直接使用）。
    """

    def __init__(self, pair_list: list, augmentation_fn=None, return_pil: bool = False):
        """
        参数:
            pair_list: [(img1_PIL, img2_PIL, label), ...]
            augmentation_fn: RandomAffineTransform 实例或 None
            return_pil: True=返回PIL图像（配合CollateWithAugmentation使用）
        """
        self.pair_list = pair_list
        self.augmentation_fn = augmentation_fn
        self.return_pil = return_pil

    def __len__(self):
        return len(self.pair_list)

    def __getitem__(self, idx):
        img1, img2, label = self.pair_list[idx]

        # 在线仿射增强（仅在非 return_pil 模式下生效）
        if self.augmentation_fn is not None and not self.return_pil:
            img1 = self.augmentation_fn(img1)
            img2 = self.augmentation_fn(img2)

        if self.return_pil:
            # 返回 PIL 图像，由 collate 函数统一做增强+转Tensor
            return img1, img2, float(label)
        else:
            # PIL → Tensor (1, H, W)，归一化到 [0, 1]
            img1 = torch.from_numpy(np.array(img1, dtype=np.float32) / 255.0).unsqueeze(0)
            img2 = torch.from_numpy(np.array(img2, dtype=np.float32) / 255.0).unsqueeze(0)
            label = torch.tensor(float(label), dtype=torch.float32)
            return img1, img2, label


class OneShotEpisodeDataset(Dataset):
    """
    单样本测试 episodic Dataset。
    每个 __getitem__ 返回一次完整的测试任务。
    """

    def __init__(self, trials: list):
        """
        参数:
            trials: 由 generate_one_shot_trials 返回的测试任务列表
        """
        self.trials = trials

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        test_img, support_imgs, correct_idx = self.trials[idx]

        test_tensor = torch.from_numpy(
            np.array(test_img, dtype=np.float32) / 255.0
        ).unsqueeze(0)

        support_tensors = torch.stack([
            torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).unsqueeze(0)
            for img in support_imgs
        ])

        return test_tensor, support_tensors, correct_idx


# ======================== DataLoader 组装函数 ========================

def get_train_loader(config):
    """
    组装训练 DataLoader。
    - 生成训练样本对（数量由 config.train_pairs 决定）
    - Dataset 直接返回 Tensor
    - num_transforms=0 时无增强，batch_size=128，不做扩展
    - num_transforms>0 时由 augment_batch_on_gpu 在训练循环中增强
    """
    train_char_index = get_train_loader._train_char_index
    rng = random.Random(config.random_seed)

    print(f"正在生成 {config.train_pairs:,} 个训练样本对...")
    pairs = generate_pairs(train_char_index, config.train_pairs, rng)
    print(f"  实际生成: {len(pairs):,} 对（正 {sum(1 for _,_,l in pairs if l==1):,} / "
          f"负 {sum(1 for _,_,l in pairs if l==0):,}）")

    # Dataset 直接返回 Tensor（无增强），collate 用默认 torch.stack
    dataset = OmniglotPairDataset(pairs, augmentation_fn=None, return_pil=False)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        # 不传 collate_fn，PyTorch 自动 stack tensors
    )

    return loader


def get_val_loader(config):
    """组装验证 DataLoader（10k 二分类样本对，无增强）"""
    val_char_index = get_val_loader._val_char_index
    rng = random.Random(config.random_seed)

    print(f"正在生成 {config.val_pairs:,} 个验证样本对...")
    pairs = generate_pairs(val_char_index, config.val_pairs, rng)

    dataset = OmniglotPairDataset(pairs, augmentation_fn=None)

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    return loader


def get_val_one_shot_loader(config, val_drawer_pool: set):
    """
    组装早停单样本验证 DataLoader（320 次 20-way 测试）。
    用于训练过程中每个 epoch 结束后的早停判断。
    """
    val_char_index = get_val_loader._val_char_index
    rng = random.Random(config.random_seed + 100)

    trials = generate_one_shot_trials(
        val_char_index, val_drawer_pool,
        config.n_way, config.val_one_shot_trials, rng,
    )

    dataset = OneShotEpisodeDataset(trials)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    return loader


def get_test_one_shot_loader(config, test_drawer_pool: set):
    """
    组装最终评估集单样本测试 DataLoader（400 次 20-way 测试）。
    书写者限定为评估集的 4 个专属书写者。
    """
    test_char_index = get_test_one_shot_loader._test_char_index
    rng = random.Random(config.random_seed + 200)

    trials, alphabets = generate_one_shot_trials(
        test_char_index, test_drawer_pool,
        config.n_way, config.test_one_shot_trials, rng,
        return_meta=True,
    )

    dataset = OneShotEpisodeDataset(trials)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    return loader, alphabets


if __name__ == "__main__":
    # 自测: 下载 → 加载 → 划分 → 生成样本对 → 生成测试任务 → DataLoader
    sys.path.insert(0, ".")
    from config import Config

    config = Config()
    config.ensure_dirs()

    print("=" * 50)
    print("Omniglot 数据集模块自测")
    print("=" * 50)

    # 1. 下载
    print("\n[1/6] 下载数据集...")
    download_omniglot(config.data_dir)

    # 2. 加载
    print("\n[2/6] 加载原始数据...")
    raw_data = load_raw_data(config.data_dir)

    # 3. 划分
    print("\n[3/6] 数据划分...")
    train_chars, val_chars, test_chars = split_alphabets_drawers(raw_data, config)

    # 4. 构建字符索引
    train_ci = build_char_index(train_chars)
    val_ci = build_char_index(val_chars)
    test_ci = build_char_index(test_chars)
    print(f"[OK]  字符索引: 训练{len(train_ci)} / 验证{len(val_ci)} / 测试{len(test_ci)}")

    # 5. 生成样本对
    print("\n[4/6] 生成训练样本对...")
    rng = random.Random(42)
    pairs = generate_pairs(train_ci, 1000, rng)
    pos = sum(1 for _, _, l in pairs if l == 1)
    neg = sum(1 for _, _, l in pairs if l == 0)
    print(f"[OK]  样本对: 共{len(pairs)}对（正{pos}, 负{neg}）")

    # 6. DataLoader 测试
    print("\n[5/6] DataLoader 测试...")
    dataset = OmniglotPairDataset(pairs[:128], augmentation_fn=None)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    imgs1, imgs2, labels = next(iter(loader))
    print(f"[OK]  batch shape: img1={imgs1.shape}, img2={imgs2.shape}, label={labels.shape}")

    # 7. 单样本测试任务
    print("\n[6/6] 单样本测试任务生成...")
    val_drawer_set = set(d[2] for d in val_chars)
    trials = generate_one_shot_trials(val_ci, val_drawer_set, 20, 10, rng)
    print(f"[OK]  生成了 {len(trials)} 次 20-way 测试任务")

    print("\n[SUCCESS]  数据集模块自测全部通过！")

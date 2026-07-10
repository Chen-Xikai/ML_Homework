"""
affine_augmentation.py —— 仿射畸变在线增强

架构：
  - DataLoader collate: PIL → CPU Tensor（无增强，速度快）
  - 训练循环: 在 GPU 上批量生成仿射矩阵 + grid_sample 增强
  - GPU grid_sample 比 CPU 快 100 倍以上

论文原文（Section 3.2）:
  T = (θ, ρx, ρy, sx, sy, tx, ty), with θ ∈ [-10.0, 10.0],
  ρx, ρy ∈ [-0.3, 0.3], sx, sy ∈ [0.8, 1.2], tx, ty ∈ [-2, 2].
  Each component is included with probability 0.5.
"""

import numpy as np
import torch
import torch.nn.functional as F


# ======================== GPU 批量仿射增强 ========================

def generate_affine_thetas(batch_size: int,
                           img_size: int,
                           config,
                           device: torch.device) -> torch.Tensor:
    """
    批量生成随机 2×3 仿射矩阵（用于 F.affine_grid）。
    与论文完全一致：7 个参数 × 50% 概率独立生效。
    """
    prob = config.prob_apply

    mask = torch.rand(batch_size, device=device) < prob
    theta_deg = torch.zeros(batch_size, device=device)
    theta_deg[mask] = torch.rand(int(mask.sum().item()), device=device) * (
        config.theta_range[1] - config.theta_range[0]) + config.theta_range[0]
    theta_rad = torch.deg2rad(theta_deg)

    mask = torch.rand(batch_size, device=device) < prob
    rhox = torch.zeros(batch_size, device=device)
    rhox[mask] = torch.rand(int(mask.sum().item()), device=device) * (
        config.rhox_range[1] - config.rhox_range[0]) + config.rhox_range[0]

    mask = torch.rand(batch_size, device=device) < prob
    rhoy = torch.zeros(batch_size, device=device)
    rhoy[mask] = torch.rand(int(mask.sum().item()), device=device) * (
        config.rhoy_range[1] - config.rhoy_range[0]) + config.rhoy_range[0]

    mask = torch.rand(batch_size, device=device) < prob
    sx = torch.ones(batch_size, device=device)
    sx[mask] = torch.rand(int(mask.sum().item()), device=device) * (
        config.sx_range[1] - config.sx_range[0]) + config.sx_range[0]

    mask = torch.rand(batch_size, device=device) < prob
    sy = torch.ones(batch_size, device=device)
    sy[mask] = torch.rand(int(mask.sum().item()), device=device) * (
        config.sy_range[1] - config.sy_range[0]) + config.sy_range[0]

    mask = torch.rand(batch_size, device=device) < prob
    tx = torch.zeros(batch_size, device=device)
    tx[mask] = torch.rand(int(mask.sum().item()), device=device) * (
        config.tx_range[1] - config.tx_range[0]) + config.tx_range[0]

    mask = torch.rand(batch_size, device=device) < prob
    ty = torch.zeros(batch_size, device=device)
    ty[mask] = torch.rand(int(mask.sum().item()), device=device) * (
        config.ty_range[1] - config.ty_range[0]) + config.ty_range[0]

    cos_t = torch.cos(theta_rad)
    sin_t = torch.sin(theta_rad)

    a_fwd = sx * (cos_t - sin_t * rhoy)
    b_fwd = sy * (cos_t * rhox - sin_t)
    c_fwd = sx * (sin_t + cos_t * rhoy)
    d_fwd = sy * (sin_t * rhox + cos_t)

    det = a_fwd * d_fwd - b_fwd * c_fwd
    det = torch.where(torch.abs(det) < 1e-8, torch.sign(det) * 1e-8 + det, det)

    a_inv = d_fwd / det
    b_inv = -b_fwd / det
    c_inv = -c_fwd / det
    d_inv = a_fwd / det

    tx_inv = -(a_inv * tx + b_inv * ty)
    ty_inv = -(c_inv * tx + d_inv * ty)

    scale = img_size / 2.0
    theta = torch.zeros(batch_size, 2, 3, device=device)
    theta[:, 0, 0] = a_inv
    theta[:, 0, 1] = b_inv
    theta[:, 0, 2] = tx_inv / scale
    theta[:, 1, 0] = c_inv
    theta[:, 1, 1] = d_inv
    theta[:, 1, 2] = ty_inv / scale

    return theta


def apply_affine_batch(imgs: torch.Tensor,
                       thetas: torch.Tensor) -> torch.Tensor:
    """对一批图像批量施加仿射变换。"""
    grid = F.affine_grid(thetas, imgs.size(), align_corners=False)
    return F.grid_sample(imgs, grid, mode='bilinear',
                         padding_mode='zeros', align_corners=False)


def augment_batch_on_gpu(imgs1: torch.Tensor,
                         imgs2: torch.Tensor,
                         labels: torch.Tensor,
                         config,
                         num_copies: int = 8):
    """
    在 GPU 上将原始 batch 扩展为 9 倍（1 原始 + 8 增强）。

    优化：一次性生成所有增强份数的 thetas，单次 grid_sample 处理全部增强图。
    """
    B = imgs1.size(0)
    img_size = config.img_size
    device = imgs1.device
    total_aug = B * num_copies  # 全部增强图数量

    # 一次性生成所有 thetas（避免多次小 kernel launch）
    thetas1 = generate_affine_thetas(total_aug, img_size, config, device)
    thetas2 = generate_affine_thetas(total_aug, img_size, config, device)

    # 源图复制 num_copies 份一次性变换
    src1 = imgs1.repeat(num_copies, 1, 1, 1)   # (B*8, 1, H, W)
    src2 = imgs2.repeat(num_copies, 1, 1, 1)

    aug1 = apply_affine_batch(src1, thetas1)
    aug2 = apply_affine_batch(src2, thetas2)

    return (torch.cat([imgs1, aug1], dim=0),   # 原始 + 全部增强
            torch.cat([imgs2, aug2], dim=0),
            labels.repeat(num_copies + 1))


# ======================== DataLoader Collate ========================

def collate_pil_to_tensor(batch: list, config):
    """
    DataLoader collate 函数：纯 PIL → CPU Tensor 转换，不做增强。

    增强由 augment_batch_on_gpu 在训练循环中在 GPU 上完成。
    """
    pil_imgs1 = [item[0] for item in batch]
    pil_imgs2 = [item[1] for item in batch]
    labels_list = [float(item[2]) for item in batch]

    batch_arr1 = np.stack([np.array(im, dtype=np.float32) for im in pil_imgs1]) / 255.0
    batch_arr2 = np.stack([np.array(im, dtype=np.float32) for im in pil_imgs2]) / 255.0

    imgs1 = torch.from_numpy(batch_arr1).unsqueeze(1)
    imgs2 = torch.from_numpy(batch_arr2).unsqueeze(1)
    labels = torch.tensor(labels_list, dtype=torch.float32)

    return imgs1, imgs2, labels


class CollateWithAugmentation:
    """
    可 pickle 的 collate 包装类（Windows spawn 多进程兼容）。

    只做 PIL → CPU Tensor 转换，不做增强。
    增强由 augment_batch_on_gpu 在训练循环 GPU 上完成。
    """

    def __init__(self, config, num_copies: int = 8):
        self.config = config
        self.num_copies = num_copies  # 保留参数，供兼容

    def __call__(self, batch):
        return collate_pil_to_tensor(batch, self.config)


# ======================== 自测 ========================

if __name__ == "__main__":
    from config import Config
    from PIL import Image
    import pickle as _pickle

    config = Config()

    # 1. 简单 collate
    batch = [(Image.new("L", (105, 105), color=128),
              Image.new("L", (105, 105), color=200), 1.0) for _ in range(4)]
    imgs1, imgs2, labels = collate_pil_to_tensor(batch, config)
    assert imgs1.is_cpu and imgs1.shape == (4, 1, 105, 105)
    print(f"[OK] Collate PIL->Tensor: shape={list(imgs1.shape)}, CPU={imgs1.is_cpu}")

    # 2. GPU 批量增强
    if torch.cuda.is_available():
        device = torch.device('cuda')
        imgs1 = imgs1.to(device)
        imgs2 = imgs2.to(device)
        labels = labels.to(device)

        a1, a2, al = augment_batch_on_gpu(imgs1, imgs2, labels, config, num_copies=2)
        assert a1.device.type == 'cuda'
        assert a1.shape == (12, 1, 105, 105)  # 4*3
        print(f"[OK] GPU augment: shape={list(a1.shape)}, device={a1.device}")

    # 3. Pickle
    collate = CollateWithAugmentation(config)
    _pickle.dumps(collate)
    print(f"[OK] CollateWithAugmentation pickle OK")

    print(f"\nAll checks passed.")

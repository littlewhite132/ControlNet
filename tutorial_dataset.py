import glob
import os

import cv2
import numpy as np
from torch.utils.data import Dataset


TRAIN_ROOT = "/media/NAS01/yyfangspx/shaoyang_ct_png_minus-1000to400/train/Z"
VAL_ROOT = "/media/NAS01/yyfangspx/shaoyang_ct_png_minus-1000to400/val/Z"


class MyDataset(Dataset):
    def __init__(self, split: str = "train", target_size: int = 512):
        if split == "train":
            data_root = TRAIN_ROOT
        elif split == "val":
            data_root = VAL_ROOT
        else:
            raise ValueError(f"Unsupported split: {split}")
        self.target_size = target_size

        # 收集对应 split 下的 PNG 路径
        self.files = sorted(glob.glob(os.path.join(data_root, "*.png")))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 读入单通道纵隔窗，形状 [H, W]；灰度图无需 BGR->RGB 转换
        img = cv2.imread(self.files[idx], cv2.IMREAD_GRAYSCALE)  # uint8
        # 统一分辨率，避免 DataLoader collate 时大小不一致；默认缩放到 512x512
        if img.shape[0] != self.target_size or img.shape[1] != self.target_size:
            img = cv2.resize(img, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)
        # 扩展为 [H, W, 1]，再复制成 3 通道以匹配 hint_channels=3、VAE 3 通道输入
        img = img[:, :, None]
        img3 = np.repeat(img, 3, axis=2)

        # 归一化：ControlNet hint 用 [0,1]，主模型输入用 [-1,1]
        hint = img3.astype(np.float32) / 255.0                # [0,1]
        target = (img3.astype(np.float32) / 127.5) - 1.0       # [-1,1]

        return dict(
            jpg=target,   # 主模型目标（加噪前的 x_start）
            txt="",       # 空 prompt
            hint=hint     # ControlNet 条件
        )

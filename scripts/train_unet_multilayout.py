#!/usr/bin/env python3
"""
Train segmentation U-Net on multi-layout rendered MHI ECGs + phone augmentations.

Supports three data sources (can be combined):
1. Pre-rendered ACS/MHI image+mask pairs from disk
2. HuggingFace parquet dataset (loaded on-the-fly from parquet files)
3. Online rendering from NPY files (slowest)

Usage:
    cd /volume/Open-ECG-Digitizer

    # Combined HuggingFace + pre-rendered ACS:
    python scripts/train_unet_multilayout.py --device cuda:1 \
        --prerendered_dir data/acs_multilayout_combined/ \
        --hf_parquet_dir /media/data1/datasets/Huggingface_ECG_Digitize/parquet/data/ \
        --save_dir weights/unet_combined/

    # Pre-rendered only:
    python scripts/train_unet_multilayout.py --prerendered_dir data/acs_multilayout_masks/

    # Online rendering:
    python scripts/train_unet_multilayout.py --device cuda:0 --max_npys 100 --n_layouts 3
"""

import argparse
import glob
import io
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset

# Add project root to path
sys.path.insert(0, "/volume/Open-ECG-Digitizer")

from src.model.unet import UNet
from src.loss.loss import DiceFocalLoss
from scripts.multi_layout_renderer import (
    LAYOUTS, LEAD_NAMES, load_ecg_npy, render_ecg_with_mask, render_ecg_random_style,
)


# ---------------------------------------------------------------------------
# Phone-photo augmentations (reused from train_segmentation_phone_augment.py)
# ---------------------------------------------------------------------------

class PhonePhotoAugmentation:
    """Smartphone photo augmentations for segmentation training.

    Geometric transforms are applied to both image and mask.
    Photometric transforms are applied to image only.
    """

    def __init__(self, p=0.8, crop_size=1024):
        self.p = p
        self.crop_size = crop_size

    def __call__(self, img: np.ndarray, mask: np.ndarray):
        # Always random crop to crop_size x crop_size
        img, mask = self._random_crop(img, mask)

        if np.random.random() > self.p:
            return img, mask

        # --- Geometric (applied to both) ---
        if np.random.random() < 0.5:
            img, mask = self._perspective_warp(img, mask)
        if np.random.random() < 0.2:
            img, mask = self._paper_curl(img, mask)
        if np.random.random() < 0.3:
            img, mask = self._random_crop_margins(img, mask)
        if np.random.random() < 0.5:
            img, mask = self._random_flip(img, mask)

        # --- Photometric (image only) ---
        if np.random.random() < 0.7:
            img = self._brightness_contrast(img)
        if np.random.random() < 0.5:
            img = self._color_jitter(img)
        if np.random.random() < 0.4:
            img = self._random_shadow(img)
        if np.random.random() < 0.4:
            img = self._gradient_lighting(img)
        if np.random.random() < 0.3:
            img = self._color_temperature(img)
        if np.random.random() < 0.3:
            img = self._motion_blur(img)
        if np.random.random() < 0.4:
            img = self._gaussian_noise(img)
        if np.random.random() < 0.5:
            img = self._jpeg_compress(img)
        if np.random.random() < 0.3:
            img = self._downsample_upsample(img)
        if np.random.random() < 0.3:
            img = self._clahe_augment(img)

        return img, mask

    def _random_crop(self, img, mask):
        h, w = img.shape[:2]
        cs = self.crop_size
        if h < cs or w < cs:
            scale = max(cs / h, cs / w) * 1.05
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
            mask = cv2.resize(mask, (int(w * scale), int(h * scale)),
                              interpolation=cv2.INTER_NEAREST)
            h, w = img.shape[:2]
        y = np.random.randint(0, h - cs + 1)
        x = np.random.randint(0, w - cs + 1)
        return img[y:y+cs, x:x+cs], mask[y:y+cs, x:x+cs]

    def _perspective_warp(self, img, mask):
        h, w = img.shape[:2]
        offset = int(min(h, w) * np.random.uniform(0.02, 0.08))
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = src.copy()
        for i in range(4):
            dst[i, 0] += np.random.randint(-offset, offset + 1)
            dst[i, 1] += np.random.randint(-offset, offset + 1)
        M = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        mask = cv2.warpPerspective(mask, M, (w, h), borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(0, 0, 0), flags=cv2.INTER_NEAREST)
        return img, mask

    def _paper_curl(self, img, mask):
        h, w = img.shape[:2]
        amplitude = np.random.uniform(2, 6)
        frequency = np.random.uniform(1, 3)
        xs = np.arange(w, dtype=np.float32)
        ys = np.arange(h, dtype=np.float32)
        map_x = np.tile(xs[None, :], (h, 1))
        map_y = np.tile(ys[:, None], (1, w))
        map_y += amplitude * np.sin(2 * np.pi * frequency * map_x / w).astype(np.float32)
        img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        mask = cv2.remap(mask, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                         borderValue=(0, 0, 0))
        return img, mask

    def _random_crop_margins(self, img, mask):
        h, w = img.shape[:2]
        top = int(h * np.random.uniform(0, 0.10)) if np.random.random() < 0.5 else 0
        bottom = int(h * np.random.uniform(0, 0.10)) if np.random.random() < 0.5 else 0
        left = int(w * np.random.uniform(0, 0.10)) if np.random.random() < 0.5 else 0
        right = int(w * np.random.uniform(0, 0.10)) if np.random.random() < 0.5 else 0
        y1, y2 = top, max(h - bottom, top + 1)
        x1, x2 = left, max(w - right, left + 1)
        img = cv2.resize(img[y1:y2, x1:x2], (w, h))
        mask = cv2.resize(mask[y1:y2, x1:x2], (w, h), interpolation=cv2.INTER_NEAREST)
        return img, mask

    def _random_flip(self, img, mask):
        if np.random.random() < 0.5:
            img = np.flip(img, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        if np.random.random() < 0.3:
            img = np.flip(img, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()
        return img, mask

    def _brightness_contrast(self, img):
        alpha = np.random.uniform(0.7, 1.3)
        beta = np.random.randint(-30, 31)
        return np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    def _color_jitter(self, img):
        for c in range(3):
            shift = np.random.randint(-15, 16)
            img[:, :, c] = np.clip(img[:, :, c].astype(np.int16) + shift, 0, 255).astype(np.uint8)
        return img

    def _random_shadow(self, img):
        h, w = img.shape[:2]
        overlay = img.astype(np.float32)
        opacity = np.random.uniform(0.3, 0.7)
        cx, cy = np.random.randint(0, w), np.random.randint(0, h)
        ax = np.random.randint(w // 6, w // 2)
        ay = np.random.randint(h // 6, h // 2)
        angle = np.random.randint(0, 180)
        shadow_mask = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(shadow_mask, (cx, cy), (ax, ay), angle, 0, 360, 1.0, -1)
        ksize = max(31, min(h, w) // 4) | 1
        shadow_mask = cv2.GaussianBlur(shadow_mask, (ksize, ksize), 0)
        result = overlay * (1.0 - shadow_mask[:, :, None] * opacity)
        return np.clip(result, 0, 255).astype(np.uint8)

    def _gradient_lighting(self, img):
        h, w = img.shape[:2]
        strength = np.random.uniform(0.15, 0.30)
        if np.random.random() < 0.5:
            grad = np.linspace(1.0 - strength, 1.0 + strength, w, dtype=np.float32)
            grad = np.tile(grad[None, :, None], (h, 1, 3))
        else:
            grad = np.linspace(1.0 - strength, 1.0 + strength, h, dtype=np.float32)
            grad = np.tile(grad[:, None, None], (1, w, 3))
        return np.clip(img.astype(np.float32) * grad, 0, 255).astype(np.uint8)

    def _color_temperature(self, img):
        warm = np.random.random() < 0.5
        shift = np.random.randint(10, 30)
        result = img.copy()
        if warm:
            result[:, :, 2] = np.clip(result[:, :, 2].astype(np.int16) + shift, 0, 255).astype(np.uint8)
            result[:, :, 1] = np.clip(result[:, :, 1].astype(np.int16) + shift // 3, 0, 255).astype(np.uint8)
            result[:, :, 0] = np.clip(result[:, :, 0].astype(np.int16) - shift // 2, 0, 255).astype(np.uint8)
        else:
            result[:, :, 0] = np.clip(result[:, :, 0].astype(np.int16) + shift, 0, 255).astype(np.uint8)
            result[:, :, 2] = np.clip(result[:, :, 2].astype(np.int16) - shift // 2, 0, 255).astype(np.uint8)
        return result

    def _motion_blur(self, img):
        size = np.random.choice([3, 5, 7])
        angle = np.random.randint(0, 180)
        kernel = np.zeros((size, size))
        kernel[size // 2, :] = 1
        M = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1.0)
        kernel = cv2.warpAffine(kernel, M, (size, size))
        kernel /= kernel.sum() + 1e-9
        return cv2.filter2D(img, -1, kernel)

    def _gaussian_noise(self, img):
        sigma = np.random.uniform(5, 20)
        noise = np.random.normal(0, sigma, img.shape)
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    def _jpeg_compress(self, img):
        quality = np.random.randint(30, 80)
        _, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    def _downsample_upsample(self, img):
        h, w = img.shape[:2]
        factor = np.random.uniform(0.3, 0.7)
        small = cv2.resize(img, (max(1, int(w * factor)), max(1, int(h * factor))))
        return cv2.resize(small, (w, h))

    def _clahe_augment(self, img):
        clip_limit = np.random.uniform(1.0, 4.0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------------------
# Online rendering dataset (renders on-the-fly, no pre-generated images)
# ---------------------------------------------------------------------------

class OnlineRenderDataset(Dataset):
    """Renders ECGs in random layouts on-the-fly during training.

    Each __getitem__ call:
    1. Picks a random NPY file
    2. Picks a random layout
    3. Renders image + mask with random style variation
    4. Applies phone-photo augmentations
    5. Returns tensor pair

    This avoids needing to pre-generate millions of images.
    """

    def __init__(self, npy_paths: list, layout_names: list = None,
                 augment: PhonePhotoAugmentation = None,
                 amplitude_factor: float = 4.88, width: int = 2500,
                 epoch_size: int = 5000):
        self.npy_paths = npy_paths
        self.layout_names = layout_names or list(LAYOUTS.keys())
        self.augment = augment or PhonePhotoAugmentation()
        self.amplitude_factor = amplitude_factor
        self.width = width
        self.epoch_size = epoch_size

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        # Random NPY and layout
        npy_path = random.choice(self.npy_paths)
        layout = random.choice(self.layout_names)

        try:
            lead_dict = load_ecg_npy(npy_path)
        except Exception:
            # Fallback to random lead dict if file is bad
            lead_dict = {name: np.random.randn(2500) * 50 for name in LEAD_NAMES}
            lead_dict['-aVR'] = -lead_dict['aVR']

        # Render with random visual style
        img, mask = render_ecg_with_mask(
            lead_dict, layout, amplitude_factor=self.amplitude_factor,
            width=self.width, random_rhythm=True,
            # Random visual style for the image
            grid_color=random.choice(['red', '#FF6B6B', '#CC0000', '#00AA00',
                                       '#4488FF', '#FF8800']),
            grid_alpha=random.uniform(0.5, 1.0),
            linewidth=random.uniform(1.5, 4.0),
            show_labels=random.random() > 0.1,
        )

        # Convert to numpy (BGR for OpenCV augmentations)
        img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        mask_np = np.array(mask)  # Keep RGB for mask

        # Apply phone augmentations
        img_np, mask_np = self.augment(img_np, mask_np)

        # Convert to tensors: (C, H, W), float32 [0, 1]
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(mask_np).permute(2, 0, 1).float() / 255.0

        return img_t, mask_t


class PreRenderedDataset(Dataset):
    """Dataset from pre-rendered image+mask pairs on disk.

    Faster than OnlineRenderDataset since no matplotlib rendering per batch.
    Use generate_training_data() to create the files first.
    """

    def __init__(self, data_dir: str, augment: PhonePhotoAugmentation = None):
        self.augment = augment or PhonePhotoAugmentation()
        self.img_dir = os.path.join(data_dir, 'images')
        self.mask_dir = os.path.join(data_dir, 'masks')
        self.files = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.png')])
        print(f"PreRenderedDataset: {len(self.files)} image-mask pairs from {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img = cv2.imread(os.path.join(self.img_dir, fname))
        mask = cv2.imread(os.path.join(self.mask_dir, fname))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        if self.augment:
            img, mask = self.augment(img, mask)

        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(mask).permute(2, 0, 1).float() / 255.0
        return img_t, mask_t


# ---------------------------------------------------------------------------
# HuggingFace parquet dataset (loads directly from parquet shards)
# ---------------------------------------------------------------------------

class ParquetDataset(Dataset):
    """Dataset that loads ECG images+masks from HuggingFace parquet files.

    Each parquet shard has columns: img (dict with 'bytes'), mask (dict with 'bytes').
    Images and masks are stored as PNG bytes.
    """

    def __init__(self, parquet_dir: str, augment: PhonePhotoAugmentation = None,
                 max_images: int = None):
        import pyarrow.parquet as pq

        self.augment = augment or PhonePhotoAugmentation()
        self.parquet_dir = parquet_dir

        # Build global index: list of (shard_path, row_idx)
        shard_paths = sorted(glob.glob(os.path.join(parquet_dir, '*.parquet')))
        if not shard_paths:
            raise ValueError(f"No parquet files found in {parquet_dir}")

        self.index = []  # (shard_path, row_idx)
        self._shard_cache = {}  # shard_path -> (img_list, mask_list)

        for shard_path in shard_paths:
            table = pq.read_table(shard_path, columns=['img', 'mask'])
            n_rows = len(table)
            # Pre-load all image+mask bytes into memory for speed
            imgs = table.column('img').to_pylist()
            masks = table.column('mask').to_pylist()
            self._shard_cache[shard_path] = (imgs, masks)
            for i in range(n_rows):
                self.index.append((shard_path, i))
            del table

        if max_images and max_images < len(self.index):
            random.seed(42)
            self.index = random.sample(self.index, max_images)

        print(f"ParquetDataset: {len(self.index)} images from {len(shard_paths)} shards in {parquet_dir}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        shard_path, row_idx = self.index[idx]
        imgs, masks = self._shard_cache[shard_path]

        # Decode PNG bytes
        img_data = imgs[row_idx]
        mask_data = masks[row_idx]

        img_bytes = img_data['bytes'] if isinstance(img_data, dict) else img_data
        mask_bytes = mask_data['bytes'] if isinstance(mask_data, dict) else mask_data

        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        mask = Image.open(io.BytesIO(mask_bytes)).convert('RGB')

        img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        mask_np = np.array(mask)  # Keep RGB

        if self.augment:
            img_np, mask_np = self.augment(img_np, mask_np)

        img_t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(mask_np).permute(2, 0, 1).float() / 255.0
        return img_t, mask_t


# ---------------------------------------------------------------------------
# Pre-generate training data (image + mask pairs)
# ---------------------------------------------------------------------------

def generate_training_data(
    npy_dir: str,
    output_dir: str,
    n_images: int = 10000,
    layouts: list = None,
    amplitude_factor: float = 4.88,
    width: int = 2500,
):
    """Pre-generate image+mask pairs for faster training.

    Args:
        npy_dir: Directory with .npy ECG files
        output_dir: Output directory (creates images/ and masks/ subdirs)
        n_images: Total number of image-mask pairs to generate
        layouts: Layout names to use (default: all 13)
        amplitude_factor: Signal scaling
        width: Image width
    """
    if layouts is None:
        layouts = list(LAYOUTS.keys())

    img_dir = os.path.join(output_dir, 'images')
    mask_dir = os.path.join(output_dir, 'masks')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    npy_files = [os.path.join(npy_dir, f) for f in os.listdir(npy_dir) if f.endswith('.npy')]
    if not npy_files:
        raise ValueError(f"No .npy files found in {npy_dir}")

    print(f"Generating {n_images} image-mask pairs from {len(npy_files)} NPYs x {len(layouts)} layouts")

    for i in range(n_images):
        npy_path = random.choice(npy_files)
        layout = random.choice(layouts)

        try:
            lead_dict = load_ecg_npy(npy_path)
            img, mask = render_ecg_with_mask(
                lead_dict, layout, amplitude_factor=amplitude_factor,
                width=width, random_rhythm=True,
                grid_color=random.choice(['red', '#FF6B6B', '#CC0000', '#00AA00',
                                           '#4488FF', '#FF8800']),
                grid_alpha=random.uniform(0.5, 1.0),
                linewidth=random.uniform(1.5, 4.0),
                show_labels=random.random() > 0.1,
            )

            basename = os.path.splitext(os.path.basename(npy_path))[0]
            fname = f"{basename}_{layout}_{i:06d}.png"
            img.save(os.path.join(img_dir, fname))
            mask.save(os.path.join(mask_dir, fname))

            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{n_images}] {fname}")

        except Exception as e:
            print(f"  ERROR [{i}]: {e}")

    print(f"Done: {n_images} pairs saved to {output_dir}")


# ---------------------------------------------------------------------------
# Validation dataset (clean, no augmentation)
# ---------------------------------------------------------------------------

class ValDataset(Dataset):
    """Simple validation dataset from pre-rendered pairs."""

    def __init__(self, data_dir: str, crop_size: int = 1024):
        self.crop_size = crop_size
        self.img_dir = os.path.join(data_dir, 'images')
        self.mask_dir = os.path.join(data_dir, 'masks')
        self.files = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img = cv2.imread(os.path.join(self.img_dir, fname))
        mask = cv2.imread(os.path.join(self.mask_dir, fname))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        # Center crop
        h, w = img.shape[:2]
        cs = self.crop_size
        if h < cs or w < cs:
            scale = max(cs / h, cs / w) * 1.05
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
            mask = cv2.resize(mask, (int(w * scale), int(h * scale)),
                              interpolation=cv2.INTER_NEAREST)
            h, w = img.shape[:2]
        y = (h - cs) // 2
        x = (w - cs) // 2
        img = img[y:y+cs, x:x+cs]
        mask = mask[y:y+cs, x:x+cs]

        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(mask).permute(2, 0, 1).float() / 255.0
        return img_t, mask_t


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device(args.device)

    augment = PhonePhotoAugmentation(p=0.8, crop_size=args.crop_size)

    train_datasets = []
    val_datasets = []

    if args.prerendered_dir:
        # --- Pre-rendered mode: load from disk ---
        prerendered_dir = os.path.abspath(args.prerendered_dir)
        all_files = sorted([f for f in os.listdir(os.path.join(prerendered_dir, 'images'))
                            if f.endswith('.png')])
        random.seed(42)
        random.shuffle(all_files)
        n_val = max(50, len(all_files) // 10)  # 10% for validation
        val_files = all_files[:n_val]
        train_files = all_files[n_val:]
        print(f"PreRendered: {len(train_files)} train, {len(val_files)} val from {prerendered_dir}")

        save_dir = os.path.abspath(args.save_dir)
        src_img = os.path.join(prerendered_dir, 'images')
        src_mask = os.path.join(prerendered_dir, 'masks')

        # Create symlinked val dir
        val_dir = os.path.join(save_dir, 'val_data')
        os.makedirs(os.path.join(val_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(val_dir, 'masks'), exist_ok=True)
        for f in val_files:
            dst_img = os.path.join(val_dir, 'images', f)
            dst_mask = os.path.join(val_dir, 'masks', f)
            if not os.path.exists(dst_img):
                os.symlink(os.path.join(src_img, f), dst_img)
            if not os.path.exists(dst_mask):
                os.symlink(os.path.join(src_mask, f), dst_mask)

        # Create symlinked train dir
        train_dir = os.path.join(save_dir, 'train_data')
        os.makedirs(os.path.join(train_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(train_dir, 'masks'), exist_ok=True)
        for f in train_files:
            dst_img = os.path.join(train_dir, 'images', f)
            dst_mask = os.path.join(train_dir, 'masks', f)
            if not os.path.exists(dst_img):
                os.symlink(os.path.join(src_img, f), dst_img)
            if not os.path.exists(dst_mask):
                os.symlink(os.path.join(src_mask, f), dst_mask)

        train_datasets.append(PreRenderedDataset(train_dir, augment=augment))
        val_datasets.append(ValDataset(val_dir, crop_size=args.crop_size))

    if args.hf_parquet_dir:
        # --- HuggingFace parquet mode: load from parquet shards ---
        hf_full = ParquetDataset(args.hf_parquet_dir, augment=augment)
        # Split 90/10 for train/val
        n_hf = len(hf_full)
        n_hf_val = max(50, n_hf // 10)
        indices = list(range(n_hf))
        random.seed(42)
        random.shuffle(indices)
        hf_val_idx = indices[:n_hf_val]
        hf_train_idx = indices[n_hf_val:]
        hf_train = torch.utils.data.Subset(hf_full, hf_train_idx)
        # Val with no augmentation: create separate dataset
        hf_val_ds = ParquetDataset(args.hf_parquet_dir,
                                   augment=PhonePhotoAugmentation(p=0.0, crop_size=args.crop_size))
        hf_val = torch.utils.data.Subset(hf_val_ds, hf_val_idx)
        print(f"HuggingFace: {len(hf_train)} train, {len(hf_val)} val")
        train_datasets.append(hf_train)
        val_datasets.append(hf_val)

    if train_datasets:
        if len(train_datasets) == 1:
            train_dataset = train_datasets[0]
            val_dataset = val_datasets[0]
        else:
            train_dataset = ConcatDataset(train_datasets)
            val_dataset = ConcatDataset(val_datasets)
        print(f"Combined: {len(train_dataset)} train, {len(val_dataset)} val total")

    elif not train_datasets:
        # --- Online rendering mode (fallback when no prerendered/hf data) ---
        # Collect NPY files
        npy_files = sorted([
            os.path.join(args.npy_dir, f)
            for f in os.listdir(args.npy_dir)
            if f.endswith('.npy')
        ])
        if args.max_npys:
            npy_files = npy_files[:args.max_npys]
        print(f"Found {len(npy_files)} NPY files")

        # Split train/val (95/5)
        random.seed(42)
        random.shuffle(npy_files)
        n_val = min(max(5, len(npy_files) // 20), len(npy_files) // 2)
        val_npys = npy_files[:n_val]
        train_npys = npy_files[n_val:]
        print(f"Split: {len(train_npys)} train, {len(val_npys)} val NPYs")

        # Select layouts
        all_layouts = list(LAYOUTS.keys())
        if args.n_layouts and args.n_layouts < len(all_layouts):
            layouts = random.sample(all_layouts, args.n_layouts)
        else:
            layouts = all_layouts
        print(f"Using {len(layouts)} layouts: {layouts}")

        # --- Pre-generate val set ---
        val_dir = os.path.join(args.save_dir, 'val_data')
        if not os.path.exists(os.path.join(val_dir, 'images')):
            print("Generating validation data...")
            generate_training_data(
                args.npy_dir, val_dir,
                n_images=args.val_size,
                layouts=layouts,
                amplitude_factor=args.amplitude,
                width=args.render_width,
            )

        train_dataset = OnlineRenderDataset(
            train_npys, layouts, augment,
            amplitude_factor=args.amplitude,
            width=args.render_width,
            epoch_size=args.epoch_size,
        )
        val_dataset = ValDataset(val_dir, crop_size=args.crop_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train: {len(train_dataset)} samples/epoch, Val: {len(val_dataset)} samples")

    # --- Model ---
    model = UNet(
        num_in_channels=3,
        num_out_channels=4,
        depth=2,
        dims=[32, 64, 128, 256, 320, 320, 320, 320],
    ).to(device)

    # Load pretrained weights
    if args.init_weights and os.path.exists(args.init_weights):
        state = torch.load(args.init_weights, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f"Loaded weights from {args.init_weights}")

    # --- Training setup ---
    criterion = DiceFocalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda') if args.amp else None

    best_val_loss = float('inf')
    patience_counter = 0

    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        n_batches = 0
        t_epoch = time.time()

        for batch_idx, (imgs, masks) in enumerate(train_loader):
            imgs = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            if scaler:
                with torch.amp.autocast('cuda'):
                    preds = model(imgs)
                    loss = criterion(preds, masks)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                preds = model(imgs)
                loss = criterion(preds, masks)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            train_loss += loss.item()
            n_batches += 1

            if (batch_idx + 1) % 50 == 0:
                avg = train_loss / n_batches
                print(f"  Epoch {epoch} | Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {avg:.4f} | Time: {time.time()-t_epoch:.0f}s")

        scheduler.step()
        train_loss /= max(n_batches, 1)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)
                preds = model(imgs)
                loss = criterion(preds, masks)
                val_loss += loss.item()
                n_val += 1
        val_loss /= max(n_val, 1)

        elapsed = time.time() - t_epoch
        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}/{args.epochs} | Train: {train_loss:.4f} | "
              f"Val: {val_loss:.4f} | LR: {lr_now:.2e} | Time: {elapsed:.0f}s")

        # --- Checkpointing ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_weights.pt'))
            print(f"  ** New best val loss: {val_loss:.4f} — saved")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping after {args.patience} epochs without improvement")
                break

        # Save periodic checkpoint
        if epoch % 5 == 0:
            torch.save(model.state_dict(),
                       os.path.join(args.save_dir, f'checkpoint_epoch{epoch}.pt'))

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best weights: {os.path.join(args.save_dir, 'best_weights.pt')}")


def main():
    parser = argparse.ArgumentParser(description='Train U-Net on multi-layout rendered ECGs')
    parser.add_argument('--npy_dir', default='/media/data1/anolin/temp_new_dataset/ecg_npy/',
                        help='Directory with NPY ECG files')
    parser.add_argument('--save_dir', default='weights/unet_multilayout/',
                        help='Output directory for weights')
    parser.add_argument('--init_weights', default='weights/unet_phone_finetuned/best_weights.pt',
                        help='Initial U-Net weights to fine-tune from')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size (small due to online rendering)')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--crop_size', type=int, default=1024)
    parser.add_argument('--render_width', type=int, default=2500)
    parser.add_argument('--amplitude', type=float, default=4.88,
                        help='Amplitude factor (4.88=raw MHI)')
    parser.add_argument('--max_npys', type=int, default=None,
                        help='Limit number of NPY files (for testing)')
    parser.add_argument('--n_layouts', type=int, default=None,
                        help='Number of layouts to use (default: all 13)')
    parser.add_argument('--epoch_size', type=int, default=5000,
                        help='Samples per epoch (online rendering)')
    parser.add_argument('--val_size', type=int, default=200,
                        help='Number of validation image-mask pairs')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers for training')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--amp', action='store_true',
                        help='Use automatic mixed precision')

    # Pre-rendered mode
    parser.add_argument('--prerendered_dir', default=None,
                        help='Directory with pre-rendered images/ and masks/ (skips online rendering)')

    # HuggingFace parquet mode
    parser.add_argument('--hf_parquet_dir', default=None,
                        help='Directory with HuggingFace parquet shards (img+mask columns)')

    # Pre-generate mode
    parser.add_argument('--generate-only', action='store_true',
                        help='Only generate training data, do not train')
    parser.add_argument('--generate-n', type=int, default=10000,
                        help='Number of images to generate')

    args = parser.parse_args()

    if args.generate_only:
        generate_training_data(
            args.npy_dir,
            os.path.join(args.save_dir, 'train_data'),
            n_images=args.generate_n,
            amplitude_factor=args.amplitude,
            width=args.render_width,
        )
    else:
        train(args)


if __name__ == '__main__':
    main()

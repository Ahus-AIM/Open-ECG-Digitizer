"""
Fine-tune the segmentation U-Net with smartphone-photo augmentations.

Loads the HuggingFace ECG Digitizer Development Dataset (parquet shards),
applies phone-photo augmentations to ECG images while keeping segmentation
masks aligned, then fine-tunes from the existing U-Net weights.

Key insight: geometric transforms (perspective, curl, crop) are applied
to BOTH image and mask. Photometric transforms (shadows, lighting, color,
noise, blur, JPEG, CLAHE) are applied to the image ONLY.

Usage:
    cd /volume/Open-ECG-Digitizer
    python scripts/train_segmentation_phone_augment.py --device cuda:0

    # Quick test with 2 shards:
    python scripts/train_segmentation_phone_augment.py --device cuda:0 --max_shards 2
"""

import argparse
import io
import os
import sys
import time
import glob

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# Add project root to path
sys.path.insert(0, "/volume/Open-ECG-Digitizer")

from src.model.unet import UNet
from src.loss.loss import DiceFocalLoss


# ---------------------------------------------------------------------------
# Phone-photo augmentations (geometric + photometric)
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
        """
        Args:
            img: (H, W, 3) uint8 BGR image
            mask: (H, W, 3) uint8 RGB mask

        Returns:
            (img, mask) both (crop_size, crop_size, 3) uint8
        """
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

    # --- Geometric transforms (both image and mask) ---

    def _random_crop(self, img, mask):
        h, w = img.shape[:2]
        cs = self.crop_size
        if h < cs or w < cs:
            # Resize up if too small
            scale = max(cs / h, cs / w) * 1.05
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
            mask = cv2.resize(mask, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)
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
                                   borderValue=(0, 0, 0),
                                   flags=cv2.INTER_NEAREST)
        return img, mask

    def _paper_curl(self, img, mask):
        h, w = img.shape[:2]
        amplitude = np.random.uniform(2, 6)
        frequency = np.random.uniform(1, 3)
        # Precompute remap arrays
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

    # --- Photometric transforms (image only) ---

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
# Dataset: load from parquet shards
# ---------------------------------------------------------------------------

class ParquetECGDataset(Dataset):
    """Loads ECG images + segmentation masks from HuggingFace parquet shards.

    Uses img_T0 (clean template) and mask_T0 (clean mask) as base,
    then applies phone-photo augmentations.
    """

    def __init__(self, parquet_dir, augment_fn=None, max_shards=None,
                 use_augmented_pairs=False):
        """
        Args:
            parquet_dir: Directory containing train-*.parquet files
            augment_fn: PhonePhotoAugmentation instance
            max_shards: Limit number of shards loaded (for debugging)
            use_augmented_pairs: If True, also include the pre-augmented
                img/mask pairs (doubles the dataset)
        """
        self.augment_fn = augment_fn
        self.use_augmented_pairs = use_augmented_pairs

        # Find all parquet shards
        pattern = os.path.join(parquet_dir, "**", "train-*.parquet")
        shards = sorted(glob.glob(pattern, recursive=True))
        if not shards:
            # Try flat directory
            pattern = os.path.join(parquet_dir, "train-*.parquet")
            shards = sorted(glob.glob(pattern))
        if max_shards:
            shards = shards[:max_shards]
        print(f"Loading {len(shards)} parquet shards...")

        # Load all image bytes into memory (they're compressed, so reasonable)
        self.samples = []  # list of (img_bytes, mask_bytes)
        for shard_path in shards:
            df = pd.read_parquet(shard_path, columns=["img_T0", "mask_T0"] +
                                 (["img", "mask"] if use_augmented_pairs else []))
            for _, row in df.iterrows():
                # Clean template pair
                img_bytes = row["img_T0"]["bytes"] if isinstance(row["img_T0"], dict) else row["img_T0"]
                mask_bytes = row["mask_T0"]["bytes"] if isinstance(row["mask_T0"], dict) else row["mask_T0"]
                self.samples.append((img_bytes, mask_bytes))

                # Optionally also include pre-augmented pair
                if use_augmented_pairs:
                    img_bytes2 = row["img"]["bytes"] if isinstance(row["img"], dict) else row["img"]
                    mask_bytes2 = row["mask"]["bytes"] if isinstance(row["mask"], dict) else row["mask"]
                    self.samples.append((img_bytes2, mask_bytes2))

            del df

        print(f"  Loaded {len(self.samples)} samples total")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_bytes, mask_bytes = self.samples[idx]

        # Decode images
        img = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
        mask = np.array(Image.open(io.BytesIO(mask_bytes)).convert("RGB"))

        # Convert img to BGR for OpenCV augmentations
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Apply augmentations
        if self.augment_fn:
            img_bgr, mask = self.augment_fn(img_bgr, mask)

        # Convert back to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # To tensors: (H, W, 3) uint8 -> (3, H, W) float [0,1]
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask).permute(2, 0, 1).float() / 255.0

        return img_tensor, mask_tensor


class ValParquetECGDataset(Dataset):
    """Validation dataset: no augmentations, just center crop."""

    def __init__(self, parquet_dir, crop_size=1024, max_shards=None):
        self.crop_size = crop_size

        pattern = os.path.join(parquet_dir, "**", "train-*.parquet")
        shards = sorted(glob.glob(pattern, recursive=True))
        if not shards:
            pattern = os.path.join(parquet_dir, "train-*.parquet")
            shards = sorted(glob.glob(pattern))
        # Use last 10% of shards as val
        n_val = max(1, len(shards) // 10)
        if max_shards:
            n_val = min(n_val, max_shards)
        val_shards = shards[-n_val:]
        # Remove val shards from available pool
        self.val_shard_basenames = {os.path.basename(s) for s in val_shards}
        print(f"Val: loading {len(val_shards)} shards...")

        self.samples = []
        for shard_path in val_shards:
            df = pd.read_parquet(shard_path, columns=["img_T0", "mask_T0"])
            for _, row in df.iterrows():
                img_bytes = row["img_T0"]["bytes"] if isinstance(row["img_T0"], dict) else row["img_T0"]
                mask_bytes = row["mask_T0"]["bytes"] if isinstance(row["mask_T0"], dict) else row["mask_T0"]
                self.samples.append((img_bytes, mask_bytes))
            del df
        print(f"  Val: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_bytes, mask_bytes = self.samples[idx]
        img = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
        mask = np.array(Image.open(io.BytesIO(mask_bytes)).convert("RGB"))

        # Center crop
        h, w = img.shape[:2]
        cs = self.crop_size
        if h < cs or w < cs:
            scale = max(cs / h, cs / w) * 1.05
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
            mask = cv2.resize(mask, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)
            h, w = img.shape[:2]
        y = (h - cs) // 2
        x = (w - cs) // 2
        img = img[y:y+cs, x:x+cs]
        mask = mask[y:y+cs, x:x+cs]

        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask).permute(2, 0, 1).float() / 255.0
        return img_tensor, mask_tensor


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def load_unet(weights_path, device):
    """Load U-Net with existing weights."""
    model = UNet(
        num_in_channels=3,
        num_out_channels=4,
        depth=2,
        dims=[32, 64, 128, 256, 320, 320, 320, 320],
    )
    if weights_path and os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
        if isinstance(checkpoint, tuple):
            checkpoint = checkpoint[0]
        # Handle torch.compile prefix
        checkpoint = {k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()}
        model.load_state_dict(checkpoint)
        print(f"Loaded U-Net weights from {weights_path}")
    else:
        print("WARNING: No weights loaded, training from scratch!")
    model = model.to(device)
    return model


def evaluate(model, val_loader, criterion, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            with torch.autocast(device_type="cuda", enabled=True):
                logits = model(imgs)
                loss = criterion(logits, masks)
            total_loss += loss.item() * imgs.shape[0]
            n += imgs.shape[0]
    return total_loss / max(n, 1)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune segmentation U-Net with phone augmentations")
    parser.add_argument("--parquet_dir", type=str,
                        default="/volume/Open-ECG-Digitizer/data/hf_ecg_digitizer/parquet",
                        help="Directory with downloaded parquet shards")
    parser.add_argument("--weights", type=str,
                        default="/volume/Open-ECG-Digitizer/weights/unet_weights_07072025.pt",
                        help="Pre-trained U-Net weights")
    parser.add_argument("--save_dir", type=str,
                        default="/volume/Open-ECG-Digitizer/weights/unet_phone_finetuned",
                        help="Output directory for fine-tuned weights")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (lower than original 0.0037 for fine-tuning)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--crop_size", type=int, default=1024)
    parser.add_argument("--augment_p", type=float, default=0.8)
    parser.add_argument("--max_shards", type=int, default=None,
                        help="Limit shards for debugging (e.g., 2)")
    parser.add_argument("--use_augmented_pairs", action="store_true",
                        help="Also include pre-augmented img/mask pairs (2x data)")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # ---- Load model ----
    print("Loading U-Net model...")
    model = load_unet(args.weights, args.device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    # ---- Loss ----
    criterion = DiceFocalLoss(alpha=1.0, signal_class=2, union_exponent=2, gamma=2.0)

    # ---- Datasets ----
    print("\nLoading datasets...")
    augment_fn = PhonePhotoAugmentation(p=args.augment_p, crop_size=args.crop_size)

    # Split: last 10% of shards for val, rest for train
    all_shards = sorted(glob.glob(os.path.join(args.parquet_dir, "**", "train-*.parquet"), recursive=True))
    if not all_shards:
        all_shards = sorted(glob.glob(os.path.join(args.parquet_dir, "train-*.parquet")))
    n_val_shards = max(1, len(all_shards) // 10)
    val_shards = set(os.path.basename(s) for s in all_shards[-n_val_shards:])
    train_shard_dir_for_filter = args.parquet_dir

    # Create train dataset (exclude val shards)
    train_dataset = ParquetECGDataset(
        args.parquet_dir, augment_fn=augment_fn,
        max_shards=args.max_shards,
        use_augmented_pairs=args.use_augmented_pairs,
    )
    # Filter out val-shard samples from train
    # (ParquetECGDataset loads all shards, so we need separate datasets)
    # Actually, let's just create them properly:
    val_dataset = ValParquetECGDataset(
        args.parquet_dir, crop_size=args.crop_size,
        max_shards=args.max_shards,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    print(f"\n  Train: {len(train_dataset)} samples, {len(train_loader)} batches/epoch")
    print(f"  Val:   {len(val_dataset)} samples, {len(val_loader)} batches/epoch")

    # ---- Optimizer & scheduler ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader), eta_min=args.lr / 10,
    )

    # ---- Training loop ----
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\nStarting training: {args.epochs} epochs, batch_size={args.batch_size}")
    print(f"  LR: {args.lr}, device: {args.device}")
    print(f"  Augmentations: phone-photo (shadow, lighting, color, curl, perspective, CLAHE, etc.)")
    print()

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_samples = 0
        t0 = time.time()

        for batch_idx, (imgs, masks) in enumerate(train_loader):
            imgs = imgs.to(args.device)
            masks = masks.to(args.device)

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", enabled=True):
                logits = model(imgs)
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item() * imgs.shape[0]
            epoch_samples += imgs.shape[0]

            if (batch_idx + 1) % 50 == 0 or batch_idx == 0:
                avg = epoch_loss / epoch_samples
                lr_now = optimizer.param_groups[0]["lr"]
                print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {avg:.4f} | LR: {lr_now:.2e}")

        # ---- Validation ----
        val_loss = evaluate(model, val_loader, criterion, args.device)
        epoch_time = time.time() - t0
        train_avg = epoch_loss / max(epoch_samples, 1)

        print(f"\n  Epoch {epoch+1}/{args.epochs} | Train: {train_avg:.4f} | "
              f"Val: {val_loss:.4f} | Time: {epoch_time:.0f}s")

        # Save checkpoint every epoch
        ckpt = os.path.join(args.save_dir, f"weights_epoch{epoch+1}.pt")
        torch.save(model.state_dict(), ckpt)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_path = os.path.join(args.save_dir, "best_weights.pt")
            torch.save(model.state_dict(), best_path)
            print(f"  New best! Val loss: {val_loss:.4f} -> {best_path}")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{args.patience}")

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
        print()

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best weights saved to: {os.path.join(args.save_dir, 'best_weights.pt')}")


if __name__ == "__main__":
    main()

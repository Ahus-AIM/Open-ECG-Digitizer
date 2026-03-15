#!/usr/bin/env python3
"""
Train a layout + orientation classifier on rendered ECG images.

Multi-task model: shared ResNet-18 backbone with THREE heads:
  - Layout head: 13 ECG layout classes
  - Rotation head: 4 classes (0°, 90°, 180°, 270° CCW)
  - Flip head: 2 classes (no flip, horizontal flip)

Decomposing rotation and flip into separate heads helps the model
distinguish between e.g. 180° rotation vs horizontal flip.

Orientations are applied randomly on-the-fly during training (no re-rendering).
At inference, predicted rotation + flip are used to correct the image.

Usage:
    cd /volume/Open-ECG-Digitizer
    python scripts/train_layout_classifier.py --device cuda:1

    # Quick test:
    python scripts/train_layout_classifier.py --device cuda:1 --max_images 1000 --epochs 5
"""

import argparse
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models

import wandb

sys.path.insert(0, "/volume/Open-ECG-Digitizer")

# Reuse phone augmentations from U-Net training (photometric only, no mask needed)
from scripts.train_unet_multilayout import PhonePhotoAugmentation


# 13 layout classes (sorted for consistent label encoding)
LAYOUT_CLASSES = [
    'cabrera_12x1',
    'cabrera_6x1_limb',
    'precordial_3x2',
    'precordial_6x1',
    'standard_12x1',
    'standard_3x1',
    'standard_3x4',
    'standard_3x4_with_r1',
    'standard_3x4_with_r2',
    'standard_3x4_with_r3',
    'standard_6x1_limb',
    'standard_6x2',
    'standard_6x2_with_r1',
]
LAYOUT_TO_IDX = {name: i for i, name in enumerate(LAYOUT_CLASSES)}

# Rotation: 4 classes
ROT_CLASSES = ['rot0', 'rot90', 'rot180', 'rot270']

# Flip: 2 classes
FLIP_CLASSES = ['no_flip', 'hflip']

# Legacy 8-class combined (for backward compat / reference)
ORIENT_CLASSES = [
    'rot0', 'rot90', 'rot180', 'rot270',
    'rot0_hflip', 'rot90_hflip', 'rot180_hflip', 'rot270_hflip',
]


class LayoutOrientModel(nn.Module):
    """ResNet-18 with three classification heads: layout + rotation + flip."""

    def __init__(self, num_layouts=13, num_rotations=4, num_flips=2):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.layout_head = nn.Linear(512, num_layouts)
        self.rot_head = nn.Linear(512, num_rotations)
        self.flip_head = nn.Linear(512, num_flips)

    def forward(self, x):
        feat = self.features(x).flatten(1)
        return self.layout_head(feat), self.rot_head(feat), self.flip_head(feat)


def apply_orientation(img, rot_label, flip_label):
    """Apply orientation transform to image. Returns contiguous array.

    Args:
        img: (H, W, 3) numpy array
        rot_label: 0=0°, 1=90°CCW, 2=180°, 3=270°CCW
        flip_label: 0=no flip, 1=horizontal flip
    """
    if rot_label > 0:
        img = np.rot90(img, k=rot_label)  # CCW rotation
    if flip_label:
        img = np.fliplr(img)
    return np.ascontiguousarray(img)


def undo_orientation(img, rot_label, flip_label):
    """Undo orientation transform to restore original image.

    Args:
        img: (H, W, 3) numpy array with orientation applied
        rot_label: 0=0°, 1=90°CCW, 2=180°, 3=270°CCW
        flip_label: 0=no flip, 1=horizontal flip
    """
    # Undo in reverse order: flip first (self-inverse), then reverse rotation
    if flip_label:
        img = np.fliplr(img)
    if rot_label > 0:
        img = np.rot90(img, k=(4 - rot_label))  # reverse rotation
    return np.ascontiguousarray(img)


def resize_pad(img, target_size):
    """Resize image to fit within target_size, zero-pad to square.

    Preserves aspect ratio. Padding is added symmetrically.
    """
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    img = cv2.resize(img, (new_w, new_h))

    # Zero-pad to target_size x target_size
    pad_h = target_size - new_h
    pad_w = target_size - new_w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return img


class LayoutDataset(Dataset):
    """ECG images with layout labels, orientation augmentation, and phone augmentations.

    Uses resize+pad (not crop) so the model sees the full ECG layout.
    """

    def __init__(self, image_paths: list, labels: list, augment: bool = True,
                 crop_size: int = 512, apply_orient: bool = True):
        self.image_paths = image_paths
        self.labels = labels
        self.augment = augment
        self.crop_size = crop_size
        self.apply_orient = apply_orient
        self._aug_p = 0.8 if augment else 0.0

    def __len__(self):
        return len(self.image_paths)

    def _photometric_augment(self, img):
        """Apply photometric augmentations (no geometric/crop)."""
        if np.random.random() > self._aug_p:
            return img
        if np.random.random() < 0.7:
            alpha = np.random.uniform(0.7, 1.3)
            beta = np.random.randint(-30, 30)
            img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        if np.random.random() < 0.5:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] *= np.random.uniform(0.7, 1.3)
            hsv[:, :, 0] += np.random.randint(-10, 10)
            hsv = np.clip(hsv, 0, 255).astype(np.uint8)
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        if np.random.random() < 0.4:
            noise = np.random.normal(0, np.random.uniform(3, 15), img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        if np.random.random() < 0.5:
            quality = np.random.randint(30, 85)
            _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if np.random.random() < 0.3:
            k = np.random.choice([3, 5])
            img = cv2.GaussianBlur(img, (k, k), 0)
        return img

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        if img is None:
            img = np.ones((self.crop_size, self.crop_size, 3), dtype=np.uint8) * 255

        label = self.labels[idx]

        # Apply random orientation (rotation + optional flip, independently)
        if self.apply_orient:
            rot_label = random.randint(0, 3)
            flip_label = random.randint(0, 1)
            img = apply_orientation(img, rot_label, flip_label)
        else:
            rot_label = 0
            flip_label = 0

        # Resize to fit + zero-pad to square FIRST (full image, no cropping)
        img = resize_pad(img, self.crop_size)

        # Then apply photometric augmentations on the smaller image
        img = self._photometric_augment(img)

        # Convert to tensor
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img_t, label, rot_label, flip_label


def train(args):
    import pandas as pd

    device = torch.device(args.device)

    # Load manifest
    df = pd.read_csv(args.manifest)
    print(f"Loaded {len(df)} images, {df['layout'].nunique()} layouts")

    # Filter to valid layouts
    df = df[df['layout'].isin(LAYOUT_CLASSES)].copy()
    df['label'] = df['layout'].map(LAYOUT_TO_IDX)

    if args.max_images and args.max_images < len(df):
        df = df.sample(args.max_images, random_state=42)

    # Check image accessibility
    exists = df['png_path'].head(100).apply(os.path.exists)
    print(f"Image accessibility (100 sampled): {exists.sum()}/100")

    # Split: 80% train, 10% val, 10% test
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42,
                                          stratify=df['label'])
    val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42,
                                        stratify=test_df['label'])

    print(f"Split: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
    print(f"Class distribution (train):")
    for layout, count in train_df['layout'].value_counts().head(5).items():
        print(f"  {layout}: {count}")

    # Datasets (orientation augmentation applied on-the-fly)
    train_dataset = LayoutDataset(
        train_df['png_path'].tolist(), train_df['label'].tolist(),
        augment=True, crop_size=args.crop_size, apply_orient=True,
    )
    val_dataset = LayoutDataset(
        val_df['png_path'].tolist(), val_df['label'].tolist(),
        augment=False, crop_size=args.crop_size, apply_orient=True,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)

    # Model: ResNet18 with three heads (layout + rotation + flip)
    model = LayoutOrientModel(
        num_layouts=len(LAYOUT_CLASSES),
        num_rotations=len(ROT_CLASSES),
        num_flips=len(FLIP_CLASSES),
    ).to(device)

    # Class weights for layout (inverse frequency)
    class_counts = train_df['label'].value_counts().sort_index().values.astype(float)
    class_weights = (1.0 / class_counts) * class_counts.sum() / len(LAYOUT_CLASSES)
    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"Layout class weights: {class_weights.cpu().numpy().round(2)}")

    layout_criterion = nn.CrossEntropyLoss(weight=class_weights)
    rot_criterion = nn.CrossEntropyLoss()  # balanced (uniform random rotations)
    flip_criterion = nn.CrossEntropyLoss()  # balanced (uniform random flips)
    orient_loss_weight = args.orient_loss_weight

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda')

    best_val_acc = 0.0
    patience_counter = 0
    os.makedirs(args.save_dir, exist_ok=True)

    # Initialize wandb
    wandb.init(
        project="ecg-layout-classifier",
        name=f"3head-{len(df)//1000}k-{args.epochs}ep",
        config={
            "model": "resnet18-3head",
            "dataset_size": len(df),
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df),
            "n_layout_classes": len(LAYOUT_CLASSES),
            "n_rot_classes": len(ROT_CLASSES),
            "n_flip_classes": len(FLIP_CLASSES),
            "orient_loss_weight": orient_loss_weight,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "epochs": args.epochs,
            "crop_size": args.crop_size,
            "patience": args.patience,
            "manifest": args.manifest,
        },
    )

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        train_layout_correct = 0
        train_rot_correct = 0
        train_flip_correct = 0
        train_total = 0
        t0 = time.time()

        for batch_idx, (imgs, layout_labels, rot_labels, flip_labels) in enumerate(train_loader):
            imgs = imgs.to(device)
            layout_labels = layout_labels.to(device)
            rot_labels = rot_labels.to(device)
            flip_labels = flip_labels.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                layout_logits, rot_logits, flip_logits = model(imgs)
                loss_layout = layout_criterion(layout_logits, layout_labels)
                loss_rot = rot_criterion(rot_logits, rot_labels)
                loss_flip = flip_criterion(flip_logits, flip_labels)
                loss = loss_layout + orient_loss_weight * (loss_rot + loss_flip)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_layout_correct += (layout_logits.argmax(1) == layout_labels).sum().item()
            train_rot_correct += (rot_logits.argmax(1) == rot_labels).sum().item()
            train_flip_correct += (flip_logits.argmax(1) == flip_labels).sum().item()
            train_total += layout_labels.size(0)

            if (batch_idx + 1) % 100 == 0:
                avg_loss = train_loss / (batch_idx + 1)
                lay_acc = 100 * train_layout_correct / train_total
                rot_acc = 100 * train_rot_correct / train_total
                flip_acc = 100 * train_flip_correct / train_total
                print(f"  Epoch {epoch} | Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {avg_loss:.4f} | L:{lay_acc:.1f}% R:{rot_acc:.1f}% F:{flip_acc:.1f}%")

        scheduler.step()
        train_loss /= len(train_loader)
        train_layout_acc = 100 * train_layout_correct / train_total
        train_rot_acc = 100 * train_rot_correct / train_total
        train_flip_acc = 100 * train_flip_correct / train_total

        # Validate
        model.eval()
        val_loss = 0.0
        val_layout_correct = 0
        val_rot_correct = 0
        val_flip_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, layout_labels, rot_labels, flip_labels in val_loader:
                imgs = imgs.to(device)
                layout_labels = layout_labels.to(device)
                rot_labels = rot_labels.to(device)
                flip_labels = flip_labels.to(device)
                with torch.amp.autocast('cuda'):
                    layout_logits, rot_logits, flip_logits = model(imgs)
                    loss_layout = layout_criterion(layout_logits, layout_labels)
                    loss_rot = rot_criterion(rot_logits, rot_labels)
                    loss_flip = flip_criterion(flip_logits, flip_labels)
                    loss = loss_layout + orient_loss_weight * (loss_rot + loss_flip)
                val_loss += loss.item()
                val_layout_correct += (layout_logits.argmax(1) == layout_labels).sum().item()
                val_rot_correct += (rot_logits.argmax(1) == rot_labels).sum().item()
                val_flip_correct += (flip_logits.argmax(1) == flip_labels).sum().item()
                val_total += layout_labels.size(0)

        val_loss /= len(val_loader)
        val_layout_acc = 100 * val_layout_correct / val_total
        val_rot_acc = 100 * val_rot_correct / val_total
        val_flip_acc = 100 * val_flip_correct / val_total
        elapsed = time.time() - t0

        print(f"Epoch {epoch}/{args.epochs} | "
              f"Train: {train_loss:.4f} (L:{train_layout_acc:.1f}% R:{train_rot_acc:.1f}% F:{train_flip_acc:.1f}%) | "
              f"Val: {val_loss:.4f} (L:{val_layout_acc:.1f}% R:{val_rot_acc:.1f}% F:{val_flip_acc:.1f}%) | "
              f"Time: {elapsed:.0f}s")

        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "train/layout_acc": train_layout_acc,
            "train/rot_acc": train_rot_acc,
            "train/flip_acc": train_flip_acc,
            "val/loss": val_loss,
            "val/layout_acc": val_layout_acc,
            "val/rot_acc": val_rot_acc,
            "val/flip_acc": val_flip_acc,
            "lr": scheduler.get_last_lr()[0],
        })

        # Checkpoint on combined accuracy (mean of layout + rot + flip)
        val_combined = (val_layout_acc + val_rot_acc + val_flip_acc) / 3
        if val_combined > best_val_acc:
            best_val_acc = val_combined
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_layout_classifier.pt'))
            print(f"  ** New best combined val acc: {val_combined:.1f}% "
                  f"(L:{val_layout_acc:.1f}% R:{val_rot_acc:.1f}% F:{val_flip_acc:.1f}%) — saved")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping after {args.patience} epochs without improvement")
                break

    print(f"\nTraining complete. Best combined val accuracy: {best_val_acc:.1f}%")

    # Final test evaluation
    print("\n=== Test Set Evaluation ===")
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_layout_classifier.pt'),
                                      map_location=device, weights_only=True))
    model.eval()

    test_dataset = LayoutDataset(
        test_df['png_path'].tolist(), test_df['label'].tolist(),
        augment=False, crop_size=args.crop_size, apply_orient=True,
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=2)

    all_layout_preds, all_layout_labels = [], []
    all_rot_preds, all_rot_labels = [], []
    all_flip_preds, all_flip_labels = [], []
    with torch.no_grad():
        for imgs, layout_labels, rot_labels, flip_labels in test_loader:
            imgs = imgs.to(device)
            with torch.amp.autocast('cuda'):
                layout_logits, rot_logits, flip_logits = model(imgs)
            all_layout_preds.extend(layout_logits.argmax(1).cpu().tolist())
            all_layout_labels.extend(layout_labels.tolist())
            all_rot_preds.extend(rot_logits.argmax(1).cpu().tolist())
            all_rot_labels.extend(rot_labels.tolist())
            all_flip_preds.extend(flip_logits.argmax(1).cpu().tolist())
            all_flip_labels.extend(flip_labels.tolist())

    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

    # Layout results
    layout_acc = accuracy_score(all_layout_labels, all_layout_preds) * 100
    layout_report = classification_report(all_layout_labels, all_layout_preds,
                                          target_names=LAYOUT_CLASSES, digits=3)
    print(f"\n--- Layout Classification (Test) ---")
    print(f"Test Accuracy: {layout_acc:.1f}%")
    print(layout_report)

    # Rotation results
    rot_acc = accuracy_score(all_rot_labels, all_rot_preds) * 100
    rot_report = classification_report(all_rot_labels, all_rot_preds,
                                       target_names=ROT_CLASSES, digits=3)
    print(f"\n--- Rotation Classification (Test) ---")
    print(f"Test Accuracy: {rot_acc:.1f}%")
    print(rot_report)
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(all_rot_labels, all_rot_preds))

    # Flip results
    flip_acc = accuracy_score(all_flip_labels, all_flip_preds) * 100
    flip_report = classification_report(all_flip_labels, all_flip_preds,
                                        target_names=FLIP_CLASSES, digits=3)
    print(f"\n--- Flip Classification (Test) ---")
    print(f"Test Accuracy: {flip_acc:.1f}%")
    print(flip_report)
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(all_flip_labels, all_flip_preds))

    wandb.log({
        "test/layout_acc": layout_acc,
        "test/rot_acc": rot_acc,
        "test/flip_acc": flip_acc,
        "best_val_combined_acc": best_val_acc,
    })
    wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='Train ECG layout classifier')
    parser.add_argument('--manifest', default='data/acs_multilayout/manifest.csv')
    parser.add_argument('--save_dir', default='weights/layout_classifier/')
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--max_images', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--orient_loss_weight', type=float, default=1.0,
                        help='Weight for orientation losses relative to layout loss')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()

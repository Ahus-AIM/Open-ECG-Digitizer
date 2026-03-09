#!/usr/bin/env python3
"""
Train a layout classifier on rendered ECG images with smartphone augmentations.

Uses the ACS multilayout manifest (57K images, 13 layout classes) and applies
phone-photo augmentations (perspective, shadows, blur, etc.) during training
so the classifier works on real smartphone photos.

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


class LayoutDataset(Dataset):
    """ECG images with layout labels and phone augmentations."""

    def __init__(self, image_paths: list, labels: list, augment: bool = True,
                 crop_size: int = 512):
        self.image_paths = image_paths
        self.labels = labels
        self.augment = augment
        self.crop_size = crop_size
        # Phone augmentations for image only (no mask needed)
        self._phone_aug = PhonePhotoAugmentation(p=0.8 if augment else 0.0,
                                                  crop_size=crop_size)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        if img is None:
            img = np.ones((self.crop_size, self.crop_size, 3), dtype=np.uint8) * 255

        label = self.labels[idx]

        # Resize to manageable size first (much faster than augmenting full-res)
        target_h = self.crop_size + 100  # Small margin for crop
        h, w = img.shape[:2]
        if h > target_h * 1.5:
            scale = target_h / h
            img = cv2.resize(img, (int(w * scale), target_h))

        # Apply phone augmentations (using a dummy mask)
        dummy_mask = np.zeros_like(img)
        img, _ = self._phone_aug(img, dummy_mask)

        # Convert to tensor
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img_t, label


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

    # Datasets
    train_dataset = LayoutDataset(
        train_df['png_path'].tolist(), train_df['label'].tolist(),
        augment=True, crop_size=args.crop_size,
    )
    val_dataset = LayoutDataset(
        val_df['png_path'].tolist(), val_df['label'].tolist(),
        augment=False, crop_size=args.crop_size,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)

    # Model: ResNet18 with 13-class head
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(LAYOUT_CLASSES))
    model = model.to(device)

    # Class weights (inverse frequency)
    class_counts = train_df['label'].value_counts().sort_index().values.astype(float)
    class_weights = (1.0 / class_counts) * class_counts.sum() / len(LAYOUT_CLASSES)
    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"Class weights: {class_weights.cpu().numpy().round(2)}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda')

    best_val_acc = 0.0
    patience_counter = 0
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        t0 = time.time()

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                logits = model(imgs)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            if (batch_idx + 1) % 100 == 0:
                avg_loss = train_loss / (batch_idx + 1)
                acc = 100 * train_correct / train_total
                print(f"  Epoch {epoch} | Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {avg_loss:.4f} | Acc: {acc:.1f}%")

        scheduler.step()
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                with torch.amp.autocast('cuda'):
                    logits = model(imgs)
                    loss = criterion(logits, labels)
                val_loss += loss.item()
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        elapsed = time.time() - t0

        print(f"Epoch {epoch}/{args.epochs} | Train: {train_loss:.4f} ({train_acc:.1f}%) | "
              f"Val: {val_loss:.4f} ({val_acc:.1f}%) | Time: {elapsed:.0f}s")

        # Checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_layout_classifier.pt'))
            print(f"  ** New best val acc: {val_acc:.1f}% — saved")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping after {args.patience} epochs without improvement")
                break

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.1f}%")

    # Final test evaluation
    print("\n=== Test Set Evaluation ===")
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_layout_classifier.pt'),
                                      map_location=device))
    model.eval()

    test_dataset = LayoutDataset(
        test_df['png_path'].tolist(), test_df['label'].tolist(),
        augment=False, crop_size=args.crop_size,
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=2)

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            with torch.amp.autocast('cuda'):
                logits = model(imgs)
            preds = logits.argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    from sklearn.metrics import classification_report, accuracy_score
    print(f"Test Accuracy: {accuracy_score(all_labels, all_preds)*100:.1f}%")
    print(classification_report(all_labels, all_preds,
                                target_names=LAYOUT_CLASSES, digits=3))


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
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()

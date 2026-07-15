import os
import random
import string
from typing import Any, Optional

import cv2
import numpy as np
import torch
from numpy import ndarray
from numpy.fft import fft2, fftshift, ifft2
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


def pink_noise(shape: tuple[int, int]) -> ndarray[Any, Any]:
    """
    Generates pink noise using the 1/f noise method.
    Pink noise has spatial correlation, which is useful for simulating natural textures.
    """
    rows, cols = shape
    x = np.linspace(-0.5, 0.5, cols)
    y = np.linspace(-0.5, 0.5, rows)
    X, Y = np.meshgrid(x, y)
    f = np.sqrt(X**2 + Y**2)
    f[rows // 2, cols // 2] = f[rows // 2, cols // 2 + 1]
    spectrum = 1 / (f + 1e-3)
    noise = np.random.normal(size=shape)
    f_noise = fft2(noise)
    pink = np.real(ifft2(f_noise * fftshift(spectrum)))
    pink = (pink - pink.min()) / (pink.max() - pink.min())
    return np.asarray(pink, dtype=np.float32)


def random_lines(img: ndarray[Any, Any], n_lines: int, image_size: int) -> ndarray[Any, Any]:
    for _ in range(n_lines):
        x1, y1 = random.randint(0, image_size), random.randint(0, image_size)
        x2, y2 = random.randint(0, image_size), random.randint(0, image_size)
        color = (1.0,)  # OpenCV expects a tuple, even for grayscale
        thickness = random.randint(1, 3)
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    return img


def intensity_gradient(img: ndarray[Any, Any], image_size: int) -> ndarray[Any, Any]:
    grad = np.tile(np.linspace(0, 1, image_size), (image_size, 1))
    direction = random.choice(["horizontal", "vertical"])
    if direction == "vertical":
        grad = grad.T
    return img * grad.astype(np.float32)


def bright_splatter(img: ndarray[Any, Any], image_size: int) -> ndarray[Any, Any]:
    n_splats = random.randint(1, 5)
    for _ in range(n_splats):
        x = random.randint(0, image_size - 1)
        y = random.randint(0, image_size - 1)
        radius = random.randint(5, 40)
        intensity = random.uniform(0.1, 1.0)
        cv2.circle(img, (x, y), radius, (intensity,), -1)
    return img


class SyntheticLeadTextDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(
        self, num_samples: int, image_size: int = 256, transform: Any = None, font_dir: str = "/usr/share/fonts"
    ) -> None:
        self.num_samples: int = num_samples
        self.image_size: int = image_size
        self.transform = ToTensor() if not transform else transform
        self.font_dir: str = font_dir
        self.labels: list[str] = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        self.excluded_substrings: list[str] = self.labels
        self.fonts: list[str] = self._load_fonts()

    def _load_fonts(self) -> list[str]:
        font_files: list[str] = []
        for root, _, files in os.walk(self.font_dir):
            for file in files:
                if file.lower().endswith(".ttf"):
                    font_files.append(os.path.join(root, file))
        return font_files

    def _smudge_text(self, img: ndarray[Any, Any]) -> ndarray[Any, Any]:
        if random.random() < 0.7:
            sigma = random.uniform(0.0, 1.0)
            img = gaussian_filter(img, sigma=sigma)
        if random.random() < 0.5:
            dropout = random.uniform(0.1, 0.8)
            mask = np.random.rand(*img.shape) < dropout
            img[mask] = 0.0
        return img

    def _generate_image_with_text(
        self, text: str, font_path: Optional[str] = None
    ) -> tuple[ndarray[Any, Any], ndarray[Any, Any]]:
        font_size = random.randint(self.image_size // 30, self.image_size // 8)
        try:
            font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()

        img = Image.new("L", (self.image_size, self.image_size), color=0)
        label = Image.new("L", (self.image_size, self.image_size), color=0)

        draw_img = ImageDraw.Draw(img)
        draw_label = ImageDraw.Draw(label)

        bbox = draw_img.textbbox((0, 0), text, font=font)
        text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]

        x_max = int(max(0, self.image_size - text_width))
        y_max = int(max(0, self.image_size - text_height))

        try:
            x = int(random.randint(0, x_max))
            y = int(random.randint(0, y_max))
        except ValueError:
            x = int((self.image_size - text_width) // 2)
            y = int((self.image_size - text_height) // 2)

        draw_img.text((x, y), text, font=font, fill=255)
        draw_label.text((x, y), text, font=font, fill=255)

        angle = random.uniform(-180, 180)
        img = img.rotate(angle)
        label = label.rotate(angle)

        img_np = np.array(img, dtype=np.float32) / 255.0
        img_np = self._smudge_text(img_np)
        label_np = (np.array(label) > 0).astype(np.float32)

        return img_np, label_np

    def _apply_transforms(self, img: ndarray[Any, Any]) -> ndarray[Any, Any]:
        if random.random() < 0.4:
            img = random_lines(img, n_lines=random.randint(1, 3), image_size=self.image_size)

        if random.random() < 0.1:
            img = intensity_gradient(img, self.image_size)

        pink = pink_noise((self.image_size, self.image_size))
        intensity = random.uniform(1e-3, 1.0) ** 2
        img += intensity * pink
        img = np.clip(img, 0, 1)

        if random.random() < 0.1:
            img = gaussian_filter(img, sigma=random.uniform(0.0, 1.0))

        rand_multiply = np.sqrt(random.uniform(0.4, 1.0))
        img = np.clip(img * rand_multiply, 0, 1)

        if random.random() < 0.1:
            img = bright_splatter(img, self.image_size)

        return img

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        img = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        label_mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)

        font_path = random.choice(self.fonts) if self.fonts else None
        lead = random.choice(self.labels)
        lead_idx = self.labels.index(lead)

        text_img, text_mask = self._generate_image_with_text(lead, font_path)
        img = np.maximum(img, text_img)
        label_mask = np.maximum(label_mask, text_mask)

        if random.random() < 0.5:
            while True:
                rand_text = "".join(random.choices(string.ascii_uppercase + string.digits, k=random.randint(1, 4)))
                if not any(sub in rand_text for sub in self.excluded_substrings):
                    break
            rand_img, _ = self._generate_image_with_text(rand_text, font_path)
            img = np.maximum(img, rand_img)

        img = self._apply_transforms(img)
        np.clip(img, 0.0, 1.0, out=img)

        img_tensor = self.transform(img)
        label_tensor = torch.zeros(13, self.image_size, self.image_size, dtype=torch.float32)
        label_tensor[lead_idx] = torch.from_numpy(label_mask)
        label_tensor[12] = (label_tensor[:12].sum(dim=0) == 0).float()

        return img_tensor.float(), label_tensor.argmax(dim=0).long()

    def __len__(self) -> int:
        return self.num_samples

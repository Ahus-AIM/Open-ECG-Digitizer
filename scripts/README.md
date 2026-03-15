# Scripts

## Multi-Layout ECG Renderer (`multi_layout_renderer.py`)

Renders NPY ECG signals into PNG images using matplotlib. Supports all 13 standard ECG layouts matching `src/config/lead_layouts_all.yml`.

### Supported Layouts

| Layout | Grid | Leads | Rhythm |
|--------|------|-------|--------|
| `standard_3x4` | 3 rows x 4 cols | I/aVR/V1/V4, II/aVL/V2/V5, III/aVF/V3/V6 | — |
| `standard_3x4_with_r1` | 3+1 rows | Same as 3x4 | II |
| `standard_3x4_with_r2` | 3+2 rows | Same as 3x4 | II, V5 |
| `standard_3x4_with_r3` | 3+3 rows | Same as 3x4 | II, V1, V5 |
| `standard_6x2` | 6 rows x 2 cols | I/V1, II/V2, III/V3, aVR/V4, aVL/V5, aVF/V6 | — |
| `standard_6x2_with_r1` | 6+1 rows | Same as 6x2 | II |
| `standard_12x1` | 12 rows x 1 col | I..V6 standard order | — |
| `cabrera_12x1` | 12 rows x 1 col | aVL, I, -aVR, II, aVF, III, V1..V6 | — |
| `precordial_6x1` | 6 rows x 1 col | V1..V6 | — |
| `standard_6x1_limb` | 6 rows x 1 col | I, II, III, aVR, aVL, aVF | — |
| `cabrera_6x1_limb` | 6 rows x 1 col | aVL, I, -aVR, II, aVF, III | — |
| `standard_3x1` | 3 rows x 1 col | I, II, III | — |
| `precordial_3x2` | 3 rows x 2 cols | V1/V4, V2/V5, V3/V6 | — |

### Usage

```bash
# Render single NPY in default layout (standard_3x4_with_r1)
python scripts/multi_layout_renderer.py /path/to/ecg.npy -o /tmp/output/

# Render single NPY in ALL 13 layouts
python scripts/multi_layout_renderer.py /path/to/ecg.npy --all-layouts -o /tmp/output/

# Render specific layouts
python scripts/multi_layout_renderer.py /path/to/ecg.npy \
    --layouts standard_3x4_with_r1 standard_6x2 cabrera_12x1 -o /tmp/output/

# Batch render folder of NPYs (all layouts, 500 per layout, randomized style)
python scripts/multi_layout_renderer.py /path/to/npy_folder/ \
    --all-layouts --random-style --n-per-layout 500 -o /path/to/output/

# With masks for segmentation training
python scripts/multi_layout_renderer.py /path/to/ecg.npy \
    --layout standard_3x4_with_r1 --mask -o /tmp/output/
```

### NPY Format

Input NPY files should have shape `(2500, 12)`, `(2500, 12, 1)`, or `(12, 2500)`. Columns are in standard 12-lead order: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6.

### Random Style Mode

`--random-style` randomizes per-image:
- Grid color (red, pink, orange, green)
- Grid alpha (0.1–0.8)
- Line width (1.5–4.0)
- Line color (black or dark gray)
- Label font size (16–32)
- Background (white or slight cream/gray)

### Python API

```python
from scripts.multi_layout_renderer import load_ecg_npy, render_ecg, render_ecg_random_style, LAYOUTS

lead_dict = load_ecg_npy("/path/to/ecg.npy")

# Deterministic render
img = render_ecg(lead_dict, "standard_3x4_with_r1", amplitude_factor=4.88, width=2500)
img.save("output.png")

# Randomized style (for training data diversity)
img = render_ecg_random_style(lead_dict, "standard_3x4_with_r1", amplitude_factor=4.88, width=2500)
img.save("output_random.png")

# With segmentation mask
img = render_ecg(lead_dict, "standard_3x4_with_r1", amplitude_factor=4.88, width=2500,
                 save_path="output.png", mask_path="mask.png")
```

## Balanced Layout Renderer (`render_balanced_layouts.py`)

Renders balanced training data — equal samples per layout class using multiprocessing.

```bash
python scripts/render_balanced_layouts.py --per_layout 10000 --workers 12
```

## Layout Classifier Training (`train_layout_classifier.py`)

Trains a 3-head ResNet-18 classifier: layout (13 classes) + rotation (4) + flip (2).

```bash
python scripts/train_layout_classifier.py \
    --manifest data/acs_multilayout_v3/manifest.csv \
    --save_dir weights/layout_classifier_v5/ \
    --batch_size 32 --epochs 30 --crop_size 512
```

### Model Architecture

```
ResNet-18 backbone → shared features (512-d)
  ├─ layout_head  → 13 classes (layout type)
  ├─ rot_head     → 4 classes  (rot0, rot90, rot180, rot270)
  └─ flip_head    → 2 classes  (no_flip, hflip)
```

On-the-fly orientation augmentation: each image is randomly rotated and/or flipped during training, so no re-rendering is needed.

### Inference

```python
from scripts.train_layout_classifier import LayoutOrientModel, resize_pad, undo_orientation

model = LayoutOrientModel(num_layouts=13, num_rotations=4, num_flips=2)
model.load_state_dict(torch.load("weights/layout_classifier_v5/best_layout_classifier.pt"))
model.eval()

# Preprocess: resize+pad to 512x512, ImageNet normalize
img = resize_pad(cv2.imread("photo.jpg"), 512)
tensor = torchvision.transforms.functional.to_tensor(img)
tensor = torchvision.transforms.functional.normalize(tensor, [0.485,0.456,0.406], [0.229,0.224,0.225])

layout_logits, rot_logits, flip_logits = model(tensor.unsqueeze(0))
layout_idx = layout_logits.argmax(1).item()
rot_idx = rot_logits.argmax(1).item()
flip_idx = flip_logits.argmax(1).item()

# Undo orientation on original image
corrected = undo_orientation(original_img, rot_idx, flip_idx)
```

## ONNX Export (`export_all_onnx.py`)

Exports trained models to ONNX format.

```bash
# Export all models
python scripts/export_all_onnx.py

# Export specific model
python scripts/export_all_onnx.py --models layout \
    --layout_weights weights/layout_classifier_v5/best_layout_classifier.pt
```

## Auto Render and Train (`auto_render_and_train.sh`)

Waits for rendering to finish, then automatically starts layout classifier training.

```bash
bash scripts/auto_render_and_train.sh
```

"""Smoke tests for the DeepECG-backed multi-layout renderer."""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.multi_layout_renderer import (
    LAYOUTS,
    load_ecg_npy,
    render_ecg,
    render_ecg_with_mask,
)


def test_load_ecg_npy_and_render_with_mask(tmp_path):
    npy_path = tmp_path / "ecg.npy"
    np.save(npy_path, np.zeros((2500, 12)))

    lead_dict = load_ecg_npy(str(npy_path))
    assert set(["I", "II", "V6", "-aVR"]).issubset(lead_dict)

    img = render_ecg(lead_dict, "standard_6x2", width=220)
    rendered, mask = render_ecg_with_mask(lead_dict, "standard_6x2", width=220)

    assert isinstance(img, Image.Image)
    assert isinstance(rendered, Image.Image)
    assert isinstance(mask, Image.Image)
    assert img.width == 220
    assert rendered.size == mask.size


def test_renderer_uses_full_deepecg_layout_catalogue():
    assert "standard_3x4_with_r3" in LAYOUTS
    assert "precordial_3x2" in LAYOUTS
    assert len(LAYOUTS) == 13

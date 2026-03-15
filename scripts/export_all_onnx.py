#!/usr/bin/env python3
"""
Export all trained models to ONNX (and optionally CoreML).

Usage:
    cd /volume/Open-ECG-Digitizer
    python scripts/export_all_onnx.py                      # Export all
    python scripts/export_all_onnx.py --models unet layout  # Export specific models
    python scripts/export_all_onnx.py --coreml              # Also export CoreML
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/volume/Open-ECG-Digitizer")

EXPORT_DIR = "weights/exported"


def export_unet(weights_path: str, out_name: str = "unet_multilayout"):
    """Export segmentation U-Net to ONNX."""
    from src.model.unet import UNet

    print(f"\n{'='*60}")
    print(f"Exporting U-Net: {weights_path}")
    print(f"{'='*60}")

    model = UNet(
        num_in_channels=3,
        num_out_channels=4,
        depth=2,
        dims=[32, 64, 128, 256, 320, 320, 320, 320],
    )

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    dummy = torch.randn(1, 3, 1024, 1024)
    onnx_path = os.path.join(EXPORT_DIR, f"{out_name}.onnx")

    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["image"],
        output_names=["segmentation"],
        dynamic_axes={
            "image": {0: "batch_size", 2: "height", 3: "width"},
            "segmentation": {0: "batch_size", 2: "height", 3: "width"},
        },
        opset_version=14,
        do_constant_folding=True,
    )
    size_mb = os.path.getsize(onnx_path) / 1e6
    print(f"  Saved: {onnx_path} ({size_mb:.1f} MB)")

    # Verify
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path)
    out = sess.run(None, {"image": dummy.numpy()})[0]
    with torch.no_grad():
        ref = model(dummy).numpy()
    diff = np.abs(out - ref).max()
    print(f"  Verification: max diff = {diff:.6f} {'OK' if diff < 1e-4 else 'WARNING'}")

    return onnx_path


def export_layout_classifier(weights_path: str, out_name: str = "layout_classifier"):
    """Export layout+orientation classifier (dual-head ResNet-18) to ONNX."""
    from scripts.train_layout_classifier import LayoutOrientModel

    print(f"\n{'='*60}")
    print(f"Exporting Layout+Orientation Classifier: {weights_path}")
    print(f"{'='*60}")

    model = LayoutOrientModel(num_layouts=13, num_rotations=4, num_flips=2)

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    dummy = torch.randn(1, 3, 512, 512)
    onnx_path = os.path.join(EXPORT_DIR, f"{out_name}.onnx")

    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["image"],
        output_names=["layout_logits", "rot_logits", "flip_logits"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "layout_logits": {0: "batch_size"},
            "rot_logits": {0: "batch_size"},
            "flip_logits": {0: "batch_size"},
        },
        opset_version=14,
        do_constant_folding=True,
    )
    size_mb = os.path.getsize(onnx_path) / 1e6
    print(f"  Saved: {onnx_path} ({size_mb:.1f} MB)")

    # Verify
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path)
    onnx_out = sess.run(None, {"image": dummy.numpy()})
    with torch.no_grad():
        ref_layout, ref_rot, ref_flip = model(dummy)
    diff_layout = np.abs(onnx_out[0] - ref_layout.numpy()).max()
    diff_rot = np.abs(onnx_out[1] - ref_rot.numpy()).max()
    diff_flip = np.abs(onnx_out[2] - ref_flip.numpy()).max()
    print(f"  Verification: layout max diff = {diff_layout:.6f} {'OK' if diff_layout < 1e-4 else 'WARNING'}")
    print(f"  Verification: rot max diff = {diff_rot:.6f} {'OK' if diff_rot < 1e-4 else 'WARNING'}")
    print(f"  Verification: flip max diff = {diff_flip:.6f} {'OK' if diff_flip < 1e-4 else 'WARNING'}")

    return onnx_path


def export_lead_id_unet(weights_path: str, out_name: str = "lead_identifier_unet"):
    """Export lead identifier U-Net to ONNX."""
    from src.model.unet import UNet

    print(f"\n{'='*60}")
    print(f"Exporting Lead ID U-Net: {weights_path}")
    print(f"{'='*60}")

    model = UNet(
        num_in_channels=1,
        num_out_channels=13,
        depth=2,
        dims=[32, 64, 128, 256, 320, 320, 320, 320],
    )

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    dummy = torch.randn(1, 1, 1024, 1024)
    onnx_path = os.path.join(EXPORT_DIR, f"{out_name}.onnx")

    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["text_mask"],
        output_names=["lead_logits"],
        dynamic_axes={
            "text_mask": {0: "batch_size", 2: "height", 3: "width"},
            "lead_logits": {0: "batch_size", 2: "height", 3: "width"},
        },
        opset_version=14,
        do_constant_folding=True,
    )
    size_mb = os.path.getsize(onnx_path) / 1e6
    print(f"  Saved: {onnx_path} ({size_mb:.1f} MB)")

    # Verify
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path)
    out = sess.run(None, {"text_mask": dummy.numpy()})[0]
    with torch.no_grad():
        ref = model(dummy).numpy()
    diff = np.abs(out - ref).max()
    print(f"  Verification: max diff = {diff:.6f} {'OK' if diff < 1e-4 else 'WARNING'}")

    return onnx_path


def export_wcr(weights_path: str, out_name: str = "acs_predictor"):
    """Export WCR ACS predictor to ONNX."""
    print(f"\n{'='*60}")
    print(f"Exporting WCR: {weights_path}")
    print(f"{'='*60}")

    sys.path.insert(0, "/volume/DeepECG_Docker/fairseq-signals")
    from fairseq_signals.utils import checkpoint_utils

    wcr_ssl = "/volume/DeepECG_Docker/weights/wcr_77_classes/base_ssl.pt"
    model, _, _ = checkpoint_utils.load_model_and_task(
        weights_path,
        overrides={"model_path": wcr_ssl},
    )
    model.eval()

    # Remove weight_norm before export
    for name, module in model.named_modules():
        if hasattr(module, "weight_g") and hasattr(module, "weight_v"):
            torch.nn.utils.remove_weight_norm(module)

    class WCRWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            result = self.model(source=x)
            return torch.sigmoid(result["out"])

    wrapper = WCRWrapper(model)
    wrapper.eval()

    dummy = torch.randn(1, 12, 2500)
    onnx_path = os.path.join(EXPORT_DIR, f"{out_name}.onnx")

    torch.onnx.export(
        wrapper, dummy, onnx_path,
        input_names=["ecg_12lead"],
        output_names=["acs_probability"],
        dynamic_axes={
            "ecg_12lead": {0: "batch_size"},
            "acs_probability": {0: "batch_size"},
        },
        opset_version=14,
        do_constant_folding=True,
    )
    size_mb = os.path.getsize(onnx_path) / 1e6
    print(f"  Saved: {onnx_path} ({size_mb:.1f} MB)")

    # Verify
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path)
    out = sess.run(None, {"ecg_12lead": dummy.numpy()})[0]
    with torch.no_grad():
        ref = wrapper(dummy).numpy()
    diff = np.abs(out - ref).max()
    print(f"  Verification: max diff = {diff:.6f} {'OK' if diff < 1e-4 else 'WARNING'}")

    return onnx_path


def export_coreml(onnx_path: str, out_name: str = None):
    """Convert ONNX model to CoreML."""
    try:
        import coremltools as ct
    except ImportError:
        print("  CoreML export skipped (coremltools not installed)")
        return

    if out_name is None:
        out_name = os.path.splitext(os.path.basename(onnx_path))[0]

    mlpackage_path = os.path.join(EXPORT_DIR, f"{out_name}.mlpackage")
    print(f"  Converting to CoreML: {mlpackage_path}")

    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path)
    inputs = sess.get_inputs()

    # Use ct.converters.onnx for conversion
    model = ct.converters.onnx.convert(model=onnx_path)
    model.save(mlpackage_path)
    print(f"  Saved: {mlpackage_path}")


def main():
    parser = argparse.ArgumentParser(description="Export models to ONNX")
    parser.add_argument(
        "--models", nargs="*",
        choices=["unet", "layout", "lead_id", "wcr", "all"],
        default=["all"],
        help="Which models to export",
    )
    parser.add_argument("--coreml", action="store_true", help="Also export CoreML")

    # Weight paths
    parser.add_argument("--unet_weights", default="weights/unet_combined/best_weights.pt")
    parser.add_argument("--layout_weights", default="weights/layout_classifier/best_layout_classifier.pt")
    parser.add_argument("--lead_id_weights", default="weights/lead_identifier_unet.pt",
                        help="Lead identifier U-Net weights")
    parser.add_argument("--wcr_weights",
                        default="/volume/DeepECG_Docker/checkpoints_acs_online_augment_from_ceiling/best_model.pt")

    args = parser.parse_args()
    os.makedirs(EXPORT_DIR, exist_ok=True)

    models_to_export = args.models
    if "all" in models_to_export:
        models_to_export = ["unet", "layout", "lead_id", "wcr"]

    exported = []

    if "unet" in models_to_export:
        if os.path.exists(args.unet_weights):
            path = export_unet(args.unet_weights)
            exported.append(path)
        else:
            print(f"  SKIP: {args.unet_weights} not found")

    if "layout" in models_to_export:
        if os.path.exists(args.layout_weights):
            path = export_layout_classifier(args.layout_weights)
            exported.append(path)
        else:
            print(f"  SKIP: {args.layout_weights} not found")

    if "lead_id" in models_to_export:
        if os.path.exists(args.lead_id_weights):
            path = export_lead_id_unet(args.lead_id_weights)
            exported.append(path)
        else:
            print(f"  SKIP: {args.lead_id_weights} not found")

    if "wcr" in models_to_export:
        if os.path.exists(args.wcr_weights):
            path = export_wcr(args.wcr_weights)
            exported.append(path)
        else:
            print(f"  SKIP: {args.wcr_weights} not found")

    if args.coreml:
        for path in exported:
            export_coreml(path)

    print(f"\n{'='*60}")
    print(f"Done! Exported {len(exported)} models to {EXPORT_DIR}/")
    for p in exported:
        print(f"  {p}")


if __name__ == "__main__":
    main()

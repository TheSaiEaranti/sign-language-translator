"""
inference/export_onnx.py

Export trained models to ONNX for edge deployment / faster inference.

Usage:
    python -m inference.export_onnx --task letters --ckpt checkpoints/letters_best.pt
    python -m inference.export_onnx --task words --ckpt checkpoints/words_best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from models.architectures import StaticGestureNet, DynamicGestureNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["letters", "words"], required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--seq-len", type=int, default=30)
    args = parser.parse_args()

    out_path = Path(args.out or f"{args.task}.onnx")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    num_classes = ckpt.get("num_classes", 26 if args.task == "letters" else 10)

    if args.task == "letters":
        model = StaticGestureNet(num_classes=num_classes)
        dummy = torch.randn(1, 2, 21, 3)
        dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}
    else:
        model = DynamicGestureNet(num_classes=num_classes)
        dummy = torch.randn(1, args.seq_len, 2, 21, 3)
        dynamic_axes = {"input": {0: "batch", 1: "time"}, "output": {0: "batch"}}

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=17,
    )

    print(f"Exported {args.task} model to {out_path}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Val accuracy: {ckpt.get('val_acc', 'unknown')}")


if __name__ == "__main__":
    main()

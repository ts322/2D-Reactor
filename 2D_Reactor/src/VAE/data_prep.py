#!/usr/bin/env python3
"""
Data preparation script for silhouettes/grayscale images.
- Converts to grayscale
- Resizes to square
- Optional binarization
- Splits into train/val/test and writes PNGs

Usage:
    python dataprep.py --src data/raw --out data/processed --img-size 128 --binary --val 0.1 --test 0.1

Outputs:
    out/train/*.png, out/val/*.png, out/test/*.png
"""
from __future__ import annotations
import argparse
import random
from pathlib import Path
import numpy as np
from PIL import Image
from tools import list_image_files


def prepare_images(src: str, out: str, img_size: int = 128, binary: bool = True, val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 0):
    rng = random.Random(seed)
    files = list_image_files(src)
    if len(files) == 0:
        raise SystemExit(f"No images found under {src}")
    rng.shuffle(files)
    n = len(files)
    n_val = int(val_ratio * n)
    n_test = int(test_ratio * n)
    n_train = n - n_val - n_test
    splits = {
        "train": files[:n_train],
        "val": files[n_train:n_train+n_val],
        "test": files[n_train+n_val:],
    }

    for sp in ["train", "val", "test"]:
        (Path(out)/sp).mkdir(parents=True, exist_ok=True)

    for sp, flist in splits.items():
        print(f"Processing {sp}: {len(flist)} images")
        for i, fp in enumerate(flist):
            try:
                img = Image.open(fp).convert("L")
            except Exception:
                continue
            img = img.resize((img_size, img_size), resample=Image.BICUBIC)
            arr = np.array(img).astype(np.float32) / 255.0
            if binary:
                arr = (arr > 0.5).astype(np.float32)
            arr = (arr * 255).astype(np.uint8)
            Image.fromarray(arr, mode="L").save(Path(out)/sp/f"img_{i:06d}.png")

    with open(Path(out)/"README.txt", "w") as f:
        f.write(f"prepared: img_size={img_size}, binary={binary}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--binary", action="store_true")
    ap.add_argument("--val", type=float, default=0.1, dest="val_ratio")
    ap.add_argument("--test", type=float, default=0.1, dest="test_ratio")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    prepare_images(args.src, args.out, args.img_size, args.binary, args.val_ratio, args.test_ratio, args.seed)

if __name__ == "__main__":
    main()

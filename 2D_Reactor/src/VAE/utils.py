#!/usr/bin/env python3
"""
General tools/utilities: file listing, dataset, dataloaders, metrics, and small recon demo.
Place this file alongside vqvae_model.py.

Usage examples:
    from tools import make_loaders, codebook_perplexity
"""
from __future__ import annotations
import os, glob
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# -----------------------------
# Files
# -----------------------------

def list_image_files(root: str) -> List[str]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
    files = []
    for p in glob.glob(os.path.join(root, "**", "*"), recursive=True):
        if os.path.splitext(p)[1].lower() in exts:
            files.append(p)
    return files

# -----------------------------
# Dataset / DataLoaders
# -----------------------------

class SimpleImageFolder(Dataset):
    def __init__(self, root: str, split: str):
        self.root = Path(root) / split
        self.files = sorted(list_image_files(str(self.root)))
        if len(self.files) == 0:
            raise RuntimeError(f"No images found in {self.root}.")
        base_tfms = [transforms.Grayscale(num_output_channels=1), transforms.ToTensor()]
        if split == "train":
            aug = transforms.RandomAffine(degrees=4, translate=(0.02, 0.02), scale=(0.98, 1.02), fill=0)
            self.tfms = transforms.Compose([aug] + base_tfms)
        else:
            self.tfms = transforms.Compose(base_tfms)
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("L")
        return self.tfms(img)


def make_loaders(data_root: str, batch_size: int = 64, num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_set = SimpleImageFolder(data_root, "train")
    val_set   = SimpleImageFolder(data_root, "val")
    test_set  = SimpleImageFolder(data_root, "test")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader

# -----------------------------
# Metrics / Utilities
# -----------------------------

@torch.no_grad()
def codebook_perplexity(codes: torch.Tensor, K: int) -> float:
    hist = torch.bincount(codes.flatten(), minlength=K).float()
    p = hist / (hist.sum() + 1e-8)
    p = p[p > 0]
    H = -(p * p.log()).sum()
    return float(torch.exp(H))

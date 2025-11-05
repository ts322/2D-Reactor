#!/usr/bin/env python3
"""
Training script for VQ-VAE on processed silhouettes.
Assumes processed folder structure created by dataprep.py with train/val/test.

Example:
  python train_vqvae.py --data data/processed --epochs 100 --batch-size 64 \
      --codebook 512 --embed-dim 128 --beta 0.25 --out runs
"""
from __future__ import annotations
import argparse
import os
import torch
from torchvision.utils import save_image
from tqdm import tqdm

from tools import make_loaders, codebook_perplexity
from vqvae_model import VQVAE


def train(
    data_root: str,
    out_dir: str = "runs",
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 2e-4,
    codebook: int = 512,
    embed_dim: int = 128,
    beta: float = 0.25,
    workers: int = 4,
    seed: int = 0,
):
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(out_dir, exist_ok=True)

    train_loader, val_loader, _ = make_loaders(data_root, batch_size=batch_size, num_workers=workers)

    model = VQVAE(K=codebook, D=embed_dim, ch=128).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for x in pbar:
            x = x.to(device)
            _, loss, logs = model(x, beta=beta)
            opt.zero_grad(); loss.backward(); opt.step()
            pbar.set_postfix(recon=float(logs["recon"]), commit=float(logs["commit"]))

        # validation preview
        model.eval()
        with torch.no_grad():
            x = next(iter(val_loader)).to(device)
            x_hat, _, out_logs = model(x, beta=beta)
            perplex = codebook_perplexity(out_logs["codes"].cpu(), codebook)
            grid = torch.cat([x[:8], x_hat[:8]], dim=0)
            save_image(grid, os.path.join(out_dir, f"recon_{epoch:03d}.png"), nrow=8)

        torch.save(model.state_dict(), os.path.join(out_dir, "vqvae.pt"))
        print(f"Saved epoch {epoch}. Codebook perplexity ~ {perplex:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--codebook", type=int, default=512)
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--beta", type=float, default=0.25)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    train(
        data_root=args.data, out_dir=args.out, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, codebook=args.codebook, embed_dim=args.embed_dim, beta=args.beta,
        workers=args.workers, seed=args.seed
    )

if __name__ == "__main__":
    main()

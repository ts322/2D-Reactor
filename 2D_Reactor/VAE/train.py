#!/usr/bin/env python3
"""
Minimal VAE training script — trains and saves model weights (state_dict) only.

Usage example:
  python train_vae_simple.py --csv data.csv --epochs 50 --batch-size 64 \
      --latent-dim 1 --hidden-dims "128,128,64" --lr 1e-3 \
      --save /path/to/vae_weights.pth
"""
import os
import argparse
from typing import Tuple
import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from vae import VAE, vae_loss  # expects vae.py in same folder / importable

def parse_hidden_dims(s: str | None) -> Tuple[int, ...]:
    if not s:
        return ()
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())

def load_dataset(csv_path: str) -> torch.Tensor:
    df = pd.read_csv(csv_path)
    required = {"p1", "p2", "p3"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {sorted(required)}")
    X = df[["p1", "p2", "p3"]].to_numpy(dtype=np.float32)
    return torch.from_numpy(X)

def build_loader(X: torch.Tensor, batch_size: int) -> DataLoader:
    ds = TensorDataset(X)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    p = argparse.ArgumentParser(description="Train VAE and save weights (state_dict).")
    p.add_argument("--csv", default= "/Users/marcobarbacci/2D-Reactor/2D_Reactor/datasets/vae_params.csv")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--latent-dim", type=int, default=1)
    p.add_argument("--hidden-dims", type=str, default="128,128,64",
                   help='Comma list, e.g. "128,128,64"')
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=1.0, help="KL weight in VAE loss")
    p.add_argument("--save", default = "/Users/marcobarbacci/2D-Reactor/2D_Reactor/VAE/Weights/vae_model.pt" )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()
    print("Device:", device)

    # Data
    X = load_dataset(args.csv)            # (N,3) float32
    loader = build_loader(X, args.batch_size)

    # Model
    hidden_dims = parse_hidden_dims(args.hidden_dims)
    if not hidden_dims:
        hidden_dims = (128, 128)  # default fallback

    model = VAE(input_dim=3, latent_dim=args.latent_dim, hidden_dims=hidden_dims).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_samples = 0
        for (batch,) in loader:
            batch = batch.to(device=device, dtype=torch.float32)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            loss = vae_loss(recon, batch, mu, logvar, beta=args.beta, reduction="mean")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n_samples += batch.size(0)
        avg_loss = total_loss / max(n_samples, 1)
        print(f"Epoch {epoch:3d}/{args.epochs}  TrainLoss: {avg_loss:.6f}")

    # Save state_dict only
    save_path = os.path.expanduser(args.save)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print("Saved model weights (state_dict) to:", save_path)

if __name__ == "__main__":
    main()
import os, argparse
from typing import Tuple
import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from ae import AE  
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
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="/Users/marcobarbacci/2D-Reactor/2D_Reactor/datasets/vae_params.csv")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--latent-dim", type=int, default=1)   # try >1 for better variability
    p.add_argument("--batch-size", type=int, default=32)  # try >1 for better variability
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--sample-interval", type=int, default=10, help="save sampled decodes every N epochs (0 to disable)")
    p.add_argument("--save", default="/Users/marcobarbacci/2D-Reactor/2D_Reactor/VAE/Weights/vae_model.pt")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()
    print("Device:", device)

    # Load and normalize data (min-max)
    X = load_dataset(args.csv)   # (N,3) torch
    X_min = X.min(dim=0).values.clone()
    X_max = X.max(dim=0).values.clone()
    eps = 1e-8
    ranges = (X_max - X_min).clamp(min=eps)
    Xn = (X - X_min) / ranges

    loader = build_loader(Xn, args.batch_size)
    
    
    model = AE(latent_dim=args.latent_dim).to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    save_path = os.path.expanduser(args.save)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    norm_path = save_path + ".norm.npz"
    
    sample_z = torch.randn(8, args.latent_dim, device=device)  # 8 sample z
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_recon = 0.0
        n_samples = 0

        for (batch,) in loader:
            batch = batch.to(device=device, dtype=torch.float32)
            optimizer.zero_grad()
            recon, x_hat = model(batch)  # recon shape (B,3)
            # reconstruction: sum over features then mean over batch (same as your vae_loss)
            recon_err = F.mse_loss(recon, batch, reduction="none").sum(dim=1)  # per-sample
            recon_loss = recon_err.mean()
            loss = recon_loss
            loss.backward()
            optimizer.step()

            epoch_recon += float(recon_loss.item()) * batch.size(0)
            n_samples += batch.size(0)

        epoch_recon /= max(n_samples, 1)
        total = epoch_recon
        
        
        print(f"Epoch {epoch:4d}/{args.epochs}  recon={epoch_recon:.6f}  total={total:.6f}")
        
        
        if args.sample_interval > 0 and epoch % args.sample_interval == 0:
            model.eval()
            with torch.no_grad():
                s = model.decoder(sample_z).cpu().numpy()  # shape (M,3)
            # un-normalize
            s_unn = s * ranges.cpu().numpy() + X_min.cpu().numpy()
            # save to CSV for quick inspection
            samples_out = os.path.join(os.path.dirname(save_path), f"samples_epoch_{epoch}.csv")
            np.savetxt(samples_out, s_unn, header="p1,p2,p3", delimiter=",", comments="")
            print(f"  [SAMPLES] wrote {samples_out}")
            
    torch.save(model.state_dict(), save_path)
    np.savez(norm_path, X_min=X_min.cpu().numpy(), X_max=X_max.cpu().numpy())
    print("Saved state_dict ->", save_path)
    print("Saved normalization stats ->", norm_path)

if __name__ == "__main__":
    main()
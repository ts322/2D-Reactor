import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
import yaml

# Import your model + loss
from vae import VAE, vae_loss   # assumes you saved model in vae.py

def main():
    # ----------------------------
    # Parse CLI args
    # ----------------------------
    parser = argparse.ArgumentParser(description="Train a VAE on (p1,p2,p3) dataset.")
    parser.add_argument("--config", default= '/Users/marcobarbacci/foam/4th-Year-Research-Project/2D_Reactor/VAE/config.yml', help="Path to YAML config file")
    parser.add_argument("--epochs", type=int, help="Override: number of epochs")
    parser.add_argument("--batch-size", type=int, help="Override: batch size")
    parser.add_argument("--latent-dim", type=int, help="Override: latent dimension")
    parser.add_argument("--hidden-dim", type=int, help="Override: hidden dimension")
    parser.add_argument("--lr", type=float, help="Override: learning rate")
    parser.add_argument("--beta", type=float, help="Override: beta parameter for beta-VAE")
    parser.add_argument("--csv", type=str, help="Override: dataset path")
    parser.add_argument("--save-model", type=str, help="Override: save model path")
    args = parser.parse_args()

    # ----------------------------
    # Load config file
    # ----------------------------
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # ----------------------------
    # Override with CLI arguments
    # ----------------------------
    for key in ["epochs", "batch_size", "latent_dim", "hidden_dim", "lr", "beta", "csv", "save_model"]:
        val = getattr(args, key.replace("-", "_"))
        if val is not None:
            cfg[key] = val

    # ----------------------------
    # Load dataset
    # ----------------------------
    df = pd.read_csv(cfg["csv"])
    if not set(["p1", "p2", "p3"]).issubset(df.columns):
        raise ValueError("CSV must contain columns: p1, p2, p3")

    X = df[["p1", "p2", "p3"]].values.astype("float64")

    # Normalize to [0,1]
    X_min, X_max = X.min(axis=0), X.max(axis=0)
    X_norm = (X - X_min) / (X_max - X_min + 1e-8)

    dataset = TensorDataset(torch.tensor(X_norm))
    dataloader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)

    # ----------------------------
    # Initialize model
    # ----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(input_dim=3, latent_dim=cfg["latent_dim"], hidden_dim=cfg["hidden_dim"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])

    # ----------------------------
    # Training loop
    # ----------------------------
    model.train()
    for epoch in range(cfg["epochs"]):
        train_loss = 0
        for batch in dataloader:
            batch = batch[0].to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(batch)
            loss = vae_loss(recon_batch, batch, mu, logvar, beta=cfg["beta"])
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        avg_loss = train_loss / len(dataset)
        print(f"Epoch {epoch+1}/{cfg['epochs']}, Loss: {avg_loss:.4f}")

    # ----------------------------
    # Save trained model
    # ----------------------------
    torch.save({
        "model_state": model.state_dict(),
        "X_min": X_min,
        "X_max": X_max,
        "config": cfg,
    }, cfg["save_model"])

    print(f"Model saved to {cfg['save_model']}")


if __name__ == "__main__":
    main()

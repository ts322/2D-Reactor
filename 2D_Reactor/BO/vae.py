import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Deeper VAE (multi-layer MLP)
# ----------------------------
class VAE(nn.Module):
    def __init__(
        self,
        input_dim: int = 3,
        latent_dim: int = 1,
        hidden_dims=(128, 128, 64),   # <- add more layers by extending this tuple
        dropout: float = 0.0,
        use_layernorm: bool = True,
        out_activation: str = "sigmoid",  # "sigmoid" for [0,1] targets; "none" for unbounded
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.out_activation = out_activation.lower()

        # -------- Encoder: [input] -> h_k
        enc_layers = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers.append(nn.Linear(prev, h))
            if use_layernorm:
                enc_layers.append(nn.LayerNorm(h))
            enc_layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                enc_layers.append(nn.Dropout(dropout))
            prev = h
        self.encoder = nn.Sequential(*enc_layers)

        self.fc_mean = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)

        # -------- Decoder: z -> [input]
        dec_layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):  # symmetric decoder
            dec_layers.append(nn.Linear(prev, h))
            if use_layernorm:
                dec_layers.append(nn.LayerNorm(h))
            dec_layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                dec_layers.append(nn.Dropout(dropout))
            prev = h
        self.decoder = nn.Sequential(*dec_layers)
        self.fc_out = nn.Linear(prev, input_dim)

        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # -------- Encode x -> (mu, logvar)
    def encode(self, x: torch.Tensor):
        h = self.encoder(x.float())
        return self.fc_mean(h), self.fc_logvar(h)

    # -------- Reparameterize
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # Deterministic at eval
            return mu

    # -------- Decode z -> x_hat
    def decode(self, z: torch.Tensor):
        h = self.decoder(z.float())
        out = self.fc_out(h)
        if self.out_activation == "sigmoid":
            return torch.sigmoid(out)  # for targets normalized to [0,1]
        return out

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# ----------------------------
# Loss
# ----------------------------
def vae_loss(recon_x, x, mu, logvar, beta: float = 1.0, reduction: str = "mean"):
    """
    - Use MSE when outputs are in [0,1] and targets are min-max normalized.
    - If using BCEWithLogitsLoss, set out_activation='none' and pass raw logits.
    """
    # Reconstruction
    # (Sum across features, then mean across batch for stable scaling)
    recon_err = F.mse_loss(recon_x, x, reduction="none").sum(dim=1)
    if reduction == "mean":
        recon_loss = recon_err.mean()
    elif reduction == "sum":
        recon_loss = recon_err.sum()
    else:
        raise ValueError("reduction must be 'mean' or 'sum'")

    # KL
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl_loss = kl.mean() if reduction == "mean" else kl.sum()

    return recon_loss + beta * kl_loss

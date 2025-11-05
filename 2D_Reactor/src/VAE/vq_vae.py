#!/usr/bin/env python3
"""
VQ-VAE model components (EMA codebook) for 128x128 single-channel images.

Usage:
    from vqvae_model import VQVAE
    model = VQVAE(K=512, D=128, ch=128)
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 1),
        )
    def forward(self, x):
        return x + self.block(x)

class Encoder(nn.Module):
    def __init__(self, in_ch: int = 1, ch: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, ch, 4, stride=2, padding=1),  # 128->64
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 4, stride=2, padding=1),     # 64->32
            ResBlock(ch), ResBlock(ch),
        )
    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, out_ch: int = 1, ch: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            ResBlock(ch), ResBlock(ch),
            nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1),  # 32->64
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ch, out_ch, 4, stride=2, padding=1),  # 64->128
            nn.Sigmoid(),
        )
    def forward(self, z):
        return self.net(z)

class VectorQuantizerEMA(nn.Module):
    def __init__(self, K: int = 512, D: int = 128, decay: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.K, self.D, self.decay, self.eps = K, D, decay, eps
        self.codebook = nn.Parameter(torch.randn(K, D))
        self.register_buffer("ema_count", torch.zeros(K))
        self.register_buffer("ema_weight", torch.randn(K, D))

    def forward(self, z_e: torch.Tensor):
        B, D, H, W = z_e.shape
        flat = z_e.permute(0, 2, 3, 1).reshape(-1, D)
        dist = flat.pow(2).sum(1, keepdim=True) - 2 * flat @ self.codebook.t() + self.codebook.pow(2).sum(1)
        idx = torch.argmin(dist, dim=1)
        z_q = self.codebook[idx].view(B, H, W, D).permute(0, 3, 1, 2).contiguous()

        if self.training:
            with torch.no_grad():
                onehot = torch.nn.functional.one_hot(idx, self.K).type(z_e.dtype)
                n = onehot.sum(0)
                dw = onehot.t() @ flat
                self.ema_count.mul_(self.decay).add_(n, alpha=1 - self.decay)
                self.ema_weight.mul_(self.decay).add_(dw, alpha=1 - self.decay)
                n = (self.ema_count + self.eps)
                self.codebook.data.copy_(self.ema_weight / n.unsqueeze(1))

        loss_commit = F.mse_loss(z_e.detach(), z_q)
        z_q_st = z_e + (z_q - z_e).detach()
        return z_q_st, idx.view(B, H, W), loss_commit

class VQVAE(nn.Module):
    def __init__(self, K: int = 512, D: int = 128, ch: int = 128, out_ch: int = 1):
        super().__init__()
        self.enc = Encoder(out_ch, ch) if out_ch != 1 else Encoder(1, ch)
        self.enc_proj = nn.Conv2d(ch, D, 1)
        self.vq = VectorQuantizerEMA(K, D)
        self.dec_in = nn.Conv2d(D, ch, 1)
        self.dec = Decoder(out_ch, ch)

    def forward(self, x, beta: float = 0.25):
        z_e = self.enc_proj(self.enc(x))
        z_q, codes, loss_commit = self.vq(z_e)
        x_hat = self.dec(self.dec_in(z_q))
        recon = F.binary_cross_entropy(x_hat, x)
        loss = recon + beta * loss_commit
        return x_hat, loss, {"recon": recon.detach(), "commit": loss_commit.detach(), "codes": codes}

@torch.no_grad()
def decode_from_codes(model: 'VQVAE', code_indices: torch.Tensor) -> torch.Tensor:
    """Decode integer code map (B,H,W) using model's codebook + decoder."""
    emb = model.vq.codebook[code_indices.view(-1)].view(code_indices.size(0), code_indices.size(1), code_indices.size(2), -1)
    z_q = emb.permute(0,3,1,2).contiguous()
    x_hat = model.dec(model.dec_in(z_q))
    return x_hat

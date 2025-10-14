import torch
import torch.nn as nn
import torch.nn.functional as F


class AE(nn.Module): 
    def __init__(
        self,
        latent_dim: int =1,
    ): 
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),        
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
        )
        # Decoder: from 2 to 4 to 10 to 20 nodes
        self.decoder = nn.Sequential(
            nn.Linear(1, 4),
            nn.ReLU(),
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
        )
        
        
    def forward(self, x):
        x = x.view(x.size(0), -1) #flatten to bathc size
        x_hat = self.encoder(x)
        return self.decoder(x_hat), x_hat

if __name__ == "__main__":
    # Print model summary 
    model = AE()
    print(model)
               
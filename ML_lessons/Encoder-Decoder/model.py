import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # [1,28,28] → [32,28,28]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # [32,28,28] → [32,14,14]
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # [32,14,14] → [64,14,14]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # [64,14,14] → [64,7,7]
        )

        self.fc_enc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 32),   # [3136] → [128] : compression
            nn.ReLU(),
        )

        self.fc_dec = nn.Sequential(
            nn.Linear(32, 64 * 7 * 7),   # [128] → [3136] : expansion
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # [64,7,7] → [32,14,14]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),   # [32,14,14] → [1,28,28]
            nn.Tanh(),                                             # sortie entre -1 et 1
        )

    def forward(self, x):
        x = self.encoder(x)                     # [1,28,28] → [64,7,7]
        x = x.view(x.size(0), -1)               # [64,7,7] → [3136]
        x = self.fc_enc(x)                      # [3136] → [128]
        x = self.fc_dec(x)                      # [128] → [3136]
        x = x.view(x.size(0), 64, 7, 7)         # [3136] → [64,7,7]
        x = self.decoder(x)                     # [64,7,7] → [1,28,28]
        return x
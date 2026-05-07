# test_hyperparametre.py
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt
import numpy as np
import time

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")

# ── Dataset ───────────────────────────────────────────────────────────────────
def add_noise(img, noise_factor=0.4):
    return torch.clamp(img + noise_factor * torch.randn_like(img), -1, 1)

class NoisyMNIST(Dataset):
    def __init__(self, dataset, noise_factor=0.4):
        self.dataset      = dataset
        self.noise_factor = noise_factor
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        return add_noise(img, self.noise_factor), img

def get_data(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_base = datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
    test_base  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_sub, _ = random_split(train_base, [6000, 54000])
    test_sub,  _ = random_split(test_base,  [1000,  9000])

    train_loader = DataLoader(NoisyMNIST(train_sub), batch_size=batch_size, shuffle=True,  pin_memory=True)
    test_loader  = DataLoader(NoisyMNIST(test_sub),  batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, test_loader

# ── Modèle dynamique ──────────────────────────────────────────────────────────
class Autoencoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc_enc = nn.Sequential(
            nn.Linear(64 * 7 * 7, latent_dim),
            nn.ReLU(),
        )
        self.fc_dec = nn.Sequential(
            nn.Linear(latent_dim, 64 * 7 * 7),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc_enc(x)
        x = self.fc_dec(x)
        x = x.view(x.size(0), 64, 7, 7)
        x = self.decoder(x)
        return x

# ── Entraînement ──────────────────────────────────────────────────────────────
def train_and_eval(latent_dim, batch_size, epochs=5):
    train_loader, test_loader = get_data(batch_size)

    model     = Autoencoder(latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    start = time.time()

    model.train()
    for epoch in range(epochs):
        for noisy_imgs, clean_imgs in train_loader:
            noisy_imgs = noisy_imgs.to(device)
            clean_imgs = clean_imgs.to(device)

            optimizer.zero_grad()
            output = model(noisy_imgs)
            loss   = criterion(output, clean_imgs)
            loss.backward()
            optimizer.step()

    elapsed = time.time() - start

    # Évaluation
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for noisy_imgs, clean_imgs in test_loader:
            noisy_imgs = noisy_imgs.to(device)
            clean_imgs = clean_imgs.to(device)
            output      = model(noisy_imgs)
            total_loss += criterion(output, clean_imgs).item()

    avg_loss = total_loss / len(test_loader)
    return avg_loss, elapsed

# ── Grille d'hyperparamètres ──────────────────────────────────────────────────
latent_dims  = [2, 8, 32, 64]
batch_sizes  = [16, 32, 64, 128]

results = np.zeros((len(latent_dims), len(batch_sizes)))
times   = np.zeros((len(latent_dims), len(batch_sizes)))

for i, latent_dim in enumerate(latent_dims):
    for j, batch_size in enumerate(batch_sizes):
        print(f"latent_dim={latent_dim}, batch_size={batch_size}...")
        loss, elapsed = train_and_eval(latent_dim, batch_size, epochs=5)
        results[i, j] = loss
        times[i, j]   = elapsed
        print(f"  → Loss : {loss:.6f} | Temps : {elapsed:.1f}s")

# ── Heatmap ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(results, cmap='RdYlGn_r', vmin=results.min(), vmax=results.max())

ax.set_xticks(range(len(batch_sizes)))
ax.set_yticks(range(len(latent_dims)))
ax.set_xticklabels([str(b) for b in batch_sizes])
ax.set_yticklabels([str(l) for l in latent_dims])
ax.set_xlabel("Batch size")
ax.set_ylabel("Taille espace latent")
ax.set_title(f"MSE Loss — espace latent vs batch size\n(device: {device}, 5 epochs, 10% MNIST)")

for i in range(len(latent_dims)):
    for j in range(len(batch_sizes)):
        ax.text(j, i - 0.15, f"{results[i, j]:.5f}",
                ha='center', va='center', fontsize=10, fontweight='bold')
        ax.text(j, i + 0.20, f"{times[i, j]:.1f}s",
                ha='center', va='center', fontsize=9, color='dimgray')

plt.colorbar(im, ax=ax, label="MSE Loss (plus bas = meilleur)")
plt.tight_layout()
plt.show()
# hyperparam_search.py
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import time

# Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_full = datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
test_full  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_size = int(0.1 * len(train_full))
test_size  = int(0.1 * len(test_full))
train_dataset, _ = random_split(train_full, [train_size, len(train_full) - train_size])
test_dataset,  _ = random_split(test_full,  [test_size,  len(test_full)  - test_size])

print(f"Train : {len(train_dataset)} images | Test : {len(test_dataset)} images")

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)

# Modèle dynamique avec 2 conv_blocks
class CNN(nn.Module):
    def __init__(self, k1, k2, ks1, ks2):
        super(CNN, self).__init__()
        p1 = ks1 // 2
        p2 = ks2 // 2

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1,  k1, kernel_size=ks1, padding=p1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # 28x28 → 14x14
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(k1, k2, kernel_size=ks2, padding=p2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # 14x14 → 7x7
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(k2 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

def train_and_eval(k1, k2, ks1, ks2, epochs=3):
    model = CNN(k1, k2, ks1, ks2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    start = time.time()

    model.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

    elapsed = time.time() - start

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            _, predicted = torch.max(model(images), dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total, elapsed

# Grille : kernel sizes fixes (3x3 pour les deux blocs)
# On fait varier k1 (kernels bloc1) et k2 (kernels bloc2)
k1_list  = [8, 16, 32, 64]
k2_list  = [16, 32, 64, 128]
ks1, ks2 = 3, 3  # kernel size fixé à 3x3

results = np.zeros((len(k1_list), len(k2_list)))
times   = np.zeros((len(k1_list), len(k2_list)))

print("\n=== Variation k1 / k2 (kernel_size=3x3) ===")
for i, k1 in enumerate(k1_list):
    for j, k2 in enumerate(k2_list):
        print(f"  k1={k1}, k2={k2}...")
        acc, elapsed = train_and_eval(k1, k2, ks1, ks2)
        results[i, j] = acc
        times[i, j]   = elapsed
        print(f"  → {acc:.2f}% | {elapsed:.1f}s")

# Heatmap k1 vs k2
fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(results, cmap='RdYlGn', vmin=90, vmax=100)
ax.set_xticks(range(len(k2_list)))
ax.set_yticks(range(len(k1_list)))
ax.set_xticklabels([str(k) for k in k2_list])
ax.set_yticklabels([str(k) for k in k1_list])
ax.set_xlabel("Kernels bloc 2 (k2)")
ax.set_ylabel("Kernels bloc 1 (k1)")
ax.set_title("Accuracy — variation k1/k2 (kernel_size=3×3, 10% dataset)")
for i in range(len(k1_list)):
    for j in range(len(k2_list)):
        ax.text(j, i - 0.15, f"{results[i, j]:.2f}%",
                ha='center', va='center', fontsize=10, fontweight='bold')
        ax.text(j, i + 0.20, f"{times[i, j]:.1f}s",
                ha='center', va='center', fontsize=8, color='dimgray')
plt.colorbar(im, ax=ax, label="Accuracy (%)")
plt.tight_layout()
plt.show()

# ── 2e figure : kernel_size bloc1 vs kernel_size bloc2 ──────────────────────
ks1_list = [3, 5, 7]
ks2_list = [3, 5, 7]
k1_fix, k2_fix = 32, 64  # nombres de kernels fixés

results2 = np.zeros((len(ks1_list), len(ks2_list)))
times2   = np.zeros((len(ks1_list), len(ks2_list)))

print("\n=== Variation ks1 / ks2 (k1=32, k2=64) ===")
for i, ks1 in enumerate(ks1_list):
    for j, ks2 in enumerate(ks2_list):
        print(f"  ks1={ks1}, ks2={ks2}...")
        acc, elapsed = train_and_eval(k1_fix, k2_fix, ks1, ks2)
        results2[i, j] = acc
        times2[i, j]   = elapsed
        print(f"  → {acc:.2f}% | {elapsed:.1f}s")

# Heatmap ks1 vs ks2
fig, ax = plt.subplots(figsize=(7, 6))
im2 = ax.imshow(results2, cmap='RdYlGn', vmin=90, vmax=100)
ax.set_xticks(range(len(ks2_list)))
ax.set_yticks(range(len(ks1_list)))
ax.set_xticklabels([f"{k}×{k}" for k in ks2_list])
ax.set_yticklabels([f"{k}×{k}" for k in ks1_list])
ax.set_xlabel("Kernel size bloc 2")
ax.set_ylabel("Kernel size bloc 1")
ax.set_title("Accuracy — variation kernel sizes (k1=32, k2=64, 10% dataset)")
for i in range(len(ks1_list)):
    for j in range(len(ks2_list)):
        ax.text(j, i - 0.15, f"{results2[i, j]:.2f}%",
                ha='center', va='center', fontsize=10, fontweight='bold')
        ax.text(j, i + 0.20, f"{times2[i, j]:.1f}s",
                ha='center', va='center', fontsize=8, color='dimgray')
plt.colorbar(im2, ax=ax, label="Accuracy (%)")
plt.tight_layout()
plt.show()
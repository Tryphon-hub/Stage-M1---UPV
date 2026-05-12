import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def evaluate(model, test_loader, device=None):
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    criterion  = nn.MSELoss()
    total_loss = 0

    with torch.no_grad():
        for noisy_imgs, clean_imgs in test_loader:
            noisy_imgs = noisy_imgs.to(device)
            clean_imgs = clean_imgs.to(device)
            output      = model(noisy_imgs)
            total_loss += criterion(output, clean_imgs).item()

    print(f"MSE moyen sur le test set : {total_loss/len(test_loader):.6f}")

def visualize(model, test_loader, n=10, device=None):
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    noisy_imgs, clean_imgs = next(iter(test_loader))  # premier batch
    noisy_imgs = noisy_imgs.to(device)
    clean_imgs = clean_imgs.to(device)

    with torch.no_grad():
        reconstructed = model(noisy_imgs)              # reconstruction

    noisy_imgs    = noisy_imgs.cpu()
    clean_imgs    = clean_imgs.cpu()
    reconstructed = reconstructed.cpu()

    fig, axes = plt.subplots(3, n, figsize=(15, 5))

    for i in range(n):
        # ligne 0 — image propre
        axes[0, i].imshow(clean_imgs[i].squeeze(), cmap='gray')

        # ligne 1 — image bruitée
        axes[1, i].imshow(noisy_imgs[i].squeeze(), cmap='gray')

        # ligne 2 — image reconstruite
        axes[2, i].imshow(reconstructed[i].squeeze(), cmap='gray')

        for row in range(3):
            axes[row, i].axis('off')

    axes[0, 0].set_ylabel("Propre",      fontsize=12)
    axes[1, 0].set_ylabel("Bruité",      fontsize=12)
    axes[2, 0].set_ylabel("Reconstruit", fontsize=12)

    plt.suptitle("Débruitage par autoencoder", fontsize=14)
    plt.tight_layout()
    plt.show()
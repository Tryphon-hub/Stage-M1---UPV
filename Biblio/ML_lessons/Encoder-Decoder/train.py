import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def train(model, train_loader, epochs=10, device=None):
    if device is None:
        device = next(model.parameters()).device
    criterion = nn.MSELoss()                                    # erreur pixel par pixel
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    loss_history = []

    for epoch in range(epochs):
        total_loss = 0

        for noisy_imgs, clean_imgs in train_loader:
            noisy_imgs = noisy_imgs.to(device)
            clean_imgs = clean_imgs.to(device)
            optimizer.zero_grad()

            output = model(noisy_imgs)          # reconstruit depuis l'image bruitée
            loss   = criterion(output, noisy_imgs)  # compare à l'image PROPRE

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, epochs+1), loss_history, marker='o', color='steelblue')
    plt.title("Évolution de la loss (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return loss_history
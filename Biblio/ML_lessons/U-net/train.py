# train.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def train(model, train_loader, epochs=10, device=None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Training sur : {device}")

    criterion = nn.CrossEntropyLoss() # erreur de classification px par px
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.to(device)
    model.train()
    loss_history = []

    for epoch in range(epochs):
        total_loss = 0

        for images, masks in train_loader:
            images = images.to(device)
            masks  = masks.to(device)

            optimizer.zero_grad()

            output = model(images)
            loss   = criterion(output, masks)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, epochs+1), loss_history, marker='o', color='steelblue')
    plt.title("Évolution de la loss (CrossEntropy)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return loss_history
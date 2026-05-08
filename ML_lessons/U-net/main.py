# main.py
import torch
from dataset import get_dataloaders
from model import UNet
from train import train
from evaluate import evaluate, visualize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")

# 1. Chargement des données
train_loader, test_loader = get_dataloaders(batch_size=16)

# 2. Modèle
model = UNet(n_classes=3).to(device)
print(model)

# 3. Entraînement
loss_history = train(model, train_loader, epochs=10)

# 4. Évaluation
evaluate(model, test_loader)

# 5. Visualisation
visualize(model, test_loader)

# 6. Sauvegarde
torch.save(model.state_dict(), "unet_pets.pth")
print("Modèle sauvegardé dans unet_pets.pth")
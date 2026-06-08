# main.py
import torch
from dataset import get_dataloaders
from model import CNN
from train import train
from evaluate import evaluate, visualize

# 1. Chargement des données
train_loader, test_loader = get_dataloaders(batch_size=64)

# 2. Initialisation du modèle
model = CNN()
print(model)  # affiche l'architecture

# 3. Entraînement
train(model, train_loader, epochs=5)

# 4. Évaluation
evaluate(model, test_loader)

# 5. Visualisation
visualize(model, test_loader)

# 6. Sauvegarde du modèle (optionnel)
torch.save(model.state_dict(), "1_conv_block.pth")
print("Modèle sauvegardé dans 1_conv_block.pth")
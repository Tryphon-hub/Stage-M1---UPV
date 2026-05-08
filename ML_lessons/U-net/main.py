# main.py
import torch
import time
from datetime import datetime
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

# 3. Entraînement
print(f"\nDébut : {datetime.now().strftime('%H:%M:%S')}")
start = time.time()

loss_history = train(model, train_loader, epochs=10)

elapsed = time.time() - start
heures  = int(elapsed // 3600)
minutes = int((elapsed % 3600) // 60)
secondes = int(elapsed % 60)

print(f"Fin   : {datetime.now().strftime('%H:%M:%S')}")
print(f"Durée : {heures:02d}h {minutes:02d}m {secondes:02d}s")

# 4. Évaluation
evaluate(model, test_loader)

# 5. Visualisation
visualize(model, test_loader)

# 6. Sauvegarde
torch.save(model.state_dict(), "unet_pets.pth")
print("Modèle sauvegardé dans unet_pets.pth")
import torch
from dataset import get_dataloaders
from model import Autoencoder
from train import train
from evaluate import evaluate, visualize
from torch.utils.data import DataLoader, random_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")

train_loader, test_loader = get_dataloaders(batch_size=64, noise_factor=1)

train_size = int(1 * len(train_loader.dataset))
test_size  = int(1 * len(test_loader.dataset))
train_sub, _ = random_split(train_loader.dataset, [train_size, len(train_loader.dataset) - train_size])
test_sub,  _ = random_split(test_loader.dataset,  [test_size,  len(test_loader.dataset)  - test_size])
train_loader  = DataLoader(train_sub, batch_size=64, shuffle=True)
test_loader   = DataLoader(test_sub,  batch_size=64, shuffle=False)

print(f"Train : {len(train_sub)} images | Test : {len(test_sub)} images")

model = Autoencoder().to(device)
print(model)

train(model, train_loader, epochs=10, device=device)
evaluate(model, test_loader, device=device)
visualize(model, test_loader, device=device)
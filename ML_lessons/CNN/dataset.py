# dataset.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),# TODO 1 : convertir en tensor
        transforms.Normalize((0.5,), (0.5,))# TODO 2 : normaliser avec mean=0.5 et std=0.5
    ])

    train_dataset = datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
    test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)

    return train_loader, test_loader
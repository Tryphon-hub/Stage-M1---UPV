import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

nf=1 

def add_noise(img, noise_factor=nf):
    noise     = torch.randn_like(img)           # bruit gaussien, même shape que img
    noisy_img = img + noise_factor * noise      # ajoute le bruit pondéré
    return torch.clamp(noisy_img, -1, 1)        # force les valeurs entre -1 et 1

class NoisyMNIST(Dataset):
    def __init__(self, dataset, noise_factor=nf):
        self.dataset      = dataset
        self.noise_factor = noise_factor

    def __len__(self):
        return len(self.dataset)                # même taille que le dataset original

    def __getitem__(self, idx):
        img, label    = self.dataset[idx]       # récupère l'image propre
        noisy_img     = add_noise(img, self.noise_factor)  # bruitage
        return noisy_img, img                   # retourne (bruité, propre)

def get_dataloaders(batch_size=64, noise_factor=nf):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_base   = datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
    test_base    = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_dataset = NoisyMNIST(train_base, noise_factor)   # wrap le dataset avec bruit
    test_dataset  = NoisyMNIST(test_base,  noise_factor)

    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader   = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
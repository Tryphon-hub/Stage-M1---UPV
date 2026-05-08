# dataset.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

def get_dataloaders(batch_size=16):

    img_transform = transforms.Compose([
        transforms.Resize((128, 128)),                        # redimensionne en 128×128
        transforms.ToTensor(),                                # PIL → tensor [3, 128, 128]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalise RGB
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((128, 128), interpolation=Image.NEAREST),  # NEAREST : pas d'interpolation sur les classes
        transforms.PILToTensor(),                             # PIL → tensor [1, 128, 128] entier
        transforms.Lambda(lambda x: x.squeeze(0).long() - 1) # [1,128,128] → [128,128] classes 0,1,2
    ])

    train_dataset = datasets.OxfordIIITPet(
        root='./data',
        split='trainval',
        target_types='segmentation',
        download=True,
        transform=img_transform,
        target_transform=mask_transform
    )

    test_dataset = datasets.OxfordIIITPet(
        root='./data',
        split='test',
        target_types='segmentation',
        download=True,
        transform=img_transform,
        target_transform=mask_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, pin_memory=True)

    print(f"Train : {len(train_dataset)} | Test : {len(test_dataset)}")
    return train_loader, test_loader
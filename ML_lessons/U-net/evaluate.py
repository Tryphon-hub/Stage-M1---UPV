# evaluate.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

COLORS = np.array([
    [50,  50,  50],    # classe 0 — fond     → gris foncé
    [255, 0,   0  ],   # classe 1 — animal   → rouge
    [0,   255, 0  ],   # classe 2 — contour  → vert
], dtype=np.uint8)

def mask_to_rgb(mask):
    """Convertit un masque [H, W] int64 en image RGB [H, W, 3]"""
    return COLORS[mask.numpy()]

def evaluate(model, test_loader, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    iou_per_class = torch.zeros(3, device=device)
    count         = torch.zeros(3, device=device)

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks  = masks.to(device)

            output    = model(images)                    # [batch, 3, 128, 128]
            predicted = torch.argmax(output, dim=1)      # [batch, 128, 128]

            for cls in range(3):
                intersection = ((predicted == cls) & (masks == cls)).sum().float()
                union        = ((predicted == cls) | (masks == cls)).sum().float()
                if union > 0:
                    iou_per_class[cls] += intersection / union
                    count[cls]         += 1

    iou_fond    = iou_per_class[0] / count[0]
    iou_animal  = iou_per_class[1] / count[1]
    iou_contour = iou_per_class[2] / count[2]
    mean_iou    = (iou_fond + iou_animal + iou_contour) / 3

    print(f"IoU fond    : {iou_fond:.4f}")
    print(f"IoU animal  : {iou_animal:.4f}")
    print(f"IoU contour : {iou_contour:.4f}")
    print(f"Mean IoU    : {mean_iou:.4f}")

    return mean_iou

def visualize(model, test_loader, device=None, n=5):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    images, masks = next(iter(test_loader))
    images_gpu    = images.to(device)

    with torch.no_grad():
        output    = model(images_gpu)
        predicted = torch.argmax(output, dim=1).cpu()  # [batch, 128, 128]

    fig, axes = plt.subplots(3, n, figsize=(15, 7))

    for i in range(n):
        # Image originale — dénormaliser
        img = images[i].permute(1, 2, 0).numpy()       # [3,128,128] → [128,128,3]
        img = (img * 0.5 + 0.5).clip(0, 1)             # [-1,1] → [0,1]
        axes[0, i].imshow(img)

        # Masque réel
        axes[1, i].imshow(mask_to_rgb(masks[i].squeeze()))

        # Masque prédit
        axes[2, i].imshow(mask_to_rgb(predicted[i]))

        for row in range(3):
            axes[row, i].axis('off')

    axes[0, 0].set_ylabel("Image",         fontsize=12)
    axes[1, 0].set_ylabel("Masque réel",   fontsize=12)
    axes[2, 0].set_ylabel("Masque prédit", fontsize=12)

    plt.suptitle("Segmentation U-Net — Oxford Pets", fontsize=14)
    plt.tight_layout()
    plt.show()
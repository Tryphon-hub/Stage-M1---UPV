# process_vizu.py
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataset import get_dataloaders
from model import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")

COLORS = np.array([
    [50,  50,  50],
    [255, 0,   0  ],
    [0,   255, 0  ],
], dtype=np.uint8)

def denorm(img):
    return (img * 0.5 + 0.5).clip(0, 1)

def mask_to_rgb(mask):
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    return COLORS[mask]

# ── Hooks ─────────────────────────────────────────────────────────────────────
activations = {}

def make_hook(name):
    def hook(module, input, output):
        activations[name] = output.detach().cpu()
    return hook

# ── Chargement modèle ─────────────────────────────────────────────────────────
def load_model():
    model = UNet(n_classes=3).to(device)
    try:
        model.load_state_dict(torch.load("results/unet_pets.pth", map_location=device))
        print("Modèle chargé depuis results/unet_pets.pth")
    except:
        print("Pas de modèle sauvegardé — modèle non entraîné utilisé")
    model.eval()
    return model

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_batch(test_loader, n=None):
    images, masks = next(iter(test_loader))
    if n is not None:
        images = images[:n]
        masks  = masks[:n]
    return images.cpu(), masks.cpu()

def run_model(model, images):
    with torch.no_grad():
        output    = model(images.to(device))
        predicted = torch.argmax(output, dim=1).cpu()
    return output.cpu(), predicted

# ── 1. Input / Output ─────────────────────────────────────────────────────────
def vizu_input_output(model, test_loader, n=5):
    images, masks     = get_batch(test_loader, n)
    output, predicted = run_model(model, images)

    fig, axes = plt.subplots(3, n, figsize=(15, 7))
    fig.suptitle("Input / Output du U-Net", fontsize=14)

    for i in range(n):
        img = denorm(images[i].permute(1, 2, 0).numpy())
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"Image {i+1}")
        axes[1, i].imshow(mask_to_rgb(masks[i]))
        axes[1, i].set_title("Masque réel")
        axes[2, i].imshow(mask_to_rgb(predicted[i]))
        axes[2, i].set_title("Masque prédit")
        for row in range(3):
            axes[row, i].axis('off')

    axes[0, 0].set_ylabel("Image",         fontsize=11)
    axes[1, 0].set_ylabel("Masque réel",   fontsize=11)
    axes[2, 0].set_ylabel("Masque prédit", fontsize=11)
    plt.tight_layout()
    plt.show()

# ── 2. Feature maps par couche ────────────────────────────────────────────────
def vizu_feature_maps(model, test_loader, n_filters=8):
    hooks = [
        model.inc.register_forward_hook(make_hook('inc')),
        model.down1.register_forward_hook(make_hook('down1')),
        model.down2.register_forward_hook(make_hook('down2')),
        model.down3.register_forward_hook(make_hook('down3')),
        model.bottleneck.register_forward_hook(make_hook('bottleneck')),
        model.up1.register_forward_hook(make_hook('up1')),
        model.up2.register_forward_hook(make_hook('up2')),
        model.up3.register_forward_hook(make_hook('up3')),
        model.up4.register_forward_hook(make_hook('up4')),
    ]

    images, _ = get_batch(test_loader, n=1)
    with torch.no_grad():
        model(images.to(device))

    for h in hooks:
        h.remove()

    layer_names = ['inc', 'down1', 'down2', 'down3', 'bottleneck',
                   'up1', 'up2', 'up3', 'up4']

    fig, axes = plt.subplots(len(layer_names), n_filters + 1, figsize=(20, 18))
    fig.suptitle("Feature maps par couche", fontsize=14)

    for row, name in enumerate(layer_names):
        feat = activations[name][0]

        axes[row, 0].text(0.5, 0.5, f"{name}\n{list(feat.shape)}",
                          ha='center', va='center', fontsize=9,
                          color='white', bbox=dict(boxstyle='round', facecolor='steelblue'))
        axes[row, 0].axis('off')

        for col in range(n_filters):
            if col < feat.shape[0]:
                axes[row, col+1].imshow(feat[col].numpy(), cmap='viridis')
            axes[row, col+1].axis('off')

    plt.tight_layout()
    plt.show()

# ── 3. Skip connections ───────────────────────────────────────────────────────
def vizu_skip_connections(model, test_loader):
    skip_inputs  = {}
    skip_outputs = {}

    def make_hook_up(name):
        def hook(module, input, output):
            skip_inputs[name]  = input[0].detach().cpu()
            skip_outputs[name] = output.detach().cpu()
        return hook

    hooks = [
        model.up1.register_forward_hook(make_hook_up('up1')),
        model.up2.register_forward_hook(make_hook_up('up2')),
        model.up3.register_forward_hook(make_hook_up('up3')),
        model.up4.register_forward_hook(make_hook_up('up4')),
    ]

    images, _ = get_batch(test_loader, n=1)
    with torch.no_grad():
        model(images.to(device))

    for h in hooks:
        h.remove()

    fig, axes = plt.subplots(3, 4, figsize=(18, 9))
    fig.suptitle("Skip connections — avant / après fusion", fontsize=14)

    for col, name in enumerate(['up1', 'up2', 'up3', 'up4']):
        inp = skip_inputs[name][0]
        out = skip_outputs[name][0]

        axes[0, col].set_title(f"{name}", fontsize=10)

        axes[0, col].imshow(inp.mean(0).numpy(), cmap='plasma')
        axes[0, col].set_ylabel("Entrée décodeur", fontsize=8)
        axes[0, col].text(2, 4, f"{list(inp.shape)}", fontsize=7, color='white')
        axes[0, col].axis('off')

        axes[1, col].imshow(out.mean(0).numpy(), cmap='plasma')
        axes[1, col].set_ylabel("Après fusion skip", fontsize=8)
        axes[1, col].text(2, 4, f"{list(out.shape)}", fontsize=7, color='white')
        axes[1, col].axis('off')

        # Redimensionne inp à la taille de out pour calculer la différence
        inp_resized = F.interpolate(
            inp.unsqueeze(0),
            size=out.shape[1:],
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        diff = (out.mean(0) - inp_resized.mean(0)).numpy()
        axes[2, col].imshow(diff, cmap='RdBu')
        axes[2, col].set_ylabel("Différence", fontsize=8)
        axes[2, col].axis('off')

    plt.tight_layout()
    plt.show()

# ── 4. Espace latent ──────────────────────────────────────────────────────────
def vizu_espace_latent(model, test_loader, n_imgs=4):
    bottleneck_acts = []

    def hook_bn(module, input, output):
        bottleneck_acts.append(output.detach().cpu())

    h = model.bottleneck.register_forward_hook(hook_bn)

    images, masks     = get_batch(test_loader, n_imgs)
    output, predicted = run_model(model, images)

    h.remove()

    bn = bottleneck_acts[0]

    fig = plt.figure(figsize=(18, n_imgs * 3))
    fig.suptitle("Espace latent — goulot [1024 × 8 × 8]", fontsize=13)

    gs = gridspec.GridSpec(n_imgs, 5, figure=fig)

    for i in range(n_imgs):
        ax0 = fig.add_subplot(gs[i, 0])
        ax0.imshow(denorm(images[i].permute(1, 2, 0).numpy()))
        ax0.set_title("Image" if i == 0 else "")
        ax0.axis('off')

        ax1 = fig.add_subplot(gs[i, 1])
        ax1.imshow(mask_to_rgb(masks[i]))
        ax1.set_title("Masque réel" if i == 0 else "")
        ax1.axis('off')

        ax2 = fig.add_subplot(gs[i, 2])
        ax2.imshow(bn[i].mean(0).numpy(), cmap='viridis')
        ax2.set_title("Latent moyen [8×8]" if i == 0 else "")
        ax2.axis('off')

        ax3 = fig.add_subplot(gs[i, 3])
        ax3.imshow(bn[i].var(0).numpy(), cmap='hot')
        ax3.set_title("Variance [8×8]" if i == 0 else "")
        ax3.axis('off')

        ax4 = fig.add_subplot(gs[i, 4])
        ax4.imshow(mask_to_rgb(predicted[i]))
        ax4.set_title("Masque prédit" if i == 0 else "")
        ax4.axis('off')

    plt.tight_layout()
    plt.show()

# ── 5. Tailles par couche ─────────────────────────────────────────────────────
def vizu_tailles(model, test_loader):
    tailles = {}

    def make_hook_size(name):
        def hook(module, input, output):
            tailles[name] = tuple(output.shape)
        return hook

    hooks = [
        model.inc.register_forward_hook(make_hook_size('inc\n[64,128,128]')),
        model.down1.register_forward_hook(make_hook_size('down1\n[128,64,64]')),
        model.down2.register_forward_hook(make_hook_size('down2\n[256,32,32]')),
        model.down3.register_forward_hook(make_hook_size('down3\n[512,16,16]')),
        model.bottleneck.register_forward_hook(make_hook_size('bottleneck\n[1024,8,8]')),
        model.up1.register_forward_hook(make_hook_size('up1\n[512,16,16]')),
        model.up2.register_forward_hook(make_hook_size('up2\n[256,32,32]')),
        model.up3.register_forward_hook(make_hook_size('up3\n[128,64,64]')),
        model.up4.register_forward_hook(make_hook_size('up4\n[64,128,128]')),
    ]

    images, _ = get_batch(test_loader, n=1)
    with torch.no_grad():
        model(images.to(device))

    for h in hooks:
        h.remove()

    noms   = list(tailles.keys())
    canaux = [tailles[n][1] for n in noms]
    resol  = [tailles[n][2] for n in noms]

    fig, axes = plt.subplots(2, 1, figsize=(14, 6))
    fig.suptitle("Évolution des dimensions par couche", fontsize=13)

    x        = range(len(noms))
    couleurs = ['#534AB7' if 'down' in n or 'inc' in n
                else '#993C1D' if 'bottle' in n
                else '#0F6E56' for n in noms]

    axes[0].bar(x, canaux, color=couleurs)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(noms, fontsize=8)
    axes[0].set_ylabel("Nombre de canaux")
    axes[0].set_title("Canaux par couche")

    axes[1].bar(x, resol, color=couleurs)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(noms, fontsize=8)
    axes[1].set_ylabel("Résolution spatiale (px)")
    axes[1].set_title("Résolution par couche")

    plt.tight_layout()
    plt.show()

# ── 6. Scores de confiance ────────────────────────────────────────────────────
def vizu_confiance(model, test_loader, n=3):
    images, masks     = get_batch(test_loader, n)
    output, predicted = run_model(model, images)

    softmax = torch.softmax(output, dim=1)

    fig, axes = plt.subplots(n, 4, figsize=(16, n * 4))
    fig.suptitle("Scores de confiance par classe", fontsize=13)

    titres = ["Image", "Confiance fond", "Confiance animal", "Confiance contour"]
    for col, t in enumerate(titres):
        axes[0, col].set_title(t, fontsize=10)

    for i in range(n):
        axes[i, 0].imshow(denorm(images[i].permute(1, 2, 0).numpy()))
        axes[i, 0].axis('off')

        for cls in range(3):
            im = axes[i, cls+1].imshow(softmax[i, cls].numpy(),
                                        cmap='hot', vmin=0, vmax=1)
            axes[i, cls+1].axis('off')
            plt.colorbar(im, ax=axes[i, cls+1], fraction=0.046)

    plt.tight_layout()
    plt.show()

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders(batch_size=16)
    model = load_model()

    print("\n1. Input / Output")
    vizu_input_output(model, test_loader)

    print("\n2. Feature maps par couche")
    vizu_feature_maps(model, test_loader)

    print("\n3. Skip connections")
    vizu_skip_connections(model, test_loader)

    print("\n4. Espace latent")
    vizu_espace_latent(model, test_loader)

    print("\n5. Tailles par couche")
    vizu_tailles(model, test_loader)

    print("\n6. Scores de confiance")
    vizu_confiance(model, test_loader)
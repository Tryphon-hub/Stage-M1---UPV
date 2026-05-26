# train.py  —  U-Net / BE_UNet
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


# ─── Loss ────────────────────────────────────────────────────────────────────

class sMAPELoss(nn.Module):
    """
    Symmetric Mean Absolute Percentage Error, pixel-à-pixel.

        L = (1/N) * sum_i  2|σ_i - σ̃_i| / (|σ_i| + |σ̃_i| + ε)
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num   = 2.0 * (pred - target).abs()
        denom = pred.abs() + target.abs() + self.eps
        return (num / denom).mean()


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _batch_to_tensors(batch: dict, device: torch.device, NETWORK: str = 'U-Net'):
    """
    Construit les tenseurs d'entrée/sortie depuis un batch.

    NETWORK='U-Net'   → retourne (x, y)          x : [B, 3, H, W]  — ρ + tx + ty
    NETWORK='BE_UNet' → retourne (rho, nodes, y)  rho : [B, 1, H, W], nodes : [B, 16]
    """
    densities = batch['Densities']   # [B, 1, n_pixels]
    stress    = batch['Stress']      # [B, n_pixels, 6]
    tractions = batch['Tractions']   # [B, 1, 2, 8]

    B        = densities.shape[0]
    n_pixels = densities.shape[-1]
    img_size = int(n_pixels ** 0.5)

    # ── densité → carte 2D [B, 1, H, W] ──
    rho = densities.squeeze(1).reshape(B, 1, img_size, img_size)

    # ── stress → [B, 3, H, W] ──
    sigma_x = stress[:, :, 0].reshape(B, 1, img_size, img_size)
    sigma_y = stress[:, :, 1].reshape(B, 1, img_size, img_size)
    tau_xy  = stress[:, :, 3].reshape(B, 1, img_size, img_size)
    y = torch.cat([sigma_x, sigma_y, tau_xy], dim=1).to(device)

    if NETWORK == 'U-Net':
        # tractions → cartes 2D [B, 2, H, W]
        tx_map, ty_map = _tractions_to_maps(tractions, img_size, torch.device('cpu'))
        x = torch.cat([rho, tx_map, ty_map], dim=1).to(device)   # [B, 3, H, W]
        return x, y

    elif NETWORK == 'BE_UNet':
        # scalaires nodaux → [B, 16]
        T     = tractions.squeeze(1)            # [B, 2, 8]
        nodes = T.reshape(B, 16).to(device)     # [B, 16]
        rho   = rho.to(device)                  # [B, 1, H, W]
        return rho, nodes, y

    else:
        raise ValueError(f"Unknown NETWORK '{NETWORK}'. Must be 'U-Net' or 'BE_UNet'.")


def _tractions_to_maps(tractions: torch.Tensor, img_size: int,
                       device: torch.device) -> tuple:
    """
    Convertit les forces nodales (B, 1, 2, 8) en deux cartes 2D (B, 1, H, W)
    par interpolation linéaire le long des 4 bords.
    """
    import numpy as np

    B = tractions.shape[0]
    T = tractions.squeeze(1).numpy()   # [B, 2, 8]

    points = np.array([
        [0,          img_size - 1],
        [img_size-1, img_size - 1],
        [img_size-1, img_size - 1],
        [img_size-1, 0           ],
        [img_size-1, 0           ],
        [0,          0           ],
        [0,          0           ],
        [0,          img_size - 1],
    ], dtype=float)

    tx_batch = np.zeros((B, img_size, img_size), dtype=np.float32)
    ty_batch = np.zeros((B, img_size, img_size), dtype=np.float32)

    for b in range(B):
        for k in range(0, 8, 2):
            p1 = points[k]
            p2 = points[k + 1]
            xs = np.round(np.linspace(p1[0], p2[0], img_size)).astype(int)
            ys = np.round(np.linspace(p1[1], p2[1], img_size)).astype(int)
            tx_batch[b, ys, xs] += np.linspace(T[b, 0, k], T[b, 0, k+1], img_size)
            ty_batch[b, ys, xs] += np.linspace(T[b, 1, k], T[b, 1, k+1], img_size)

    tx_map = torch.from_numpy(tx_batch).unsqueeze(1).to(device)
    ty_map = torch.from_numpy(ty_batch).unsqueeze(1).to(device)
    return tx_map, ty_map


def _forward(model, batch_tensors, NETWORK: str):
    """Appelle model() avec les bons arguments selon NETWORK."""
    if NETWORK == 'U-Net':
        x, y = batch_tensors
        return model(x), y
    elif NETWORK == 'BE_UNet':
        rho, nodes, y = batch_tensors
        return model(rho, nodes), y


# ─── Checkpoint ──────────────────────────────────────────────────────────────

def save_checkpoint(path, model, optimizer, scheduler,
                    epoch, best_val, train_losses, val_losses):
    torch.save({
        'epoch'        : epoch,
        'model_state'  : model.state_dict(),
        'optim_state'  : optimizer.state_dict(),
        'sched_state'  : scheduler.state_dict(),
        'best_val'     : best_val,
        'train_losses' : train_losses,
        'val_losses'   : val_losses,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optim_state'])
    scheduler.load_state_dict(ckpt['sched_state'])

    epoch_start  = ckpt['epoch'] + 1
    best_val     = ckpt['best_val']
    train_losses = ckpt['train_losses']
    val_losses   = ckpt['val_losses']

    print(f"  Checkpoint chargé : epoch {ckpt['epoch']}  "
          f"best_val={best_val:.5f}  "
          f"({len(train_losses)} epochs effectuées)")

    return epoch_start, best_val, train_losses, val_losses


# ─── Entraînement ────────────────────────────────────────────────────────────

def train(model, train_loader, val_loader=None,
          epochs          : int   = 50,
          lr              : float = 1e-3,
          eps             : float = 1e-6,
          device                  = None,
          checkpoint_path         = "unet_checkpoint.pth",
          best_path               = "unet_best.pth",
          resume          : bool  = False,
          tb_log_dir              = "runs/unet",
          BASE                    = None,
          name_file               = None,
          NETWORK         : str   = 'U-Net'):
    """
    Entraîne le modèle U-Net ou BE_UNet.

    Parameters
    ----------
    NETWORK : 'U-Net' ou 'BE_UNet'
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entraînement sur : {device}  |  NETWORK : {NETWORK}")

    model.to(device)

    criterion = sMAPELoss(eps=eps)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    epoch_start  = 1
    best_val     = float('inf')
    train_losses = []
    val_losses   = []

    if resume:
        epoch_start, best_val, train_losses, val_losses = load_checkpoint(
            checkpoint_path, model, optimizer, scheduler, device)

    epoch_end = epoch_start + epochs - 1

    writer = SummaryWriter(log_dir=str(tb_log_dir))
    print(f"TensorBoard : tensorboard --logdir {tb_log_dir}\n")

    # Graphe du modèle
    try:
        if NETWORK == 'U-Net':
            writer.add_graph(model, torch.zeros(1, 3, 32, 32, device=device))
        elif NETWORK == 'BE_UNet':
            writer.add_graph(model, (torch.zeros(1, 1, 32, 32, device=device),
                                     torch.zeros(1, 16, device=device)))
    except Exception:
        pass

    print(f"Epochs {epoch_start} → {epoch_end}  "
          f"({'reprise' if resume else 'nouveau départ'})\n")

    for epoch in range(epoch_start, epoch_end + 1):

        # ── train ──
        model.train()
        total_train = 0.0
        for batch in train_loader:
            tensors = _batch_to_tensors(batch, device, NETWORK)
            optimizer.zero_grad()
            pred, y = _forward(model, tensors, NETWORK)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_train += loss.item()

        avg_train  = total_train / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        train_losses.append(avg_train)
        writer.add_scalar('Loss/train', avg_train,  epoch)
        writer.add_scalar('LR',         current_lr, epoch)

        # ── validation ──
        if val_loader is not None:
            model.eval()
            total_val = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    tensors    = _batch_to_tensors(batch, device, NETWORK)
                    pred, y    = _forward(model, tensors, NETWORK)
                    total_val += criterion(pred, y).item()

            avg_val = total_val / len(val_loader)
            val_losses.append(avg_val)
            writer.add_scalar('Loss/val',          avg_val,                    epoch)
            writer.add_scalars('Loss/comparaison', {'train': avg_train,
                                                    'val':   avg_val},         epoch)
            scheduler.step(avg_val)

            if avg_val < best_val:
                best_val = avg_val
                torch.save(model.state_dict(), best_path)
                writer.add_scalar('Best_val', best_val, epoch)

            print(f"Epoch {epoch:4d}/{epoch_end}  "
                  f"train={avg_train:.5f}  val={avg_val:.5f}  "
                  f"best={best_val:.5f}  lr={current_lr:.2e}")
        else:
            print(f"Epoch {epoch:4d}/{epoch_end}  "
                  f"train={avg_train:.5f}  lr={current_lr:.2e}")

        save_checkpoint(checkpoint_path, model, optimizer, scheduler,
                        epoch, best_val, train_losses, val_losses)

    writer.close()

    # ── loss curve ──
    all_epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(9, 4))
    plt.plot(all_epochs, train_losses, label='Train', color='steelblue')
    if val_losses:
        plt.plot(all_epochs, val_losses, label='Val', color='coral')
    if resume:
        plt.axvline(x=epoch_start - 1, color='gray', linestyle='--',
                    linewidth=0.8, label=f'Reprise epoch {epoch_start}')
    plt.title(f"sMAPE loss — {NETWORK}")
    plt.xlabel("Epoch")
    plt.ylabel("sMAPE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if BASE is not None and name_file is not None:
        save_dir = BASE / 'Software' / 'OT_NN' / NETWORK / 'illustrations' / name_file
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / "loss_curve.png", dpi=150)

    plt.show()
    return train_losses, val_losses
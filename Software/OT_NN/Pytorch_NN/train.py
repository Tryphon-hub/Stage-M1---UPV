# train.py  —  U-Net / BE_UNet
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


# ─── Loss ────────────────────────────────────────────────────────────────────

class sMAPELoss(nn.Module):
    """
    Symmetric Mean Absolute Percentage Error, computed pixel-wise.

        L = (1/N) * sum_i  2|σ_i - σ̃_i| / (|σ_i| + |σ̃_i| + ε)

    The symmetric form bounds each term in [0, 2] and stays finite when the
    target is zero, unlike the standard MAPE.
    """
    def __init__(self, eps: float = 1e-6):
        """
        Parameters
        ----------
        eps : float — small constant added to the denominator to avoid
            division by zero where both prediction and target vanish.
        """
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the mean symmetric APE between `pred` and `target`.

        Parameters
        ----------
        pred, target : torch.Tensor — same-shaped stress tensors.

        Returns
        -------
        torch.Tensor — scalar loss.
        """
        num   = 2.0 * (pred - target).abs()
        denom = pred.abs() + target.abs() + self.eps
        return (num / denom).mean()


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _batch_to_tensors(batch: dict, device: torch.device, NETWORK: str = 'U-Net'):
    """
    Build the input/output tensors from a dataloader batch.

    NETWORK='U-Net'   → returns (x, y)           x : [B, 3, H, W]  — ρ + tx + ty
    NETWORK='BE_UNet' → returns (rho, nodes, y)  rho : [B, 1, H, W], nodes : [B, 16]

    The target `y` stacks the three independent stress components (σx, σy, τxy);
    indices 0/1/3 of the 6-component Voigt stress are kept (2, 4, 5 are zero in
    plane stress).

    Parameters
    ----------
    batch   : dict — one batch from the DataLoader (Densities/Stress/Tractions).
    device  : torch.device — target device for the returned tensors.
    NETWORK : str — 'U-Net' or 'BE_UNet'.

    Returns
    -------
    tuple — (x, y) for U-Net, or (rho, nodes, y) for BE_UNet.
    """
    densities = batch['Densities']   # [B, 1, n_pixels]
    stress    = batch['Stress']      # [B, n_pixels, 6]
    tractions = batch['Tractions']   # [B, 1, 2, 8]

    B        = densities.shape[0]
    n_pixels = densities.shape[-1]
    img_size = int(n_pixels ** 0.5)

    # ── density → 2D map [B, 1, H, W] ──
    rho = densities.squeeze(1).reshape(B, 1, img_size, img_size)

    # ── stress → [B, 3, H, W] (Voigt components 0, 1, 3 = σx, σy, τxy) ──
    sigma_x = stress[:, :, 0].reshape(B, 1, img_size, img_size)
    sigma_y = stress[:, :, 1].reshape(B, 1, img_size, img_size)
    tau_xy  = stress[:, :, 3].reshape(B, 1, img_size, img_size)
    y = torch.cat([sigma_x, sigma_y, tau_xy], dim=1).to(device)

    if NETWORK == 'U-Net':
        # tractions → 2D maps [B, 2, H, W], concatenated with ρ as input channels
        tx_map, ty_map = _tractions_to_maps(tractions, img_size, torch.device('cpu'))
        x = torch.cat([rho, tx_map, ty_map], dim=1).to(device)   # [B, 3, H, W]
        return x, y

    elif NETWORK == 'BE_UNet':
        # nodal scalars → [B, 16], fed to the BoundaryEmbedding instead of maps
        T     = tractions.squeeze(1)            # [B, 2, 8]
        nodes = T.reshape(B, 16).to(device)     # [B, 16]
        rho   = rho.to(device)                  # [B, 1, H, W]
        return rho, nodes, y

    else:
        raise ValueError(f"Unknown NETWORK '{NETWORK}'. Must be 'U-Net' or 'BE_UNet'.")


def _tractions_to_maps(tractions: torch.Tensor, img_size: int,
                       device: torch.device) -> tuple:
    """
    Convert nodal forces (B, 1, 2, 8) into two 2D maps (B, 1, H, W) by linear
    interpolation along the 4 edges of the square domain. Corner pixels
    accumulate contributions from both adjacent edges.

    The stored rows are per-edge (Tn, Tt) = (normal, tangential): they are
    converted to global (tx, ty) and each node placed at its physical boundary
    position. Kept identical to `dataset.get_traction_distribution` so training
    and inference feed the model the same representation.

    Parameters
    ----------
    tractions : torch.Tensor [B, 1, 2, 8] — nodal (Tn, Tt) per node.
    img_size  : int — side length of the square output maps.
    device    : torch.device — target device for the returned maps.

    Returns
    -------
    tuple(torch.Tensor, torch.Tensor) — (tx_map, ty_map), each [B, 1, H, W].
    """
    import numpy as np
    from dataset import tractions_to_global, node_positions_pixels

    B = tractions.shape[0]
    T = tractions.squeeze(1).numpy()   # [B, 2, 8] = (Tn, Tt)

    points = node_positions_pixels(img_size, inclusive=False)   # (8, 2) physical

    tx_batch = np.zeros((B, img_size, img_size), dtype=np.float32)
    ty_batch = np.zeros((B, img_size, img_size), dtype=np.float32)

    for b in range(B):
        T_global = tractions_to_global(T[b])   # (2, 8) global (tx, ty)
        for k in range(0, 8, 2):
            p1 = points[k]
            p2 = points[k + 1]
            xs = np.round(np.linspace(p1[0], p2[0], img_size)).astype(int)
            ys = np.round(np.linspace(p1[1], p2[1], img_size)).astype(int)
            tx_batch[b, ys, xs] += np.linspace(T_global[0, k], T_global[0, k+1], img_size)
            ty_batch[b, ys, xs] += np.linspace(T_global[1, k], T_global[1, k+1], img_size)

    tx_map = torch.from_numpy(tx_batch).unsqueeze(1).to(device)
    ty_map = torch.from_numpy(ty_batch).unsqueeze(1).to(device)
    return tx_map, ty_map


def _forward(model, batch_tensors, NETWORK: str):
    """
    Call the model with the right positional arguments for the architecture and
    return its prediction together with the target.

    Parameters
    ----------
    model         : nn.Module — the network.
    batch_tensors : tuple — output of `_batch_to_tensors` for this NETWORK.
    NETWORK       : str — 'U-Net' (model(x)) or 'BE_UNet' (model(rho, nodes)).

    Returns
    -------
    tuple(torch.Tensor, torch.Tensor) — (prediction, target y).
    """
    if NETWORK == 'U-Net':
        x, y = batch_tensors
        return model(x), y
    elif NETWORK == 'BE_UNet':
        rho, nodes, y = batch_tensors
        return model(rho, nodes), y


# ─── Checkpoint ──────────────────────────────────────────────────────────────

def save_checkpoint(path, model, optimizer, scheduler,
                    epoch, best_val, train_losses, val_losses):
    """
    Save a full training checkpoint (model, optimizer, scheduler and history)
    so that training can later be resumed exactly where it stopped.

    Parameters
    ----------
    path         : path-like — destination file.
    model        : nn.Module — network whose state is saved.
    optimizer    : torch optimizer.
    scheduler    : torch LR scheduler.
    epoch        : int — last completed epoch.
    best_val     : float — best validation loss so far.
    train_losses, val_losses : list[float] — loss history.

    Returns
    -------
    None
    """
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
    """
    Restore model, optimizer, scheduler and loss history from a checkpoint
    saved by `save_checkpoint`, and return the state needed to resume training.

    Parameters
    ----------
    path      : path-like — checkpoint file.
    model     : nn.Module — network to load weights into (in place).
    optimizer : torch optimizer — restored in place.
    scheduler : torch LR scheduler — restored in place.
    device    : torch.device — map location for the loaded tensors.

    Returns
    -------
    tuple(int, float, list, list)
        (epoch_start, best_val, train_losses, val_losses).
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optim_state'])
    scheduler.load_state_dict(ckpt['sched_state'])

    # Resume on the epoch following the last completed one.
    epoch_start  = ckpt['epoch'] + 1
    best_val     = ckpt['best_val']
    train_losses = ckpt['train_losses']
    val_losses   = ckpt['val_losses']

    print(f"  Checkpoint loaded: epoch {ckpt['epoch']}  "
          f"best_val={best_val:.5f}  "
          f"({len(train_losses)} epochs done)")

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
          illustation_dir                    = None,
          name_file               = None,
          NETWORK         : str   = 'U-Net'):
    """
    Train a U-Net or BE_UNet model with the sMAPE loss, Adam and a
    ReduceLROnPlateau scheduler. Logs to TensorBoard, saves a checkpoint every
    epoch and the best-validation weights separately, and plots the loss curve.

    Parameters
    ----------
    model            : nn.Module — network to train.
    train_loader     : DataLoader — training data.
    val_loader       : DataLoader | None — optional validation data.
    epochs           : int — number of epochs to run this call.
    lr               : float — initial learning rate.
    eps              : float — sMAPE denominator epsilon.
    device           : torch.device | None — auto-selected (CUDA if available).
    checkpoint_path  : path-like — resumable checkpoint file.
    best_path        : path-like — best-validation weights file.
    resume           : bool — resume from `checkpoint_path`.
    tb_log_dir       : path-like — TensorBoard log directory.
    illustation_dir  : path-like | None — where to save the loss curve.
    name_file        : str | None — required (with illustation_dir) to save the plot.
    NETWORK          : str — 'U-Net' or 'BE_UNet'.

    Returns
    -------
    tuple(list[float], list[float]) — (train_losses, val_losses) histories.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}  |  NETWORK: {NETWORK}")

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

    # Log the model graph to TensorBoard (best-effort; may fail on some setups)
    try:
        if NETWORK == 'U-Net':
            writer.add_graph(model, torch.zeros(1, 3, 32, 32, device=device))
        elif NETWORK == 'BE_UNet':
            writer.add_graph(model, (torch.zeros(1, 1, 32, 32, device=device),
                                     torch.zeros(1, 16, device=device)))
    except Exception:
        pass

    print(f"Epochs {epoch_start} → {epoch_end}  "
          f"({'resumed' if resume else 'fresh start'})\n")

    for epoch in range(epoch_start, epoch_end + 1):

        # ── training pass ──
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

        # ── validation pass ──
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

    # ── loss curve over all epochs ──
    all_epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(9, 4))
    plt.plot(all_epochs, train_losses, label='Train', color='steelblue')
    if val_losses:
        plt.plot(all_epochs, val_losses, label='Val', color='coral')
    if resume:
        plt.axvline(x=epoch_start - 1, color='gray', linestyle='--',
                    linewidth=0.8, label=f'Resumed at epoch {epoch_start}')
    plt.title(f"sMAPE loss — {NETWORK}")
    plt.xlabel("Epoch")
    plt.ylabel("sMAPE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if illustation_dir is not None and name_file is not None:
        illustation_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(illustation_dir / "loss_curve.png", dpi=150)

    plt.close()
    return train_losses, val_losses
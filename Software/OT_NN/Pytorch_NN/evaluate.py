# evaluate.py  —  U-Net / BE_UNet
import torch
import numpy as np
import matplotlib.pyplot as plt
from train import sMAPELoss, _batch_to_tensors, _forward


def evaluate(model, loader, device=None, eps: float = 1e-6,
             SAVE_DIR=None, name_file=None, NETWORK: str = 'U-Net'):
    """
    Compute the mean sMAPE over a loader and the per-component MAE (σx, σy, τxy).

    Parameters
    ----------
    model     : nn.Module — trained network.
    loader    : DataLoader — evaluation data.
    device    : torch.device | None — auto-selected (CUDA if available).
    eps       : float — sMAPE denominator epsilon.
    SAVE_DIR, name_file : unused here (kept for signature symmetry with the
                          plotting functions).
    NETWORK   : str — 'U-Net' or 'BE_UNet'.

    Returns
    -------
    dict — {'smape', 'mae_sx', 'mae_sy', 'mae_txy'}.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    criterion  = sMAPELoss(eps=eps)
    components = ['σx', 'σy', 'τxy']

    total_smape = 0.0
    mae_sum     = torch.zeros(3)
    n_batches   = 0

    with torch.no_grad():
        for batch in loader:
            tensors    = _batch_to_tensors(batch, device, NETWORK)
            pred, y    = _forward(model, tensors, NETWORK)

            total_smape += criterion(pred, y).item()
            # Mean abs error per channel, averaged over batch and spatial dims.
            mae_sum     += (pred - y).abs().mean(dim=(0, 2, 3)).cpu()
            n_batches   += 1

    smape_mean = total_smape / n_batches
    mae_mean   = mae_sum / n_batches

    print("─" * 40)
    print(f"  mean sMAPE    : {smape_mean:.5f}")
    for i, name in enumerate(components):
        print(f"  MAE {name:<4}      : {mae_mean[i]:.5f}")
    print("─" * 40)

    return {
        'smape'  : smape_mean,
        'mae_sx' : mae_mean[0].item(),
        'mae_sy' : mae_mean[1].item(),
        'mae_txy': mae_mean[2].item(),
    }


def visualize(model, loader, device=None, n: int = 3,
              SAVE_DIR=None, name_file=None, NETWORK: str = 'U-Net'):
    """
    Display n examples side by side: ground truth vs prediction for σx, σy, τxy.
    Each (GT, prediction) pair shares a common color scale.

    Parameters
    ----------
    model     : nn.Module — trained network.
    loader    : DataLoader — source of the displayed batch.
    device    : torch.device | None — auto-selected (CUDA if available).
    n         : int — number of examples to show (capped by batch size).
    SAVE_DIR  : path-like | None — base directory to save the figure.
    name_file : str | None — sub-directory name; with SAVE_DIR, enables saving.
    NETWORK   : str — 'U-Net' or 'BE_UNet'.

    Returns
    -------
    None — displays (and optionally saves) a matplotlib figure.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    batch = next(iter(loader))
    tensors = _batch_to_tensors(batch, device, NETWORK)

    with torch.no_grad():
        pred, y = _forward(model, tensors, NETWORK)

    n      = min(n, y.shape[0])
    labels = ['σx', 'σy', 'τxy']
    n_cols = 2 * n

    fig, axes = plt.subplots(3, n_cols, figsize=(4 * n_cols, 12), squeeze=False)

    for comp in range(3):
        for ex in range(n):
            gt_map   = y[ex, comp].cpu().numpy()
            pred_map = pred[ex, comp].cpu().numpy()

            vmin = min(gt_map.min(), pred_map.min())
            vmax = max(gt_map.max(), pred_map.max())

            col_gt   = 2 * ex
            col_pred = 2 * ex + 1

            im_gt = axes[comp, col_gt].imshow(
                gt_map, cmap='RdBu', origin='lower', vmin=vmin, vmax=vmax)
            axes[comp, col_gt].set_title(f"GT {labels[comp]} — ex {ex+1}", fontsize=10)
            axes[comp, col_gt].axis('off')
            fig.colorbar(im_gt, ax=axes[comp, col_gt], fraction=0.046)

            im_pr = axes[comp, col_pred].imshow(
                pred_map, cmap='RdBu', origin='lower', vmin=vmin, vmax=vmax)
            axes[comp, col_pred].set_title(f"Pred {labels[comp]} — ex {ex+1}", fontsize=10)
            axes[comp, col_pred].axis('off')
            fig.colorbar(im_pr, ax=axes[comp, col_pred], fraction=0.046)

    plt.suptitle(f"GT vs Prediction — {NETWORK}", fontsize=14)
    plt.tight_layout()

    if SAVE_DIR is not None and name_file is not None:
        save_dir = SAVE_DIR / name_file
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / "visualisation_contraintes.png", dpi=150)
    plt.close()


def visualize_error(model, loader, device=None, n: int = 3,
                    SAVE_DIR=None, name_file=None, NETWORK: str = 'U-Net'):
    """
    Display the absolute error maps |GT - Pred| for σx, σy, τxy over n examples.

    Parameters
    ----------
    model     : nn.Module — trained network.
    loader    : DataLoader — source of the displayed batch.
    device    : torch.device | None — auto-selected (CUDA if available).
    n         : int — number of examples to show (capped by batch size).
    SAVE_DIR  : path-like | None — base directory to save the figure.
    name_file : str | None — sub-directory name; with SAVE_DIR, enables saving.
    NETWORK   : str — 'U-Net' or 'BE_UNet'.

    Returns
    -------
    None — displays (and optionally saves) a matplotlib figure.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    batch = next(iter(loader))
    tensors = _batch_to_tensors(batch, device, NETWORK)

    with torch.no_grad():
        pred, y = _forward(model, tensors, NETWORK)

    n      = min(n, y.shape[0])
    labels = ['σx', 'σy', 'τxy']

    fig, axes = plt.subplots(3, n, figsize=(4 * n, 10), squeeze=False)

    for comp in range(3):
        for ex in range(n):
            err = (y[ex, comp] - pred[ex, comp]).abs().cpu().numpy()
            im  = axes[comp, ex].imshow(err, cmap='hot', origin='lower')
            axes[comp, ex].set_title(f"|err| {labels[comp]} — ex {ex+1}", fontsize=10)
            axes[comp, ex].axis('off')
            fig.colorbar(im, ax=axes[comp, ex], fraction=0.046)

    plt.suptitle(f"Absolute error |GT − Pred| — {NETWORK}", fontsize=14)
    plt.tight_layout()

    if SAVE_DIR is not None and name_file is not None:
        save_dir = SAVE_DIR / name_file
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / "erreur_contraintes.png", dpi=150)
    plt.close()


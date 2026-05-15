# tune_Unet.py
# Sequential hyperparameter tuning — 10 trials × 300 epochs each.
# Each trial saves its best weights to a dedicated .pth file.
# All results are stored in the "tune_hyperparameters" folder.
# Expected runtime: 40–60 hours on a GTX 1650.

import torch
import torch.nn as nn
import time
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')   # no interactive display — saves figures to disk only
import matplotlib.pyplot as plt

from datetime            import datetime
from pathlib             import Path
from torch.utils.data    import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from dataset import Dataset_TopOpt, IterationDataset, load_mat
from train   import sMAPELoss, _batch_to_tensors, save_checkpoint


# ═══════════════════════════════════════════════════════════════════════════════
#  Global paths & settings
# ═══════════════════════════════════════════════════════════════════════════════

DATA_PATH   = Path(r"D:\Maxence\Heavy files\data\dataset.mat")
RESULTS_DIR = Path(r"D:\Maxence\Heavy files\U-net\tune_hyperparameters")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE    = RESULTS_DIR / "tuning_results.json"

EPOCHS      = 300
NUM_WORKERS = 0
SEED        = 42
EPS_SMAPE   = 1e-6
VAL_SPLIT   = 0.15

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
print(f"Results directory : {RESULTS_DIR}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  Architecture definitions
#  Two variants:
#    "proposed" — 4 levels + CBAM at bottleneck         (our proposal)
#    "article"  — 4 levels, 3 conv/block, no CBAM       (from the paper)
# ═══════════════════════════════════════════════════════════════════════════════

class DoubleConv(nn.Module):
    """2 conv per level — used in the 'proposed' architecture."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class TripleConv(nn.Module):
    """3 conv per level — used in the 'article' architecture."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, conv_block):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), conv_block(in_ch, out_ch))
    def forward(self, x): return self.block(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, conv_block):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = conv_block(in_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class ChannelAttention(nn.Module):
    def __init__(self, channels, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, channels // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // ratio, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg = self.mlp(self.avg_pool(x))
        mx  = self.mlp(self.max_pool(x))
        return self.sigmoid(avg + mx).unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg   = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1,  keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    def __init__(self, channels, ratio=8):
        super().__init__()
        self.channel_att = ChannelAttention(channels, ratio)
        self.spatial_att = SpatialAttention()
    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x


def build_model(arch: str, nif: int, n_in: int = 3, n_out: int = 3) -> nn.Module:
    """
    Build a U-Net model from architecture name and nif.

    arch : 'proposed'  — 4 levels, 2 conv/block, CBAM at bottleneck (2×2)
           'article'   — 4 levels, 3 conv/block, no CBAM             (2×2)

    With input 32×32 and 4 downsampling steps:
      Level 0 : 32×32
      Level 1 : 16×16
      Level 2 :  8×8
      Level 3 :  4×4
      Bottleneck:  2×2
    """

    conv_block = DoubleConv if arch == 'proposed' else TripleConv
    use_cbam   = (arch == 'proposed')

    class UNetVariant(nn.Module):
        def __init__(self):
            super().__init__()
            f = nif
            self.inc        = conv_block(n_in, f)
            self.down1      = Down(f,      f * 2,  conv_block)
            self.down2      = Down(f * 2,  f * 4,  conv_block)
            self.down3      = Down(f * 4,  f * 8,  conv_block)
            self.bottleneck = Down(f * 8,  f * 16, conv_block)
            self.cbam       = CBAM(f * 16) if use_cbam else nn.Identity()
            self.up1        = Up(f * 16, f * 8,  conv_block)
            self.up2        = Up(f * 8,  f * 4,  conv_block)
            self.up3        = Up(f * 4,  f * 2,  conv_block)
            self.up4        = Up(f * 2,  f,      conv_block)
            self.outc       = nn.Conv2d(f, n_out, kernel_size=1)

        def forward(self, x):
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            xb = self.cbam(self.bottleneck(x4))
            x  = self.up1(xb, x4)
            x  = self.up2(x,  x3)
            x  = self.up3(x,  x2)
            x  = self.up4(x,  x1)
            return self.outc(x)

    return UNetVariant()


# ═══════════════════════════════════════════════════════════════════════════════
#  Trial runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_trial(trial_id: int, cfg: dict, ds_iter: IterationDataset) -> dict:
    """
    Train one trial with the given hyperparameter configuration.

    Parameters
    ----------
    trial_id : int   — trial index (for logging)
    cfg      : dict  — hyperparameter configuration
    ds_iter  : IterationDataset — full iteration dataset

    Returns
    -------
    dict with trial results (best_val, train_time, config)
    """

    arch       = cfg['arch']
    nif        = cfg['nif']
    lr         = cfg['lr']
    batch_size = cfg['batch_size']
    data_frac  = cfg['data_frac']   # fraction of the dataset to use [0, 1]

    # ── Trial name ─────────────────────────────────────────────────────────
    trial_name = (f"trial{trial_id:02d}"
                  f"_arch-{arch}"
                  f"_nif{nif}"
                  f"_lr{lr:.0e}"
                  f"_bs{batch_size}"
                  f"_frac{int(data_frac*100)}pct")

    best_path  = RESULTS_DIR / f"{trial_name}_best.pth"
    ckpt_path  = RESULTS_DIR / f"{trial_name}_checkpoint.pth"
    tb_dir     = str(RESULTS_DIR / "runs" / trial_name)
    fig_path   = RESULTS_DIR / f"{trial_name}_loss.png"

    print(f"\n{'═'*70}")
    print(f"  Trial {trial_id:02d} / {N_TRIALS}  —  {trial_name}")
    print(f"  arch={arch}  nif={nif}  lr={lr}  batch_size={batch_size}  data_frac={data_frac}")
    print(f"{'═'*70}\n")

    # ── Dataset subset ─────────────────────────────────────────────────────
    n_total  = len(ds_iter)
    n_subset = max(1, int(n_total * data_frac))
    indices  = torch.randperm(n_total, generator=torch.Generator().manual_seed(SEED))[:n_subset]
    subset   = torch.utils.data.Subset(ds_iter, indices.tolist())

    n_val    = max(1, int(n_subset * VAL_SPLIT))
    n_train  = n_subset - n_val
    train_ds, val_ds = random_split(
        subset, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    print(f"  Dataset : {n_train} train / {n_val} val  "
          f"({len(train_loader)} / {len(val_loader)} batches)")

    # ── Model ──────────────────────────────────────────────────────────────
    model     = build_model(arch=arch, nif=nif).to(device)
    n_params  = sum(p.numel() for p in model.parameters())
    print(f"  Parameters : {n_params:,}\n")

    # ── Training objects ───────────────────────────────────────────────────
    criterion = sMAPELoss(eps=EPS_SMAPE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    writer = SummaryWriter(log_dir=tb_dir)

    train_losses = []
    val_losses   = []
    best_val     = float('inf')
    t_start      = time.time()

    # ── Training loop ──────────────────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):

        # train
        model.train()
        total_train = 0.0
        for batch in train_loader:
            x, y = _batch_to_tensors(batch, device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_train += loss.item()

        avg_train  = total_train / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        train_losses.append(avg_train)
        writer.add_scalar('Loss/train', avg_train,  epoch)
        writer.add_scalar('LR',         current_lr, epoch)

        # validation
        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x, y  = _batch_to_tensors(batch, device)
                pred  = model(x)
                total_val += criterion(pred, y).item()

        avg_val = total_val / len(val_loader)
        val_losses.append(avg_val)
        writer.add_scalar('Loss/val', avg_val, epoch)
        writer.add_scalars('Loss/compare', {'train': avg_train, 'val': avg_val}, epoch)

        scheduler.step(avg_val)

        # save best weights
        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), best_path)
            writer.add_scalar('Best_val', best_val, epoch)

        # save checkpoint every 50 epochs
        if epoch % 50 == 0:
            save_checkpoint(str(ckpt_path), model, optimizer, scheduler,
                            epoch, best_val, train_losses, val_losses)

        print(f"  Epoch {epoch:3d}/{EPOCHS}  "
              f"train={avg_train:.5f}  val={avg_val:.5f}  "
              f"best={best_val:.5f}  lr={current_lr:.2e}")

    writer.close()

    elapsed = time.time() - t_start
    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)
    s = int(elapsed % 60)
    print(f"\n  Trial {trial_id:02d} done — {h:02d}h {m:02d}m {s:02d}s  |  best_val={best_val:.5f}")

    # ── Loss curve ─────────────────────────────────────────────────────────
    plt.figure(figsize=(9, 4))
    plt.plot(range(1, EPOCHS + 1), train_losses, label='Train', color='steelblue')
    plt.plot(range(1, EPOCHS + 1), val_losses,   label='Val',   color='coral')
    plt.title(f"sMAPE — {trial_name}")
    plt.xlabel("Epoch")
    plt.ylabel("sMAPE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=120)
    plt.close()

    return {
        'trial_id'    : trial_id,
        'trial_name'  : trial_name,
        'config'      : cfg,
        'best_val'    : best_val,
        'n_params'    : n_params,
        'train_time_s': int(elapsed),
        'best_path'   : str(best_path),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Hyperparameter grid
#  10 trials designed to explore orthogonal axes:
#   - architecture  : proposed (2conv+CBAM) vs article (3conv, no CBAM)
#   - nif           : 16 / 32 / 64
#   - learning rate : 1e-3 / 5e-4 / 1e-4
#   - batch size    : 16 / 32 / 64
#   - data fraction : 50% / 75% / 100%
# ═══════════════════════════════════════════════════════════════════════════════

TRIALS = [
    # trial 01 — baseline (our default config)
    {'arch': 'proposed', 'nif': 32, 'lr': 1e-3, 'batch_size': 32, 'data_frac': 1.00},

    # trial 02 — article architecture, same other hyperparams
    {'arch': 'article',  'nif': 32, 'lr': 1e-3, 'batch_size': 32, 'data_frac': 1.00},

    # trial 03 — smaller model (nif=16), proposed arch
    {'arch': 'proposed', 'nif': 16, 'lr': 1e-3, 'batch_size': 32, 'data_frac': 1.00},

    # trial 04 — larger model (nif=48, bs=16 to stay within 4GB VRAM)
    {'arch': 'proposed', 'nif': 48, 'lr': 1e-3, 'batch_size': 16, 'data_frac': 1.00},

    # trial 05 — smaller model (nif=16), article arch
    {'arch': 'article',  'nif': 16, 'lr': 1e-3, 'batch_size': 32, 'data_frac': 1.00},

    # trial 06 — lower learning rate
    {'arch': 'proposed', 'nif': 32, 'lr': 5e-4, 'batch_size': 32, 'data_frac': 1.00},

    # trial 07 — much lower learning rate
    {'arch': 'proposed', 'nif': 32, 'lr': 1e-4, 'batch_size': 32, 'data_frac': 1.00},

    # trial 08 — smaller batch size (more gradient updates per epoch)
    {'arch': 'proposed', 'nif': 32, 'lr': 1e-3, 'batch_size': 16, 'data_frac': 1.00},

    # trial 09 — larger batch size (capped at 48 to stay within 4GB VRAM)
    {'arch': 'proposed', 'nif': 32, 'lr': 1e-3, 'batch_size': 48, 'data_frac': 1.00},

    # trial 10 — reduced dataset (50%), check overfitting sensitivity
    {'arch': 'proposed', 'nif': 32, 'lr': 1e-3, 'batch_size': 32, 'data_frac': 0.50},
]

N_TRIALS = len(TRIALS)


# ═══════════════════════════════════════════════════════════════════════════════
#  Load dataset once — shared across all trials
# ═══════════════════════════════════════════════════════════════════════════════

print("Loading dataset...")
data_raw = load_mat(DATA_PATH)
ds_base  = Dataset_TopOpt(data_raw)
ds_iter  = IterationDataset(ds_base)

print(f"  Traction distributions : {len(ds_base)}")
print(f"  Total iterations       : {len(ds_iter)}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  Run all trials
# ═══════════════════════════════════════════════════════════════════════════════

all_results = []
global_start = time.time()

for trial_id, cfg in enumerate(TRIALS, start=1):
    try:
        result = run_trial(trial_id, cfg, ds_iter)

    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            torch.cuda.empty_cache()
            print()
            print(f"  ⚠️  Trial {trial_id} — CUDA out of memory, skipping.")
            print(f"  Config : {cfg}")
            print(f"  Suggestion : reduce batch_size or nif for this trial.")
            print()
            result = {
                'trial_id'    : trial_id,
                'trial_name'  : f"trial{trial_id:02d}_OOM",
                'config'      : cfg,
                'best_val'    : float('inf'),
                'n_params'    : 0,
                'train_time_s': 0,
                'best_path'   : 'OOM — skipped',
                'status'      : 'OOM',
            }
        else:
            with open(LOG_FILE, 'w') as f:
                json.dump(all_results, f, indent=2)
            raise e

    all_results.append(result)

    # Save results after each trial — safe if the run is interrupted mid-way
    with open(LOG_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"  Results saved → {LOG_FILE}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Summary
# ═══════════════════════════════════════════════════════════════════════════════

total_elapsed = time.time() - global_start
th = int(total_elapsed // 3600)
tm = int((total_elapsed % 3600) // 60)

print(f"\n{'═'*70}")
print(f"  All {N_TRIALS} trials completed in {th:02d}h {tm:02d}m")
print(f"{'═'*70}\n")

# Sort by best_val ascending
all_results.sort(key=lambda r: r['best_val'])

print(f"  {'Rank':<5} {'Trial':<55} {'best_val':>10} {'params':>12} {'time':>8}")
print(f"  {'-'*95}")
for rank, r in enumerate(all_results, start=1):
    h = r['train_time_s'] // 3600
    m = (r['train_time_s'] % 3600) // 60
    print(f"  {rank:<5} {r['trial_name']:<55} {r['best_val']:>10.5f} "
          f"{r['n_params']:>12,} {h:02d}h{m:02d}m")

print(f"\n  Best model : {all_results[0]['trial_name']}")
print(f"  Best val   : {all_results[0]['best_val']:.5f}")
print(f"  Weights    : {all_results[0]['best_path']}")

# ── Final comparison plot ──────────────────────────────────────────────────────
# Bar chart of best_val per trial
names     = [r['trial_name'].replace('_', '\n') for r in all_results]
best_vals = [r['best_val'] for r in all_results]
colors    = ['steelblue' if r['config']['arch'] == 'proposed' else 'coral'
             for r in all_results]

fig, ax = plt.subplots(figsize=(16, 5))
bars = ax.bar(range(len(all_results)), best_vals, color=colors, edgecolor='black', linewidth=0.5)
ax.set_xticks(range(len(all_results)))
ax.set_xticklabels([r['trial_name'] for r in all_results],
                   rotation=30, ha='right', fontsize=7)
ax.set_ylabel("Best val sMAPE")
ax.set_title("Hyperparameter tuning — best validation sMAPE per trial (lower is better)")
ax.grid(axis='y', alpha=0.4)

from matplotlib.patches import Patch
ax.legend(handles=[Patch(color='steelblue', label='proposed (2conv+CBAM)'),
                   Patch(color='coral',     label='article  (3conv, no CBAM)')],
          fontsize=9)

for bar, val in zip(bars, best_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
            f"{val:.4f}", ha='center', va='bottom', fontsize=7)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "tuning_summary.png", dpi=150)
plt.close()
print(f"\n  Summary plot saved → {RESULTS_DIR / 'tuning_summary.png'}")

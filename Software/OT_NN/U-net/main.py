#%% main.py
import torch
import time
import scipy.io
from datetime import datetime
from pathlib import Path

from dataset  import *
from model    import UNetTopo
from train    import train
from evaluate import evaluate, visualize, visualize_error

# ═══════════════════════════════════════════════════════════════════════════════
#%%  Configuration
# ═══════════════════════════════════════════════════════════════════════════════

user = 'laptop'  # 'laptop' ou 'server'

name_file = 'dataset_macro'


if user == 'laptop':
    BASE = Path(r'C:\Users\maxen\Documents\Stage')
elif user == 'server':
    BASE = Path(r'D:\Maxence\Stage-M1---UPV')

DATA_PATH       = BASE / 'HeavyFiles' / 'data' / (name_file+'.mat')
RESULTS_DIR     = BASE / 'HeavyFiles' / 'U-net' / 'results'
CHECKPOINT_PATH = RESULTS_DIR / ("unet_" + name_file + "_checkpoint.pth")
BEST_PATH       = RESULTS_DIR / ("unet_" + name_file + "_best.pth")
TB_LOG_DIR      = RESULTS_DIR / ("runs_" + name_file + "_") / ("unet_" + name_file)

BATCH_SIZE  = 16
VAL_SPLIT   = 0.15
NUM_WORKERS = 0

NIF         = 32
USE_CBAM    = True

LR          = 1e-3
EPS_SMAPE   = 1e-6

RESUME = True
EPOCHS = 0

#   Premier lancement   →  RESUME = False  /  EPOCHS = 50
#   Reprendre           →  RESUME = True   /  EPOCHS = nombre d'epochs à AJOUTER
#
#   Exemple : après 50 epochs, si la convergence n'est pas atteinte :
#       RESUME = True
#       EPOCHS = 500       ← 500 epochs supplémentaires
#

# ═══════════════════════════════════════════════════════════════════════════════
#%%  Device
# ═══════════════════════════════════════════════════════════════════════════════

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")

# ═══════════════════════════════════════════════════════════════════════════════
#  1. Données
# ═══════════════════════════════════════════════════════════════════════════════

print("\nChargement du dataset...")
data    = load_mat(DATA_PATH)
ds_base = Dataset_TopOpt(data)
ds_iter = IterationDataset(ds_base)

print(f"  Distributions de forces : {len(ds_base)}")
print(f"  Itérations totales      : {len(ds_iter)}")

n_val   = int(len(ds_iter) * VAL_SPLIT)
n_train = len(ds_iter) - n_val
train_ds, val_ds = torch.utils.data.random_split(
    ds_iter, [n_train, n_val],
    generator=torch.Generator().manual_seed(42)   # split identique à chaque run
)

train_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
val_loader   = torch.utils.data.DataLoader(
    val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

print(f"  Train : {n_train} samples  ({len(train_loader)} batches)")
print(f"  Val   : {n_val}   samples  ({len(val_loader)} batches)")

# ═══════════════════════════════════════════════════════════════════════════════
#%%  2. Modèle
# ═══════════════════════════════════════════════════════════════════════════════

model = UNetTopo(nif=NIF, n_in=3, n_out=3, use_cbam=USE_CBAM).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"\nModèle : UNetTopo(nif={NIF}, cbam={USE_CBAM})")
print(f"Paramètres : {n_params:,}")

if RESUME:
    print(f"\nMode reprise — chargement de {CHECKPOINT_PATH}")
    print(f"Epochs supplémentaires : {EPOCHS}")
else:
    print(f"\nNouveau départ — {EPOCHS} epochs")

# ═══════════════════════════════════════════════════════════════════════════════
#%%  3. Entraînement
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\nDébut : {datetime.now().strftime('%H:%M:%S')}")
start = time.time()

train_losses, val_losses = train(
    model           = model,
    train_loader    = train_loader,
    val_loader      = val_loader,
    epochs          = EPOCHS,
    lr              = LR,
    eps             = EPS_SMAPE,
    device          = device,
    checkpoint_path = CHECKPOINT_PATH,
    best_path       = BEST_PATH,
    resume          = RESUME,
    tb_log_dir      = TB_LOG_DIR,
    BASE            = BASE,
    name_file       = name_file,
)

elapsed = time.time() - start
h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
print(f"\nFin   : {datetime.now().strftime('%H:%M:%S')}")
print(f"Durée : {h:02d}h {m:02d}m {s:02d}s")

# ═══════════════════════════════════════════════════════════════════════════════
#%%  4. Évaluation sur le meilleur modèle
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\nChargement du meilleur modèle ({BEST_PATH})...")
model.load_state_dict(torch.load(BEST_PATH, map_location=device))

print("\n── Métriques sur le jeu de validation ──")
evaluate(model, val_loader, device=device, eps=EPS_SMAPE)

# ═══════════════════════════════════════════════════════════════════════════════
#%%  5. Visualisation
# ═══════════════════════════════════════════════════════════════════════════════

visualize(model, val_loader, device=device, n=3, BASE=BASE, name_file=name_file)
visualize_error(model, val_loader, device=device, n=3, BASE=BASE, name_file=name_file)
# %%

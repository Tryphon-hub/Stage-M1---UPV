#%% training_benchmark.py
# Train one model per configuration of `list_benchmark` and store the
# evaluation metrics in a CSV. This is a mix of:
#   - main.py            : loads the data, builds a model, trains and evaluates it
#   - TopOpt_benchmark.py: loops over list_benchmark and writes a results CSV
#
# Only the training-relevant columns of list_benchmark are used here
# (Model, NIF, N_conv, use cbam, use augmentation, probability of augmentation,
# dataset portion); Strategy and First step concern the hybrid topology
# optimization and are kept only to stay aligned with the list_benchmark format.

import csv
import time
from datetime import datetime
from pathlib import Path

import torch

from dataset        import *
from model          import *
from train          import *
from evaluate       import *
from topology_utils import *

# ═══════════════════════════════════════════════════════════════════════════════
#%%  Configuration
# ═══════════════════════════════════════════════════════════════════════════════

name_file = 'dataset_1k'

BASE      = Path(__file__).parents[3]
DATA_PATH = BASE / 'HeavyFiles' / 'data' / (name_file + '.mat')

# Training hyperparameters shared by every configuration
VAL_SPLIT   = 0.15
NUM_WORKERS = 0    # parallelisation
LR          = 1e-3
EPS_SMAPE   = 1e-6
EPOCHS      = 150
RESUME      = False

# Embedding hyperparameters (BE_UNet only)
HIDDEN_LAYERS_MLP = [32, 64]
EMBED_OUT         = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ═══════════════════════════════════════════════════════════════════════════════
#%%  Benchmark definition
# ═══════════════════════════════════════════════════════════════════════════════

# [Strategy, Model, First step, NIF, N_conv, use cbam, use augmentation, probability of augmentation, dataset portion, batch size]
list_benchmark = [
    # ['Only UNet', 'U-Net', 'UNet', 16, 2, False, False, 0.2, 0.5, 16],
    # ['Only UNet', 'U-Net', 'UNet', 32, 2, False, False, 0.2, 1  , 16],
    # ['Only UNet', 'U-Net', 'UNet', 32, 2, False, False, 0.2, 1  ,  8],
    # ['Only UNet', 'U-Net', 'UNet', 32, 2, False, False, 0.2, 1  , 32],
    # ['Only UNet', 'U-Net', 'UNet', 32, 2, False, False, 0.2, 0.5, 16],
    ['Only UNet','U-Net'  , 'UNet', 32, 2, False, True , 0.5, 1  , 16],
    ['Only UNet','BE_UNet', 'UNet', 32, 2, False, False, 0.2, 1  , 16],
    ['Only UNet','BE_UNet', 'UNet', 32, 2, False, False, 0.2, 0.5, 16],
    ['Only UNet','BE_UNet', 'UNet', 32, 2, True, False , 0.2, 1  , 16],
    ['Only UNet','BE_UNet', 'UNet', 32, 2, False, True , 0.2, 1  , 16],
    # ['Only UNet', 'U-Net', 'UNet', 32, 3, False, False, 0.2, 1  , 16],
    # ['Only UNet', 'U-Net', 'UNet', 32, 2, True , False, 0.2, 1  , 16],
    # ['Only UNet', 'U-Net', 'UNet', 32, 2, False, True , 0.2, 1  , 16],
    # ['Only UNet', 'U-Net', 'UNet', 32, 3, True , True , 0.2, 1  , 16],
    # ['Only UNet', 'U-Net', 'UNet', 32, 2, True , True , 0.2, 0.5, 16],
]

# Column layout of each list_benchmark entry, reused for the CSV header.
CONFIG_COLUMNS = ['Strategy', 'Model', 'First step', 'NIF', 'N_conv',
                  'use cbam', 'use augmentation', 'probability of augmentation',
                  'dataset portion', 'batch size']

# Metrics computed for each trained configuration.
RESULT_COLUMNS = ['n_params', 'final_train_loss', 'final_val_loss',
                  'sMAPE', 'MAE_sx', 'MAE_sy', 'MAE_txy', 'training_time_s']

RESET_BENCHMARK = False            # overwrite the CSV (False → append)
TYPE_WRITE      = 'w' if RESET_BENCHMARK else 'a'

name_benchmark_file = 'benchmark_training_loss.csv'
RESULTS_ROOT        = BASE / 'Software' / 'OT_NN' / 'Pytorch_NN' / 'results'

# ═══════════════════════════════════════════════════════════════════════════════
#%%  Load the dataset once (re-split per configuration inside the loop)
# ═══════════════════════════════════════════════════════════════════════════════

print("\nLoading dataset...")
data    = load_mat(DATA_PATH)
ds_base = Dataset_TopOpt(data)
ds_iter = IterationDataset(ds_base)
print(f"  Force distributions : {len(ds_base)}")
print(f"  Total iterations    : {len(ds_iter)}")

# ═══════════════════════════════════════════════════════════════════════════════
#%%  Run benchmark
# ═══════════════════════════════════════════════════════════════════════════════

RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

with open(RESULTS_ROOT / name_benchmark_file, TYPE_WRITE, newline='') as benchmark_file:

    writer = csv.writer(benchmark_file)
    if RESET_BENCHMARK:
        writer.writerow(CONFIG_COLUMNS + RESULT_COLUMNS)

    for STRATEGY, NETWORK, FIRST_STEP, NIF, N_CONV, USE_CBAM, USE_AUGMENTATION, AUGMENTATION_P, PORTION_DATA, BATCH_SIZE in list_benchmark:

        print("\n" + "═" * 79)
        print(f"Config: {NETWORK} | NIF={NIF} | N_conv={N_CONV} | cbam={USE_CBAM} | "
              f"aug={USE_AUGMENTATION} | portion={int(PORTION_DATA*100)}% | bs={BATCH_SIZE}")
        print("═" * 79)

        # ── Output directories for this configuration (mirrors main.py) ──
        if NETWORK == 'U-Net':
            tag = f'{name_file}_NIF={NIF}_{N_CONV}_conv_CBAM={USE_CBAM}_aug={USE_AUGMENTATION if not USE_AUGMENTATION else int(100*AUGMENTATION_P)}%_portion={int(PORTION_DATA*100)}_batch={BATCH_SIZE}'
        else:
            tag = f'{name_file}_NIF={NIF}_{N_CONV}_conv_{HIDDEN_LAYERS_MLP}_CBAM={USE_CBAM}_aug={USE_AUGMENTATION if not USE_AUGMENTATION else int(100*AUGMENTATION_P)}%_portion={int(PORTION_DATA*100)}%_batch={BATCH_SIZE}'

        RESULTS_DIR       = RESULTS_ROOT / NETWORK / tag
        ILLUSTRATIONS_DIR = BASE / 'Software' / 'OT_NN' / 'Pytorch_NN' / 'illustrations' / NETWORK / tag
        CHECKPOINT_PATH   = RESULTS_DIR / ('unet_' + name_file + '_checkpoint.pth')
        BEST_PATH         = RESULTS_DIR / ('unet_' + name_file + '_best.pth')
        TB_LOG_DIR        = RESULTS_DIR / ('runs_' + name_file) / ('unet_' + name_file)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        # ── Data split (uses only PORTION_DATA of the iterations) ──
        n_val   = int(PORTION_DATA * len(ds_iter) * VAL_SPLIT)
        n_train = int(PORTION_DATA * len(ds_iter)) - n_val
        n_unused = len(ds_iter) - n_train - n_val

        train_ds, val_ds, _ = torch.utils.data.random_split(
            ds_iter, [n_train, n_val, n_unused],
            generator=torch.Generator().manual_seed(42)
        )

        # Augmentation is applied to the train split only.
        train_ds = AugmentedIterationDataset(
            train_ds, p=AUGMENTATION_P,
            rotation_90=rotation_90, symmetry_x=symmetry_x, symmetry_y=symmetry_y,
            enabled=USE_AUGMENTATION,
        )

        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        print(f"  Train: {n_train} samples  |  Val: {n_val} samples")

        # ── Build the model (mirrors main.py) ──
        if NETWORK == 'U-Net':
            N_in  = 3
            model = UNetTopo(
                nif=NIF, n_in=N_in, n_out=3,
                use_cbam=USE_CBAM, N_conv=N_CONV,
            ).to(device)

        elif NETWORK == 'BE_UNet':
            N_in  = 1
            model = BE_UNetTopo(
                nif=NIF, n_in=N_in, n_out=3,
                use_cbam=USE_CBAM,
                hidden_layers_MLP=HIDDEN_LAYERS_MLP,
                embed_out=EMBED_OUT,
                N_conv=N_CONV,
            ).to(device)

        else:
            raise ValueError(f"Unknown NETWORK '{NETWORK}'. Choose 'U-Net' or 'BE_UNet'.")

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        # ── Train ──
        print(f"  Start: {datetime.now().strftime('%H:%M:%S')}")
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
            illustation_dir = ILLUSTRATIONS_DIR,
            name_file       = name_file,
            NETWORK         = NETWORK,
        )

        training_time = time.time() - start

        # ── Evaluate the best checkpoint on the validation set ──
        model.load_state_dict(torch.load(BEST_PATH, map_location=device))
        metrics = evaluate(model, val_loader, device=device, eps=EPS_SMAPE, NETWORK=NETWORK)

        # ── Store the results ──
        writer.writerow([
            STRATEGY, NETWORK, FIRST_STEP, NIF, N_CONV,
            USE_CBAM, USE_AUGMENTATION, AUGMENTATION_P, PORTION_DATA, BATCH_SIZE,
            n_params,
            train_losses[-1] if train_losses else float('nan'),
            val_losses[-1]   if val_losses   else float('nan'),
            metrics['smape'], metrics['mae_sx'], metrics['mae_sy'], metrics['mae_txy'],
            training_time,
        ])
        benchmark_file.flush()   # persist after each config in case of interruption

        print(f"  Done in {training_time:.1f}s  |  sMAPE={metrics['smape']:.5f}")

print(f"\nBenchmark finished. Results written to {RESULTS_ROOT / name_benchmark_file}")

# %%

#%% TopOpt_benchmark_seed.py
# Seed-noise assessment for one or several fixed configurations.
#
# For every configuration in CONFIGS, trains N_SEEDS identical U-Net models
# that differ ONLY by their random seed, then runs the hybrid
# topology-optimization benchmark for each of them and appends every
# per-sample result (tagged with its full config + seed) to the shared
# benchmark_seed.csv. Aggregating the per-seed means finally gives the
# seed-to-seed variability, i.e. the reference error bar below which two
# configurations of the main architecture benchmark cannot be told apart.
# All configurations run back-to-back automatically.
#
# The seed governs three sources of randomness at once:
#   - weight initialization (Kaiming uniform, PyTorch default)
#   - DataLoader shuffling  (torch global RNG)
#   - the train/val split    (split_generator derived from the seed)
# so what is measured here is the *total* retraining noise, not the init alone.
#
# Structure (mirrors training_benchmark.py + TopOpt_benchmark_model.py):
#   Phase 1  RUN_TRAINING  : train one model per seed          (no MATLAB)
#   Phase 2  RUN_BENCHMARK : hybrid TopOpt benchmark per seed  (needs MATLAB)
#   Phase 3  read/aggregate: per-seed means + seed-to-seed std

#%% Import libraries
import sys
import time
import csv
import random
from datetime import datetime
from pathlib import Path

import matlab.engine
import numpy as np
import pandas as pd
import torch

from model          import *
from dataset        import *
from train          import *
from evaluate       import *
from topology_utils import *

#%% User settings
RUN_TRAINING  = False   # Phase 1 — train one model per (config, seed)
RUN_BENCHMARK = False   # Phase 2 — evaluate each model with the TopOpt benchmark
RESET_BENCHMARK = False # overwrite BENCHMARK_CSV once at the start of the run (False -> append)

# If a config's checkpoint already exists, skip retraining it. Useful when
# several configs share the same model-relevant fields (NIF, N_conv, cbam,
# augmentation, portion, batch size) and only differ by Strategy/First step,
# which only affect Phase 2.
SKIP_EXISTING_TRAINING = True


#%% Reproducibility helper

def set_seed(seed):
    """Fix all random generators for full reproducibility."""
    random.seed(seed)                      # data augmentation (np.random.choice)
    np.random.seed(seed)                   # data augmentation (np.random.randint)
    torch.manual_seed(seed)                # poids + shuffle DataLoader
    torch.cuda.manual_seed_all(seed)       # GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


#%% Paths & constants

name_file = 'dataset_1k'    # training set
name_data = 'dataset_test'  # benchmark evaluation set

BASE         = Path(__file__).parents[3]
RESULTS_ROOT = BASE / 'Software' / 'OT_NN' / 'Pytorch_NN' / 'results'
GMSH_EXE     = BASE.parent / 'gmsh' / 'gmsh.exe'

TRAIN_PATH = BASE / 'HeavyFiles' / 'data' / (name_file + '.mat')
DATA_PATH  = BASE / 'HeavyFiles' / 'data' / (name_data + '.mat')

sys.path.append(str(BASE / 'Software' / 'OT_NN' / 'Pytorch_NN'))
sys.path.append(str(BASE / 'Software' / 'OT_Functions'))
sys.path.append(str(BASE / 'Software' / 'OT_Software'))

# All models and the benchmark CSV live here.
SEED_BENCH_DIR = RESULTS_ROOT / 'U-Net' / 'seed_benchmark'
SEED_BENCH_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 32
PENAL    = 3
RMIN     = 1.5
NGPpL    = 2
NGPpS    = 9
E        = 1000
NU       = 0.3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


#%% Seed-benchmark definition

# The configurations under test (each fixed across every seed):
# [Strategy, Model, First step, NIF, N_conv, use cbam, use augmentation,
#  probability of augmentation, dataset portion, batch size]
CONFIGS = [
    # ['Only UNet', 'U-Net', 'UNet', 32, 2, False, False,0.2, 0.2, 16],
    ['Only UNet', 'U-Net', 'UNet', 32, 2, False, False, 0.2, 0.2, 16],
    ['Only UNet', 'U-Net', 'UNet', 32, 2, False, False, 0.2, 0.5, 16],
    ['Only UNet', 'U-Net', 'UNet', 32, 2, False, False, 0.2,  1 , 16],
    # Add more configs here, e.g.:
    # ['Hybrid',    'U-Net', 'UNet', 32, 2, False, True, 0.2, 1, 16],
    # ['Only UNet', 'U-Net', 'UNet', 16, 2, False, True, 0.2, 1, 16],
]

SEEDS = list(range(0,20))   # 10 seeds: 0..9

# Training hyperparameters (shared by every seed) — from training_benchmark.py
VAL_SPLIT   = 0.15
NUM_WORKERS = 0
LR          = 1e-3
EPS_SMAPE   = 1e-6
EPOCHS      = 150
RESUME      = False
N_in        = 3   # U-Net: rho + 2 traction channels

# Benchmark (hybrid TopOpt) settings
SIZE_LOOP = 100   # number of force distributions evaluated per seed

CONFIG_COLUMNS = ['Strategy', 'Model', 'First step', 'NIF', 'N_conv',
                  'use cbam', 'use augmentation', 'probability of augmentation',
                  'dataset portion', 'batch size']
SEED_COLUMN    = ['seed']
RESULT_COLUMNS = ['Input ID',
                  'Number of FEM iterations for the full-FEM strategy',
                  'Number of FEM iterations for the hybrid strategy',
                  'Ratio of FEM iterations (Hybrid / full-FEM)',
                  'Relative compliance error']

# Single CSV shared by every config (each row already carries its full config
# via CONFIG_COLUMNS, so all configs/seeds can be appended to the same file).
name_benchmark_file = 'benchmark_seed.csv'
BENCHMARK_CSV        = SEED_BENCH_DIR / name_benchmark_file




def model_tag(config, seed):
    """Model tag for a given config+seed (only the training-relevant fields)."""
    (_STRATEGY, _NETWORK, _FIRST_STEP, NIF, N_CONV, USE_CBAM,
     USE_AUGMENTATION, AUGMENTATION_P, PORTION_DATA, BATCH_SIZE) = config
    aug_tag = f'{int(100*AUGMENTATION_P)}%' if USE_AUGMENTATION else 'False'
    return (f'{name_file}_normalised_NIF={NIF}_{N_CONV}_conv_'
            f'CBAM={USE_CBAM}_aug={aug_tag}_portion={int(PORTION_DATA*100)}%_'
            f'batch={BATCH_SIZE}_seed={seed}')


def config_label(config):
    """Short label identifying a full config, used for logs and plot filenames."""
    (STRATEGY, _NETWORK, FIRST_STEP, NIF, N_CONV, USE_CBAM,
     USE_AUGMENTATION, AUGMENTATION_P, PORTION_DATA, BATCH_SIZE) = config
    aug_tag = f'{int(100*AUGMENTATION_P)}%' if USE_AUGMENTATION else 'False'
    return (f'NIF={NIF}_{N_CONV}conv_CBAM={USE_CBAM}_aug={aug_tag}_'
            f'portion={int(PORTION_DATA*100)}%_batch={BATCH_SIZE}_'
            f'strategy={STRATEGY.replace(" ", "")}_first={FIRST_STEP}')


#%% Phase 1 — train one model per (config, seed)

if RUN_TRAINING:
    print("\nLoading training dataset...")
    train_data = load_mat(TRAIN_PATH)
    ds_train   = Dataset_TopOpt(train_data)
    ds_train   = ds_train.normalize_dataset()   # per-sample normalisation
    ds_iter    = IterationDataset(ds_train)
    print(f"  Force distributions : {len(ds_train)}")
    print(f"  Total iterations    : {len(ds_iter)}")

    for CONFIG in CONFIGS:
        (STRATEGY, NETWORK, FIRST_STEP, NIF, N_CONV, USE_CBAM,
         USE_AUGMENTATION, AUGMENTATION_P, PORTION_DATA, BATCH_SIZE) = CONFIG

        for SEED in SEEDS:
            print("\n" + "=" * 79)
            print(f"TRAIN  config={config_label(CONFIG)}  seed={SEED}  |  "
                  f"{NETWORK} NIF={NIF} N_conv={N_CONV} "
                  f"cbam={USE_CBAM} aug={USE_AUGMENTATION} bs={BATCH_SIZE}")
            print("=" * 79)

            tag             = model_tag(CONFIG, SEED)
            RESULTS_DIR     = SEED_BENCH_DIR / tag
            ILLUSTRATIONS_DIR = BASE / 'Software' / 'OT_NN' / 'Pytorch_NN' / 'illustrations' / 'U-Net' / 'seed_benchmark' / tag
            CHECKPOINT_PATH = RESULTS_DIR / ('unet_' + name_file + '_checkpoint.pth')
            BEST_PATH       = RESULTS_DIR / ('unet_' + name_file + '_best.pth')
            TB_LOG_DIR      = RESULTS_DIR / 'tb'
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)

            if SKIP_EXISTING_TRAINING and BEST_PATH.exists():
                print(f"  Skipping (already trained) -> {BEST_PATH}")
                continue

            # Fix every RNG before the split, the model init and the training loop.
            set_seed(SEED)

            # Split — separate generator, but derived from the same seed so the
            # train/val partition is reproducible and varies with the seed.
            n_total  = int(PORTION_DATA * len(ds_iter))
            n_val    = int(n_total * VAL_SPLIT)
            n_train  = n_total - n_val
            n_unused = len(ds_iter) - n_train - n_val

            split_generator = torch.Generator().manual_seed(SEED)
            train_ds, val_ds, _ = torch.utils.data.random_split(
                ds_iter, [n_train, n_val, n_unused],
                generator=split_generator,
            )

            # Augmentation applies to the train split only.
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

            model = UNetTopo(
                nif=NIF, n_in=N_in, n_out=3,
                use_cbam=USE_CBAM, N_conv=N_CONV,
            ).to(device)
            print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

            print(f"  Start: {datetime.now().strftime('%H:%M:%S')}")
            start = time.time()
            train(
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
            print(f"  Done in {time.time() - start:.1f}s  ->  {BEST_PATH}")


#%% Phase 2 — hybrid TopOpt benchmark for each seed

if RUN_BENCHMARK:
    # ── Load & prepare the evaluation dataset (dataset_test) ──
    print("\nLoading benchmark dataset...")
    data      = load_mat(DATA_PATH)
    ds_base   = Dataset_TopOpt(data)
    ds_base   = ds_base.normalize_dataset()      # same normalisation as training
    ds_filtre = ds_base.filtre_dataset(rho_min=0.15, rho_max=0.85)
    IterData_FEM = IterationDataset(ds_filtre)

    # ── Start MATLAB engine and (re)build the mesh ──
    print("Starting MATLAB engine...")
    eng = matlab.engine.start_matlab()
    eng.addpath(str(BASE / 'Software' / 'OT_Functions'))
    eng.addpath(str(BASE / 'Software' / 'OT_Software'))

    geo_file  = BASE / 'Software' / 'OT_Software' / 'Square.geo'
    mesh_file = BASE / 'Software' / 'OT_Software' / 'Square.msh'
    eng.workspace['GmshExe']     = str(GMSH_EXE)
    eng.workspace['GeoFileName'] = str(geo_file)
    eng.workspace['Mesh_File']   = str(mesh_file)
    eng.eval(rf"""
if isfile(Mesh_File)
    delete(Mesh_File)
end
CallString = ['"' GmshExe '" "' GeoFileName '" -setnumber numLayers {IMG_SIZE} -o "' Mesh_File '" -'];
disp(CallString)
status = system(CallString);
disp(['system() status: ' num2str(status)])
[MeshData] = ReadGMSH(Mesh_File);
""", nargout=0)
    print(f"NumEls after regen: {eng.eval('length(MeshData.Surf.Elements)')}")
    eng.eval("D = DHooks2D(1000, 0.3, 'Plane Stress');", nargout=0)

    # ── Progress window over every (config, seed, sample) triple ──
    total  = len(CONFIGS) * len(SEEDS) * SIZE_LOOP
    win    = ProgressWindow(total)
    thread = threading.Thread(target=run_window, args=(win,), daemon=True)
    thread.start()

    # RESET_BENCHMARK is evaluated once, before any config runs: it decides
    # whether the shared BENCHMARK_CSV is truncated at the start of the run.
    # Every config/seed below then appends to that same open file handle.
    write_header = RESET_BENCHMARK or not BENCHMARK_CSV.exists()
    TYPE_WRITE   = 'w' if RESET_BENCHMARK else 'a'

    with open(BENCHMARK_CSV, TYPE_WRITE, newline='') as benchmark_file:
        writer = csv.writer(benchmark_file)
        if write_header:
            writer.writerow(CONFIG_COLUMNS + SEED_COLUMN + RESULT_COLUMNS)

        for CONFIG in CONFIGS:
            (STRATEGY, NETWORK, FIRST_STEP, NIF, N_CONV, USE_CBAM,
             USE_AUGMENTATION, AUGMENTATION_P, PORTION_DATA, BATCH_SIZE) = CONFIG

            for SEED in SEEDS:
                print("\n" + "=" * 79)
                print(f"BENCHMARK  config={config_label(CONFIG)}  seed={SEED}")
                print("=" * 79)

                tag       = model_tag(CONFIG, SEED)
                BEST_PATH = SEED_BENCH_DIR / tag / ('unet_' + name_file + '_best.pth')

                model = UNetTopo(
                    nif=NIF, n_in=N_in, n_out=3,
                    use_cbam=USE_CBAM, N_conv=N_CONV,
                )
                model.load_state_dict(torch.load(BEST_PATH, map_location='cpu'))
                model.eval()

                for ID in range(SIZE_LOOP):
                    sample_start = IterationSample(
                        IterationDataset(ds_filtre.get_series(ID)), 0)

                    List_iterations, List_count_FEM = run_topology_optimization(
                        sample_start, eng, model,
                        N_in=N_in,
                        N_max_iterations=100,
                        RULE=STRATEGY,
                        TYPE_FIRST=FIRST_STEP,
                        threshold=0.2,
                        N_end_FEM_iterations=0,
                        window_Unet=3,
                        window_FEM=1,
                        tol_c=1e-3,
                        tol_rho=0.1,
                        end_FEM=True,
                    )
                    win.increment()

                    idx_FEM_sol = IterData_FEM.last_iteration_index[ID]
                    FEM_sample  = IterationSample(IterData_FEM, idx_FEM_sol)

                    c_FEM  = FEM_sample.c.item()
                    c_Unet = List_iterations[-1].c.item()

                    err_rel_c  = (c_Unet - c_FEM) / c_FEM
                    number_FEM = len(List_count_FEM)
                    ds_iter_id = IterationDataset(ds_filtre.get_series(ID))

                    writer.writerow([
                        STRATEGY, NETWORK, FIRST_STEP, NIF, N_CONV,
                        USE_CBAM, USE_AUGMENTATION, AUGMENTATION_P, PORTION_DATA, BATCH_SIZE,
                        SEED,
                        ID, len(ds_iter_id), number_FEM,
                        number_FEM / len(ds_iter_id),
                        err_rel_c,
                    ])
                benchmark_file.flush()   # persist after each seed

    win.close()
    print(f"\nBenchmark written to {BENCHMARK_CSV}")


#%% Phase 3 — read results & quantify the seed-to-seed noise (per config)

RATIO_COL = 'Ratio of FEM iterations (Hybrid / full-FEM)'
ERR_COL   = 'Relative compliance error'

df_all = pd.read_csv(BENCHMARK_CSV)

for CONFIG in CONFIGS:
    label = config_label(CONFIG)
    mask  = True
    for col, val in zip(CONFIG_COLUMNS, CONFIG):
        mask &= (df_all[col] == val)
    df = df_all[mask]

    if df.empty:
        print(f"\n[skip] no rows for config {label} in {BENCHMARK_CSV}")
        continue

    # Per-seed means over the SIZE_LOOP samples.
    per_seed = df.groupby('seed').agg(
        mean_ratio = (RATIO_COL, 'mean'),
        mean_err   = (ERR_COL,   'mean'),
        n_samples  = ('Input ID', 'count'),
    ).reset_index()

    print("\n" + "#" * 79)
    print(f"Config: {label}")
    print("#" * 79)
    print(per_seed.to_string(index=False))

    # Seed-to-seed statistics = the noise assessment we are after.
    # std over the per-seed means (ddof=1) is the retraining error bar; the
    # pooled std over all samples gives the total spread for reference.
    print("\n" + "=" * 60)
    print("Seed-to-seed noise (over per-seed means, n = %d seeds)" % len(per_seed))
    print("=" * 60)
    for stat_label, col in [('FEM-iteration ratio', 'mean_ratio'),
                             ('Relative compliance error', 'mean_err')]:
        m  = per_seed[col].mean()
        sd = per_seed[col].std(ddof=1)
        cv = sd / abs(m) if m != 0 else float('nan')
        print(f"{stat_label:28s}: mean={m:.4f}  std={sd:.4f}  "
              f"CV={100*cv:5.1f}%  (min={per_seed[col].min():.4f}, "
              f"max={per_seed[col].max():.4f})")


#%% Phase 3b — visualise the seed-to-seed variability
# Uses the CONFIGS defined at the top of the file (the ones actually benchmarked),
# so the figures show exactly those configurations and nothing stale in the CSV.

# CONFIGS = [
#     ['Only UNet', 'U-Net', 'UNet', 32, 2, False, True, 0.2, 0.2, 16],
#     ['Only UNet', 'U-Net', 'UNet', 32, 2, False, True, 0.2, 0.5, 16],
#     ['Only UNet', 'U-Net', 'UNet', 32, 2, False, True, 0.2,  1 , 16],
# ]

CONFIGS = [
    ['Only UNet', 'U-Net', 'UNet', 32, 2, False, False, 0.2, 0.2, 16],
    ['Only UNet', 'U-Net', 'UNet', 32, 2, False, False, 0.2, 0.5, 16],
    ['Only UNet', 'U-Net', 'UNet', 32, 2, False, False, 0.2,  1 , 16],
]


# Per-seed boxplots + mean ± 1σ band, for one dataset portion.
plot_seed_benchmark(BENCHMARK_CSV, portion=0.5,
                    save_path=SEED_BENCH_DIR / 'seed_benchmark.png',
                    scale_font=1.5)

# Cost/quality plane: one point per seed, one colour and one 1σ ellipse per config.
plot_seed_quality_cost(BENCHMARK_CSV, configs=CONFIGS,
                       save_path=SEED_BENCH_DIR / 'seed_quality_cost.png',
                       scale_font=1.5, scale_dot=2, SHOW_TITLE=False)

# %%

#%% TopOpt_benchmark_data_augment.py
# Equivariance / data-augmentation assessment.
#
# For each configuration of `list_benchmark` (typically the same model trained
# WITH and WITHOUT data augmentation), loads the trained model and runs the
# hybrid TopOpt benchmark on the same 100 force distributions three times:
#   - 'original'    : the samples as they are,
#   - 'rotation_90' : each sample rotated by 90 degrees,
#   - 'symmetry_x'  : each sample mirrored about the vertical axis.
# Rotation and horizontal mirror are physical symmetries of the problem, so a
# perfectly equivariant model returns the SAME cost and quality on all three.
# The full-FEM optimum (compliance and iteration count) is invariant under
# these transforms, so the ORIGINAL reference is reused for every operation and
# the per-sample gap measures how much the model breaks the symmetry.
#
# Every per-sample result is tagged with its config + operation and stored in
# benchmark_data_augment.csv. The read section plots, per config and per
# operation, the difference (vs. the original) of the FEM-iteration ratio (%)
# and of the relative compliance error (%): 100 points per operation.

#%% Import libraries
import sys
import csv
from pathlib import Path

import matlab.engine
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from model         import *
from dataset       import *
from topology_utils import *


#%% Paths & constants

name_file = 'dataset_1k'    # training set (defines the model tag)
name_data = 'dataset_test'  # benchmark evaluation set

BASE         = Path(__file__).parents[3]
RESULTS_ROOT = BASE / 'Software' / 'OT_NN' / 'Pytorch_NN' / 'results'
GMSH_EXE     = BASE.parent / 'gmsh' / 'gmsh.exe'

DATA_PATH = BASE / 'HeavyFiles' / 'data' / (name_data + '.mat')

sys.path.append(str(BASE / 'Software' / 'OT_NN' / 'Pytorch_NN'))
sys.path.append(str(BASE / 'Software' / 'OT_Functions'))
sys.path.append(str(BASE / 'Software' / 'OT_Software'))

IMG_SIZE = 32
PENAL    = 3
RMIN     = 1.5
NGPpL    = 2
NGPpS    = 9
E        = 1000
NU       = 0.3

HIDDEN_LAYERS_MLP = [32, 64]
EMBED_OUT         = 128


#%% Load dataset (same preparation as TopOpt_benchmark_model.py)

data    = load_mat(DATA_PATH)
ds_base = Dataset_TopOpt(data)
ds_base = ds_base.normalize_dataset()             # per-sample normalisation
ds_filtre = ds_base.filtre_dataset(rho_min=0.15, rho_max=0.85)
IterData_FEM = IterationDataset(ds_filtre)


#%% Benchmark definition

# [Strategy, Model, First step, NIF, N_conv, use cbam, use augmentation,
#  probability of augmentation, dataset portion, batch size]
list_benchmark = [
    ['Only UNet', 'U-Net', 'UNet', 32, 2, False, False, 0.2, 1, 16],  # without augmentation
    ['Only UNet', 'U-Net', 'UNet', 32, 2, False, True,  0.5, 1, 16],  # with augmentation (50%)
]

# The three data operations applied to each starting sample. 'original' is the
# reference; the other two are physical symmetries the model should preserve.
OPERATIONS = {
    'original':    lambda s: s,
    'rotation_90': lambda s: rotation_90(s, N_rot=1),
    'symmetry_x':  symmetry_x,
}


def model_tag(NETWORK, NIF, N_CONV, USE_CBAM, USE_AUGMENTATION, AUGMENTATION_P,
              PORTION_DATA, BATCH_SIZE):
    """Directory tag of the trained model (matches results/<NETWORK>/<tag>)."""
    aug_tag = f'{int(100*AUGMENTATION_P)}%' if USE_AUGMENTATION else 'False'
    if NETWORK == 'BE_UNet':
        return (f'{name_file}_NIF={NIF}_{N_CONV}_conv_{HIDDEN_LAYERS_MLP}_'
                f'CBAM={USE_CBAM}_aug={aug_tag}_portion={int(PORTION_DATA*100)}%_'
                f'batch={BATCH_SIZE}')
    return (f'{name_file}_NIF={NIF}_{N_CONV}_conv_CBAM={USE_CBAM}_aug={aug_tag}_'
            f'portion={int(PORTION_DATA*100)}%_batch={BATCH_SIZE}')


def load_model(NETWORK, N_in, NIF, N_CONV, USE_CBAM, USE_AUGMENTATION,
               AUGMENTATION_P, PORTION_DATA, BATCH_SIZE):
    """Build the network of a config row and load its best checkpoint."""
    if NETWORK == 'BE_UNet':
        model = BE_UNetTopo(nif=NIF, n_in=N_in, n_out=3, use_cbam=USE_CBAM,
                            hidden_layers_MLP=HIDDEN_LAYERS_MLP,
                            embed_out=EMBED_OUT, N_conv=N_CONV)
    elif NETWORK == 'U-Net':
        model = UNetTopo(nif=NIF, n_in=N_in, n_out=3,
                         use_cbam=USE_CBAM, N_conv=N_CONV)
    else:
        raise ValueError("Invalid NETWORK value. Choose 'U-Net' or 'BE_UNet'.")

    tag       = model_tag(NETWORK, NIF, N_CONV, USE_CBAM, USE_AUGMENTATION,
                          AUGMENTATION_P, PORTION_DATA, BATCH_SIZE)
    best_path = RESULTS_ROOT / NETWORK / tag / ('unet_' + name_file + '_best.pth')
    model.load_state_dict(torch.load(best_path, map_location='cpu'))
    model.eval()
    return model


#%% CSV layout

CONFIG_COLUMNS = ['Strategy', 'Model', 'First step', 'NIF', 'N_conv',
                  'use cbam', 'use augmentation', 'probability of augmentation',
                  'dataset portion', 'batch size']
OP_COLUMN      = ['Operation']
RESULT_COLUMNS = ['Input ID',
                  'Number of FEM iterations for the full-FEM strategy',
                  'Number of FEM iterations for the hybrid strategy',
                  'Ratio of FEM iterations (Hybrid / full-FEM)',
                  'Relative compliance error']

SIZE_LOOP           = 100
name_benchmark_file = 'benchmark_data_augment.csv'
BENCHMARK_CSV       = RESULTS_ROOT / 'U-Net' / name_benchmark_file

RUN_BENCHMARK   = True    # Set to False to skip the benchmark and only read the CSV
RESET_BENCHMARK = True    # overwrite the CSV (False -> append)


#%% Run benchmark

if RUN_BENCHMARK:
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

    # ── Progress window over every (config, operation, sample) triple ──
    total  = len(list_benchmark) * len(OPERATIONS) * SIZE_LOOP
    win    = ProgressWindow(total)
    thread = threading.Thread(target=run_window, args=(win,), daemon=True)
    thread.start()

    TYPE_WRITE = 'w' if RESET_BENCHMARK else 'a'
    with open(BENCHMARK_CSV, TYPE_WRITE, newline='') as benchmark_file:
        writer = csv.writer(benchmark_file)
        if RESET_BENCHMARK:
            writer.writerow(CONFIG_COLUMNS + OP_COLUMN + RESULT_COLUMNS)

        for (STRATEGY, NETWORK, FIRST_STEP, NIF, N_CONV, USE_CBAM,
             USE_AUGMENTATION, AUGMENTATION_P, PORTION_DATA, BATCH_SIZE) in list_benchmark:

            N_in  = 1 if NETWORK == 'BE_UNet' else 3
            model = load_model(NETWORK, N_in, NIF, N_CONV, USE_CBAM,
                               USE_AUGMENTATION, AUGMENTATION_P, PORTION_DATA, BATCH_SIZE)
            print("\n" + "=" * 79)
            print(f"CONFIG  {NETWORK}  aug={USE_AUGMENTATION} "
                  f"(p={AUGMENTATION_P})  cbam={USE_CBAM}")
            print("=" * 79)

            for OP_NAME, transform in OPERATIONS.items():
                print(f"  operation: {OP_NAME}")

                for ID in range(SIZE_LOOP):
                    # Original starting sample, then the operation applied to it.
                    sample_start = IterationSample(
                        IterationDataset(ds_filtre.get_series(ID)), 0)
                    sample_start = transform(sample_start)

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

                    # Full-FEM reference: invariant under rotation/mirror, so
                    # reuse the ORIGINAL optimum (compliance and iteration count)
                    # for all operations. The gap then isolates the model's
                    # non-equivariance.
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
                        OP_NAME,
                        ID, len(ds_iter_id), number_FEM,
                        number_FEM / len(ds_iter_id),
                        err_rel_c,
                    ])
                benchmark_file.flush()   # persist after each operation

    win.close()
    print(f"\nBenchmark written to {BENCHMARK_CSV}")


#%% Read File Benchmark

RATIO_COL = 'Ratio of FEM iterations (Hybrid / full-FEM)'
ERR_COL   = 'Relative compliance error'

df = pd.read_csv(BENCHMARK_CSV)

# Operations to compare against 'original' (everything but the reference itself).
compared_ops = [op for op in OPERATIONS if op != 'original']


def _config_mask(df, bench):
    """Dtype-robust match of one list_benchmark config against the CSV rows.

    FEM-only rows leave numeric config columns blank, forcing those columns to
    object (string) dtype on read; a plain `==` would compare '32' == 32 and
    never match. Numeric fields are coerced to numbers, the rest to trimmed
    strings (with NaN placeholders normalised to '').
    """
    mask = pd.Series(True, index=df.index)
    for col, val in zip(CONFIG_COLUMNS, bench):
        if isinstance(val, bool):
            mask &= df[col].astype(str).str.strip() == str(val)
        elif isinstance(val, (int, float)):
            mask &= pd.to_numeric(df[col], errors='coerce') == val
        else:
            col_norm = df[col].where(df[col].notna(), '').astype(str).str.strip()
            mask &= col_norm == str(val).strip()
    return mask


def _config_label(bench):
    """Short human-readable label for a config (augmentation state)."""
    use_aug, aug_p = bench[6], bench[7]
    return f'aug={int(100*aug_p)}%' if use_aug else 'no aug'


# One subplot per config so with/without augmentation sit side by side.
n_cfg = len(list_benchmark)
fig, axes = plt.subplots(1, n_cfg, figsize=(7 * n_cfg, 7), squeeze=False)
axes = axes[0]
colors = plt.cm.tab10(np.linspace(0, 1, len(compared_ops)))

for ax, bench in zip(axes, list_benchmark):
    sub   = df[_config_mask(df, bench)]
    ratio = sub.pivot_table(index='Input ID', columns='Operation', values=RATIO_COL)
    err   = sub.pivot_table(index='Input ID', columns='Operation', values=ERR_COL)

    print(f"\n=== {_config_label(bench)} ===")
    for op, color in zip(compared_ops, colors):
        # Per-sample difference vs. the original, in percentage points.
        d_ratio = (ratio[op] - ratio['original']) * 100
        d_err   = (err[op]   - err['original'])   * 100
        ax.scatter(d_ratio, d_err, s=40, color=color, alpha=0.7,
                   edgecolor='black', linewidth=0.3,
                   label=f'{op}  (n={int(d_ratio.notna().sum())})')
        print(f"  {op}:")
        print(f"    d FEM-iteration ratio (%) : mean={d_ratio.mean():+.2f}  "
              f"std={d_ratio.std():.2f}  |d|mean={d_ratio.abs().mean():.2f}")
        print(f"    d compliance error    (%) : mean={d_err.mean():+.3f}  "
              f"std={d_err.std():.3f}  |d|mean={d_err.abs().mean():.3f}")

    ax.axhline(0, color='black', linewidth=0.8, zorder=1)
    ax.axvline(0, color='black', linewidth=0.8, zorder=1)
    ax.set_xlabel('Difference of FEM-iteration ratio vs. original (pts)', fontsize=11)
    ax.set_ylabel('Difference of relative compliance error vs. original (pts)', fontsize=11)
    ax.set_title(_config_label(bench), fontsize=13)
    ax.legend(fontsize=10, loc='best')

fig.suptitle(f'Data-augmentation robustness — {SIZE_LOOP} samples per operation',
             fontsize=15)
plt.tight_layout()
plt.savefig(RESULTS_ROOT / 'U-Net' / 'benchmark_data_augment.png', dpi=150)
plt.show()

# %%

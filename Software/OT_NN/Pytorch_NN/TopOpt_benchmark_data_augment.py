#%% TopOpt_benchmark_data_augment.py
# Equivariance assessment of a single, fixed model.
#
# Loads ONE trained model and runs the hybrid TopOpt benchmark on the same 100
# force distributions three times:
#   - 'original'    : the samples as they are,
#   - 'rotation_90' : each sample rotated by 90 degrees,
#   - 'symmetry_x'  : each sample mirrored about the vertical axis.
# Rotation and horizontal mirror are physical symmetries of the problem, so a
# perfectly equivariant model would return the SAME cost and quality on all
# three. The reference full-FEM optimum (compliance and iteration count) is
# invariant under these transforms, hence the ORIGINAL reference is reused for
# every operation and the per-sample gap measures how much the model breaks.
#
# Every per-sample result is tagged with its operation and stored in
# benchmark_data_augment.csv. The read section plots, per operation and per
# sample, the difference (vs. the original) of the FEM-iteration ratio (%) and
# of the relative compliance error (%): 100 points per operation.

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


#%% Load dataset (same preparation as TopOpt_benchmark_model.py)

data    = load_mat(DATA_PATH)
ds_base = Dataset_TopOpt(data)
ds_base = ds_base.normalize_dataset()             # per-sample normalisation
ds_filtre = ds_base.filtre_dataset(rho_min=0.15, rho_max=0.85)
IterData_FEM = IterationDataset(ds_filtre)


#%% Model under test

# [Strategy, Model, First step, NIF, N_conv, use cbam, use augmentation,
#  probability of augmentation, dataset portion, batch size]
CONFIG = ['Only UNet', 'U-Net', 'UNet', 32, 2, False, False, 0.2, 1, 16]  # reference U-Net

(STRATEGY, NETWORK, FIRST_STEP, NIF, N_CONV, USE_CBAM,
 USE_AUGMENTATION, AUGMENTATION_P, PORTION_DATA, BATCH_SIZE) = CONFIG

HIDDEN_LAYERS_MLP = [32, 64]
EMBED_OUT         = 128
N_in              = 1 if NETWORK == 'BE_UNet' else 3


def model_tag():
    """Directory tag of the trained model (matches results/<NETWORK>/<tag>)."""
    aug_tag = f'{int(100*AUGMENTATION_P)}%' if USE_AUGMENTATION else 'False'
    if NETWORK == 'BE_UNet':
        return (f'{name_file}_NIF={NIF}_{N_CONV}_conv_{HIDDEN_LAYERS_MLP}_'
                f'CBAM={USE_CBAM}_aug={aug_tag}_portion={int(PORTION_DATA*100)}%_'
                f'batch={BATCH_SIZE}')
    return (f'{name_file}_NIF={NIF}_{N_CONV}_conv_CBAM={USE_CBAM}_aug={aug_tag}_'
            f'portion={int(PORTION_DATA*100)}%_batch={BATCH_SIZE}')


def load_model():
    """Build the network of CONFIG and load its best checkpoint."""
    if NETWORK == 'BE_UNet':
        model = BE_UNetTopo(nif=NIF, n_in=N_in, n_out=3, use_cbam=USE_CBAM,
                            hidden_layers_MLP=HIDDEN_LAYERS_MLP,
                            embed_out=EMBED_OUT, N_conv=N_CONV)
    elif NETWORK == 'U-Net':
        model = UNetTopo(nif=NIF, n_in=N_in, n_out=3,
                         use_cbam=USE_CBAM, N_conv=N_CONV)
    else:
        raise ValueError("Invalid NETWORK value. Choose 'U-Net' or 'BE_UNet'.")

    best_path = RESULTS_ROOT / NETWORK / model_tag() / ('unet_' + name_file + '_best.pth')
    model.load_state_dict(torch.load(best_path, map_location='cpu'))
    model.eval()
    return model


# The three data operations applied to each starting sample. 'original' is the
# reference; the other two are physical symmetries the model should preserve.
OPERATIONS = {
    'original':    lambda s: s,
    'rotation_90': lambda s: rotation_90(s, N_rot=1),
    'symmetry_x':  symmetry_x,
}


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
BENCHMARK_CSV       = RESULTS_ROOT / NETWORK / name_benchmark_file

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

    model = load_model()
    print(f"Model loaded: {NETWORK} / {model_tag()}")

    # ── Progress window over every (operation, sample) pair ──
    total  = len(OPERATIONS) * SIZE_LOOP
    win    = ProgressWindow(total)
    thread = threading.Thread(target=run_window, args=(win,), daemon=True)
    thread.start()

    TYPE_WRITE = 'w' if RESET_BENCHMARK else 'a'
    with open(BENCHMARK_CSV, TYPE_WRITE, newline='') as benchmark_file:
        writer = csv.writer(benchmark_file)
        if RESET_BENCHMARK:
            writer.writerow(CONFIG_COLUMNS + OP_COLUMN + RESULT_COLUMNS)

        for OP_NAME, transform in OPERATIONS.items():
            print("\n" + "=" * 79)
            print(f"OPERATION  {OP_NAME}")
            print("=" * 79)

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

                # Full-FEM reference: invariant under rotation/mirror, so reuse
                # the ORIGINAL optimum (compliance and iteration count) for all
                # operations. The gap then isolates the model's non-equivariance.
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

# One row per (operation, sample); pivot so each metric has one column per op.
ratio = df.pivot_table(index='Input ID', columns='Operation', values=RATIO_COL)
err   = df.pivot_table(index='Input ID', columns='Operation', values=ERR_COL)

# Operations to compare against 'original' (everything but the reference itself).
compared_ops = [op for op in OPERATIONS if op != 'original']

fig, ax = plt.subplots(figsize=(8, 7))
colors = plt.cm.tab10(np.linspace(0, 1, len(compared_ops)))

for op, color in zip(compared_ops, colors):
    # Per-sample difference vs. the original, in percentage points.
    d_ratio = (ratio[op] - ratio['original']) * 100
    d_err   = (err[op]   - err['original'])   * 100
    ax.scatter(d_ratio, d_err, s=40, color=color, alpha=0.7,
               edgecolor='black', linewidth=0.3,
               label=f'{op}  (n={int(d_ratio.notna().sum())})')
    print(f"\n{op}:")
    print(f"  d FEM-iteration ratio (%) : mean={d_ratio.mean():+.2f}  "
          f"std={d_ratio.std():.2f}  |d|mean={d_ratio.abs().mean():.2f}")
    print(f"  d compliance error    (%) : mean={d_err.mean():+.3f}  "
          f"std={d_err.std():.3f}  |d|mean={d_err.abs().mean():.3f}")

ax.axhline(0, color='black', linewidth=0.8, zorder=1)
ax.axvline(0, color='black', linewidth=0.8, zorder=1)
ax.set_xlabel('Difference of FEM-iteration ratio vs. original (percentage points)', fontsize=12)
ax.set_ylabel('Difference of relative compliance error vs. original (percentage points)', fontsize=12)
ax.set_title(f'Data-augmentation robustness — {NETWORK}\n'
             f'{SIZE_LOOP} samples per operation', fontsize=14)
ax.legend(fontsize=11, loc='best')
plt.tight_layout()
plt.savefig(RESULTS_ROOT / NETWORK / 'benchmark_data_augment.png', dpi=150)
plt.show()

# %%

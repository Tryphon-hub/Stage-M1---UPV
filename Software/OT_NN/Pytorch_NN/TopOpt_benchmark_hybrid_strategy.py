#%% Import libraries
import sys
import matlab.engine
from pathlib import Path
import re
import csv
import pandas as pd

import torch
import numpy as np
from model         import *
from dataset       import *
from topology_utils import *


name_file = 'dataset_1k'
name_data = 'dataset_test'


BASE = Path(__file__).parents[3]

GMSH_EXE = BASE.parent / 'gmsh' / 'gmsh.exe'

DATA_PATH       = BASE / 'HeavyFiles' / 'data' / (name_data + '.mat')

sys.path.append(str(BASE / 'Software' / 'OT_NN' / 'Pytorch_NN'))
sys.path.append(str(BASE / 'Software' / 'OT_Functions'))
sys.path.append(str(BASE / 'Software' / 'OT_Software'))


#%% Benchmark settings

RUN_BENCHMARK = False
RESET_BENCHMARK = False # deletes old benchmark csv file

#%% Constants
IMG_SIZE = 32
PENAL    = 3
RMIN     = 1.5
NGPpL    = 2
NGPpS    = 9
E        = 1000
NU       = 0.3



#%% Load dataset
data    = load_mat(DATA_PATH)
ds_base = Dataset_TopOpt(data)
ds_filtre = ds_base.filtre_dataset(rho_min=0.15, rho_max=0.85)
IterData_FEM = IterationDataset(ds_filtre)

List_List_iterations=[]

#%% Define model
NETWORK = 'U-Net'
NIF = 32
USE_CBAM = False
N_CONV = 2
USE_AUGMENTATION = False
PORTION_DATA = 1
AUGMENTATION_P = 0.0
BATCH_SIZE = 16



#%% Load Model

HIDDEN_LAYERS_MLP=[32,64]
EMBED_OUT   = 128     # dimension de l'embedding

if NETWORK=='BE_UNet':
    N_in=1
else:
    N_in=3


# ── Output directories for this configuration (mirrors main.py) ──
RESULTS_ROOT        = BASE / 'Software' / 'OT_NN' / 'Pytorch_NN' / 'results'

aug_tag = f'{int(100*AUGMENTATION_P)}%' if USE_AUGMENTATION else 'False'

if NETWORK == 'U-Net':
    tag = f'{name_file}_NIF={NIF}_{N_CONV}_conv_CBAM={USE_CBAM}_aug={aug_tag}_portion={int(PORTION_DATA*100)}%_batch={BATCH_SIZE}'
else:
    mlp_tag = '-'.join(str(h) for h in HIDDEN_LAYERS_MLP)
    tag = f'{name_file}_NIF={NIF}_{N_CONV}_conv_{mlp_tag}_CBAM={USE_CBAM}_aug={aug_tag}_portion={int(PORTION_DATA*100)}%_batch={BATCH_SIZE}'


RESULTS_DIR       = RESULTS_ROOT / NETWORK / tag
ILLUSTRATIONS_DIR = BASE / 'Software' / 'OT_NN' / 'Pytorch_NN' / 'illustrations' / NETWORK / tag
BEST_PATH         = RESULTS_DIR / ('unet_' + name_file + '_best.pth')


# Load model
if NETWORK=='BE_UNet':
    model = BE_UNetTopo(
        nif           = NIF,
        n_in          = N_in,          # ρ seul — tractions via BoundaryEmbedding
        n_out         = 3,
        use_cbam      = USE_CBAM,
        hidden_layers_MLP = HIDDEN_LAYERS_MLP,
        embed_out     = EMBED_OUT,
        N_conv=N_CONV,
    )
    
elif NETWORK=='U-Net':
    model = UNetTopo(
        nif=32, 
        n_in=N_in, 
        n_out=3, 
        use_cbam=USE_CBAM,
        N_conv=N_CONV,
        )

else:
    raise ValueError("Invalid NETWORK value. Choose 'U-net' or 'BE_Unet'.")

state_dict = torch.load(
    BEST_PATH,
    map_location='cpu'
)

model.load_state_dict(state_dict)
model.eval()





#%% Start MATLAB engine
eng = matlab.engine.start_matlab()
eng.addpath(str(BASE / 'Software' / 'OT_Functions'))
eng.addpath(str(BASE / 'Software' / 'OT_Software'))

#%% Regenerate and load the mesh

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


#%% Define benchmark

# [Strategy, Model, First step, NIF, N_conv, use cbam, use augmentation, probability of augmentation, dataset portion]
list_benchmark = [
    ['Only UNet', 'UNet'],
    ['Only UNet', 'FEM'],
    # ['10 Unet - 1 FEM', 'UNet'],
    # ['10 Unet - 1 FEM', 'FEM'],
    ['10 Unet - 3 FEM', 'UNet'],
    ['10 Unet - 3 FEM', 'FEM'],
    ['Decreasing compliance', 'UNet'],
    ['Decreasing compliance', 'FEM'],
]

# Column layout of each list_benchmark entry, reused for the CSV header,
# the per-run rows and the aggregation at the end.
CONFIG_COLUMNS = ['Strategy', 'First step']

RESULT_COLUMNS = ['Input ID',
                  'Number of FEM iterations for the full-FEM strategy',
                  'Number of FEM iterations for the hybrid strategy',
                  'Ratio of FEM iterations (Hybrid / full-FEM)',
                  'Relative compliance error']




if RESET_BENCHMARK:
    TYPE_WRITE = 'w'
else: 
    TYPE_WRITE = 'a'

SIZE_LOOP = 100

name_benchmark_file = 'benchmark_hybrid_strategy.csv'




#%% Run benchmark

if RUN_BENCHMARK:

    total = len(list_benchmark) * SIZE_LOOP
    win = ProgressWindow(total)
    thread = threading.Thread(target=run_window, args=(win,), daemon=True)
    thread.start()


    with open(BASE / 'Software' / 'OT_NN' / 'Pytorch_NN' / 'results' / name_benchmark_file, TYPE_WRITE, newline='') as benchmark_file:

        writer = csv.writer(benchmark_file)
        if RESET_BENCHMARK:
            writer.writerow(CONFIG_COLUMNS + RESULT_COLUMNS)

        for STRATEGY, FIRST_STEP in list_benchmark:
                    
            for ID in range(SIZE_LOOP):

                sample_start = IterationSample(IterationDataset(ds_filtre.get_series(ID)), 0)

                List_iterations, List_count_FEM = run_topology_optimization(
                    sample_start,  
                    eng, 
                    model,
                    N_in=N_in, 
                    N_max_iterations=100,
                    RULE=STRATEGY,
                    TYPE_FIRST=FIRST_STEP,
                    threshold=0.0, 
                    N_end_FEM_iterations=0,
                    window_Unet=3, 
                    window_FEM=1,
                    tol_c=1e-3, 
                    tol_rho=0.1, 
                    end_FEM=True
                )

                win.increment()

                idx_FEM_sol = IterData_FEM.last_iteration_index[ID]
                FEM_sample  = IterationSample(IterData_FEM, idx_FEM_sol)

                c_FEM  = FEM_sample.c.item()
                # Re-evaluate the final density with FEM: List_iterations[-1].c
                # may hold a U-Net *predicted* pseudo-compliance when the run
                # ends on a U-Net step (e.g. '10 Unet - 1 FEM' hitting the cap),
                # which would otherwise give errors of thousands of %.
                c_Unet = compliance_FEM(eng, List_iterations[-1])

                err_rel_c  = (c_Unet - c_FEM) / c_FEM
                number_FEM = len(List_count_FEM)
                ds_iter    = IterationDataset(ds_filtre.get_series(ID))

                writer.writerow([
                    STRATEGY, FIRST_STEP,
                    ID, len(ds_iter), number_FEM,
                    number_FEM / len(ds_iter),
                    err_rel_c
                ])

    win.close()

    

#%% Read File Benchmark

df = pd.read_csv(BASE / 'Software' / 'OT_NN' / 'Pytorch_NN' / 'results' / name_benchmark_file)

# Agrégation par configuration
summary = df.groupby(CONFIG_COLUMNS).agg(
    mean_full_FEM  = ('Number of FEM iterations for the full-FEM strategy', 'mean'),
    mean_hybrid    = ('Number of FEM iterations for the hybrid strategy',   'mean'),
    std_hybrid     = ('Number of FEM iterations for the hybrid strategy',   'std'),
    mean_ratio     = ('Ratio of FEM iterations (Hybrid / full-FEM)',        'mean'),
    std_ratio      = ('Ratio of FEM iterations (Hybrid / full-FEM)',        'std'),
    mean_err       = ('Relative compliance error',                          'mean'),
    std_err        = ('Relative compliance error',                          'std'),
    n_samples      = ('Input ID',                                           'count')
).reset_index()

print(summary.to_string())


# Retrieve values in lists ordered as list_benchmark
Tab_ratio_FEM = []
Tab_err_rel_c = []

for bench in list_benchmark:
    mask = (df[CONFIG_COLUMNS] == pd.Series(bench, index=CONFIG_COLUMNS)).all(axis=1)
    row  = df[mask]

    Tab_ratio_FEM.append(row['Ratio of FEM iterations (Hybrid / full-FEM)'].values)
    Tab_err_rel_c.append(row['Relative compliance error'].values)

Tab_ratio_FEM = np.array(Tab_ratio_FEM)  # (n_configs, SIZE_LOOP)
Tab_err_rel_c = np.array(Tab_err_rel_c)  # (n_configs, SIZE_LOOP)

plot_FEM_error_c(list_benchmark, Tab_ratio_FEM, Tab_err_rel_c, up_legend=1.15,
                 scale_font=1.5)


plot_pareto_front_c(list_benchmark, Tab_ratio_FEM, Tab_err_rel_c,
                        TYPE_BENCHMARK='Hybrid', use_abs_error=False,
                        show_error=False, SAVE_PATH=None, SHOW_TITLE=False,
                        scale_font = 1.5, scale_dot = 2,
                        low_margin=0.1, right_margin=0.1,left_margin=0.05)



COL_PERF = 'Ratio of FEM iterations (Hybrid / full-FEM)'
COL_PREC = 'Relative compliance error'



#%% Comparisons quoted in the text (Sections 5.2.6 and 5.2.7)
compare_configs(df, ['Only UNet', 'UNet'], ['Only UNet', 'FEM'], CONFIG_COLUMNS, COL_PERF, COL_PREC)
compare_configs(df, ['Decreasing compliance', 'UNet'], ['Only UNet', 'UNet'], CONFIG_COLUMNS, COL_PERF, COL_PREC)
compare_configs(df, ['Decreasing compliance', 'FEM'],  ['Only UNet', 'FEM'], CONFIG_COLUMNS, COL_PERF, COL_PREC)
compare_configs(df, ['10 Unet - 3 FEM', 'UNet'], ['Only UNet', 'UNet'], CONFIG_COLUMNS, COL_PERF, COL_PREC)


#%% Paired comparison figures 
comparisons = [
    (['Only UNet', 'UNet'], ['Only UNet', 'FEM']),
    (['Decreasing compliance', 'UNet'], ['Only UNet', 'UNet']),
    (['Decreasing compliance', 'FEM'],  ['Only UNet', 'FEM']),
    (['10 Unet - 3 FEM', 'UNet'], ['Only UNet', 'UNet']),
]
short_labels = ['Only U-Net:\nU-Net vs FEM start',
                'Decreasing vs Only U-Net\n(U-Net start)',
                'Decreasing vs Only U-Net\n(FEM start)',
                'Periodic vs Only U-Net\n(U-Net start)']

FIGURES_DIR = BASE / 'Software' / 'OT_NN' / 'Pytorch_NN' / 'illustrations' / 'Benchmark_hybrid'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Forest plot: mean differences, 95% CI, Student/Wilcoxon concordance
plot_forest_paired(
    df, comparisons, short_labels,
    CONFIG_COLUMNS, COL_PERF, COL_PREC,
    scale_font=1.2,
    SAVE_PATH=FIGURES_DIR / 'forest_paired_hybrid.pdf')

# Distributions of the paired differences d_i: what the Wilcoxon test sees
plot_paired_distributions(
    df, comparisons, CONFIG_COLUMNS, labels=short_labels,
    metric='Ratio of FEM iterations (Hybrid / full-FEM)',
    xlabel='Performance difference (pp)',
    scale_font=1.2,
    SAVE_PATH=FIGURES_DIR / 'paired_distributions_performance.pdf')

plot_paired_distributions(
    df, comparisons, CONFIG_COLUMNS,labels=short_labels,
    metric='Relative compliance error',
    xlabel='Precision difference (pp)',
    scale_font=1.2,
    SAVE_PATH=FIGURES_DIR / 'paired_distributions_precision.pdf')


#%% Helper for robust config matching
# `config_mask` lives in topology_utils (imported with *), shared by every
# benchmark script.

# Rebuilt from scratch here: the cell above already turned these two names into
# numpy arrays, so re-running this cell would otherwise fail on `.append`.
counts        = []
Tab_ratio_FEM = []
Tab_err_rel_c = []

for bench in list_benchmark:
    mask = config_mask(df, bench, CONFIG_COLUMNS)
    row  = df[mask]
    counts.append(len(row))

    Tab_ratio_FEM.append(row['Ratio of FEM iterations (Hybrid / full-FEM)'].values)
    Tab_err_rel_c.append(row['Relative compliance error'].values)

# All configs must match the same (non-zero) number of rows, otherwise the
# stack below fails with a cryptic "inhomogeneous shape" numpy error.
if len(set(counts)) != 1 or counts[0] == 0:
    details = '\n'.join(f'  {n:3d} rows  {bench}'
                        for n, bench in zip(counts, list_benchmark))
    raise ValueError(
        f"Each configuration must match the same non-zero number of rows in "
        f"'{name_benchmark_file}', but got:\n{details}\n"
        f"Configs with 0 rows are absent from the CSV (run the benchmark for "
        f"them, or remove them from list_benchmark)."
    )

Tab_ratio_FEM = np.array(Tab_ratio_FEM)  # (n_configs, SIZE_LOOP)
Tab_err_rel_c = np.array(Tab_err_rel_c)  # (n_configs, SIZE_LOOP)

# 'Hybrid': the entries of list_benchmark hold [Strategy, First step] only, so
# there are no architecture parameters to tabulate under the bars.
plot_FEM_error_c(list_benchmark, Tab_ratio_FEM, Tab_err_rel_c, 'Hybrid',
                 scale_font=1.2, SHOW_TITLE=False)



# SMAPE Benchmark

# Samples on which the sMAPE is evaluated (full filtered dataset).
ds_iter_smape = IterationDataset(ds_filtre)

name_smape_file = 'benchmark_smape_architecture.csv'
smape_csv       = RESULTS_ROOT / name_smape_file


#%%
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
name_data_energy = 'dataset_10k'


BASE = Path(__file__).parents[3]
RESULTS_ROOT = BASE / 'Software' / 'OT_NN' / 'Pytorch_NN' / 'results'

GMSH_EXE = BASE.parent / 'gmsh' / 'gmsh.exe'

DATA_PATH       = BASE / 'HeavyFiles' / 'data' / (name_data + '.mat')

sys.path.append(str(BASE / 'Software' / 'OT_NN' / 'Pytorch_NN'))
sys.path.append(str(BASE / 'Software' / 'OT_Functions'))
sys.path.append(str(BASE / 'Software' / 'OT_Software'))



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

# Bigger dataset for the energy-based method
data_energy    = load_mat(BASE / 'HeavyFiles' / 'data' / (name_data_energy + '.mat'))
ds_energy = Dataset_TopOpt(data_energy)
ds_energy_filtre = ds_energy.filtre_dataset(rho_min=0.15, rho_max=0.85)
ds_acc = AcceleratedDataset(ds_energy_filtre)
# ds_acc_aug    = ds_acc.augment()



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

# [Strategy, Model, First step, NIF, N_conv, use cbam, use augmentation, probability of augmentation, dataset portion, batch size]
# list_benchmark = [
    # ['Only UNet', 'U-Net', 'UNet', 16, 2, False, False, 0.2, 0.5, 16],
    # ['Only UNet', 'U-Net', 'UNet', 32, 2, False, False, 0.2, 1  , 16], # reference configuration
    # ['Only UNet', 'U-Net', 'UNet', 32, 2, False, False, 0.2, 1  ,  8], # batch size 8
    # ['Only UNet', 'U-Net', 'UNet', 32, 2, False, False, 0.2, 1  , 32], # batch size 32
    # ['Only UNet', 'U-Net', 'UNet', 32, 2, False, False, 0.2, 0.5, 16],
    # ['Only UNet','U-Net'  , 'UNet', 32,2, False, True , 0.5, 1  , 16], # data augmentation
    # ['Only UNet', 'U-Net', 'UNet', 32, 3, False, False, 0.2, 1  , 16], # N_conv = 3
    # ['Only UNet', 'U-Net', 'UNet', 32, 2, True , False, 0.2, 1  , 16], #CBAM
    # ['Only UNet', 'U-Net', 'UNet', 32, 2, False, True , 0.2, 1  , 16], # data augmentation
    # ['Only UNet', 'U-Net', 'UNet', 32, 3, True , True , 0.2, 1  , 16],
    # ['Only UNet', 'U-Net', 'UNet', 32, 2, True , True , 0.2, 0.5, 16],
    # ['Only UNet','BE_UNet', 'UNet', 32, 2, False, False, 0.2, 1 , 16], # reference BE_UNet configuration
    # ['Only UNet','BE_UNet', 'UNet', 32, 2, False, False, 0.2,0.5, 16],
    # ['Only UNet','BE_UNet', 'UNet', 32, 2, True, False , 0.2, 1 , 16],
    # ['Only UNet','BE_UNet', 'UNet', 32, 2, True, True , 0.2 , 1 , 16],
    # ['Only FEM', 'Energy', 'FEM', ' ', ' ', ' ', False, ' ', ' ' , ' '], # reference energy-based method
    # ['Only FEM', 'Energy', 'FEM', ' ', ' ', ' ', True, ' ', ' ' , ' '], # data augmentation

# ]


list_benchmark = [
    ['Only UNet', 'U-Net', 'UNet', 32, 2, False, False, 0.2, 1  , 16], # reference configuration
    ['Only UNet','BE_UNet', 'UNet', 32, 2, False, False, 0.2, 1 , 16], # reference BE_UNet configuration
    ['Only FEM', 'Energy', 'FEM', ' ', ' ', ' ', False, ' ', ' ' , ' '], # reference energy-based method
]

# Column layout of each list_benchmark entry, reused for the CSV header,
# the per-run rows and the aggregation at the end.
CONFIG_COLUMNS = ['Strategy', 'Model', 'First step', 'NIF', 'N_conv',
                  'use cbam', 'use augmentation', 'probability of augmentation',
                  'dataset portion', 'batch size']

RESULT_COLUMNS = ['Input ID',
                  'Number of FEM iterations for the full-FEM strategy',
                  'Number of FEM iterations for the hybrid strategy',
                  'Ratio of FEM iterations (Hybrid / full-FEM)',
                  'Relative compliance error']

RESET_BENCHMARK = False # deletes old benchmark csv file

if RESET_BENCHMARK:
    TYPE_WRITE = 'w'
else: 
    TYPE_WRITE = 'a'

SIZE_LOOP = 100

name_benchmark_file = 'benchmark_architecture.csv'

RUN_BENCHMARK = False  # Set to False to skip the benchmark and only read the results file

#%% Run benchmark

if RUN_BENCHMARK:
    print(f"Running benchmark for {len(list_benchmark)} configurations, {SIZE_LOOP} runs each.")
    total = len(list_benchmark) * SIZE_LOOP
    win = ProgressWindow(total)
    thread = threading.Thread(target=run_window, args=(win,), daemon=True)
    thread.start()


    with open(BASE / 'Software' / 'OT_NN' / 'Pytorch_NN' / 'results' / name_benchmark_file, TYPE_WRITE, newline='') as benchmark_file:

        writer = csv.writer(benchmark_file)
        if RESET_BENCHMARK:
            writer.writerow(CONFIG_COLUMNS + RESULT_COLUMNS)

        for STRATEGY, NETWORK, FIRST_STEP, NIF, N_CONV, USE_CBAM, USE_AUGMENTATION, AUGMENTATION_P, PORTION_DATA, BATCH_SIZE in list_benchmark:

            HIDDEN_LAYERS_MLP=[32,64]
            EMBED_OUT   = 128     # dimension de l'embedding

            if NETWORK=='BE_UNet':
                N_in=1
            else:
                N_in=3


            if STRATEGY == 'Only FEM':
                model = None  # No model needed for the energy-based method

            else:

                aug_tag = f'{int(100*AUGMENTATION_P)}%' if USE_AUGMENTATION else 'False'
                tag = f'{name_file}_NIF={NIF}_{N_CONV}_conv_CBAM={USE_CBAM}_aug={aug_tag}_portion={int(PORTION_DATA*100)}%_batch={BATCH_SIZE}'
            
                # Load model
                if NETWORK=='BE_UNet':
                    model = BE_UNetTopo(
                        nif           = NIF,
                        n_in          = N_in,          # ρ only — tractions via BoundaryEmbedding
                        n_out         = 3,
                        use_cbam      = USE_CBAM,
                        hidden_layers_MLP = HIDDEN_LAYERS_MLP,
                        embed_out     = EMBED_OUT,
                        N_conv=N_CONV,
                    )
                    
                elif NETWORK=='U-Net':
                    tag = f'{name_file}_NIF={NIF}_{N_CONV}_conv_{HIDDEN_LAYERS_MLP}_CBAM={USE_CBAM}_aug={aug_tag}_portion={int(PORTION_DATA*100)}%_batch={BATCH_SIZE}'

                    model = UNetTopo(
                        nif=NIF,
                        n_in=N_in,
                        n_out=3,
                        use_cbam=USE_CBAM,
                        N_conv=N_CONV,
                        )

                else:
                    raise ValueError("Invalid NETWORK value. Choose 'U-net' or 'BE_Unet'.")

                RESULTS_DIR       = RESULTS_ROOT / NETWORK / tag
                ILLUSTRATIONS_DIR = BASE / 'Software' / 'OT_NN' / 'Pytorch_NN' / 'illustrations' / NETWORK / tag
                CHECKPOINT_PATH   = RESULTS_DIR / ('unet_' + name_file + '_checkpoint.pth')
                BEST_PATH         = RESULTS_DIR / ('unet_' + name_file + '_best.pth')



                state_dict = torch.load(
                    BEST_PATH,
                    map_location='cpu'
                )

                model.load_state_dict(state_dict)
                model.eval()
                    
            for ID in range(SIZE_LOOP):
                
                sample_start = IterationSample(IterationDataset(ds_filtre.get_series(ID)), 0)
                if NETWORK == 'Energy':
                    if USE_AUGMENTATION:
                        idx,dist,sample_start = ds_acc_aug.closest_point(sample_start)
                    else:
                        idx,dist,sample_start = ds_acc.closest_point(sample_start)

                List_iterations, List_count_FEM = run_topology_optimization(
                    sample_start,  
                    eng, 
                    model,
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
                    end_FEM=True
                )

                win.increment()

                idx_FEM_sol = IterData_FEM.last_iteration_index[ID]
                FEM_sample  = IterationSample(IterData_FEM, idx_FEM_sol)

                c_FEM  = FEM_sample.c.item()
                c_Unet = List_iterations[-1].c.item()

                err_rel_c  = (c_Unet - c_FEM) / c_FEM
                number_FEM = len(List_count_FEM)
                ds_iter    = IterationDataset(ds_filtre.get_series(ID))

                writer.writerow([
                    STRATEGY, NETWORK, FIRST_STEP, NIF, N_CONV,
                    USE_CBAM, USE_AUGMENTATION, AUGMENTATION_P, PORTION_DATA, BATCH_SIZE,
                    ID, len(ds_iter), number_FEM,
                    number_FEM / len(ds_iter),
                    err_rel_c
                ])

    win.close()

    

#%% Read File Benchmark

list_benchmark = [
    ['Only UNet', 'U-Net', 'UNet', 32, 2, False, False, 0.2, 1  , 16], # reference configuration
    ['Only UNet','BE_UNet', 'UNet', 32, 2, False, False, 0.2, 1 , 16], # reference BE_UNet configuration
    ['Only FEM', 'Energy', 'FEM', ' ', ' ', ' ', False, ' ', ' ' , ' '], # reference energy-based method
]


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


def _config_mask(df, bench):
    """Dtype-robust match of one list_benchmark config against the CSV rows.

    FEM-only rows leave the numeric config columns blank, which forces those
    columns to object (string) dtype on read. A plain `df[cols] == bench`
    would then compare e.g. '32' == 32 and never match, so we coerce numeric
    fields to numbers and compare everything else as trimmed strings.
    """
    mask = pd.Series(True, index=df.index)
    for col, val in zip(CONFIG_COLUMNS, bench):
        if isinstance(val, bool):
            mask &= df[col].astype(str).str.strip() == str(val)
        elif isinstance(val, (int, float)):
            mask &= pd.to_numeric(df[col], errors='coerce') == val
        else:
            # val is a string; blank placeholders (' ' in the FEM-only config)
            # must match blank CSV cells, which pandas reads as NaN. Normalise
            # NaN/whitespace to '' on both sides before comparing.
            col_norm = df[col].where(df[col].notna(), '').astype(str).str.strip()
            mask &= col_norm == str(val).strip()
    return mask


counts = []
for bench in list_benchmark:
    mask = _config_mask(df, bench)
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

plot_FEM_error_c(list_benchmark, Tab_ratio_FEM, Tab_err_rel_c, 'Architecture')


#%%
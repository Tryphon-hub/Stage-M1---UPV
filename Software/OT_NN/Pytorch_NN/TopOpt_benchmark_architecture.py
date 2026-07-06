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
list_benchmark = [
    ['Only UNet', 'U-Net' , 'UNet', 16, 2, False, False, 0.2, 0.5, 16],
    ['Only UNet', 'U-Net' , 'UNet', 32, 2, False, False, 0.2, 1  ,  8],
    ['Only UNet', 'U-Net' , 'UNet', 32, 2, False, False, 0.2, 1  , 16],
    ['Only UNet', 'U-Net' , 'UNet', 32, 2, False, False, 0.2, 1  , 32],
    ['Only UNet', 'U-Net' , 'UNet', 32, 2, False, False, 0.2, 0.5, 16],
    ['Only UNet', 'U-Net' , 'UNet', 32, 2, False, True , 0.5, 1  , 16],
    ['Only UNet', 'U-Net' , 'UNet', 32, 3, False, False, 0.2, 1  , 16],
    ['Only UNet', 'U-Net' , 'UNet', 32, 2, True , False, 0.2, 1  , 16],
    ['Only UNet', 'U-Net' , 'UNet', 32, 2, False, True , 0.2, 1  , 16],
    ['Only UNet', 'U-Net' , 'UNet', 32, 3, True , True , 0.2, 1  , 16],
    ['Only UNet', 'U-Net' , 'UNet', 32, 2, True , True , 0.2, 0.5, 16],
    ['Only UNet','BE_UNet', 'UNet', 32, 2, False, False, 0.2, 1  , 16],
    ['Only UNet','BE_UNet', 'UNet', 32, 2, False, False, 0.2, 0.5, 16],
    ['Only UNet','BE_UNet', 'UNet', 32, 2, True, False , 0.2, 1  , 16],
    ['Only UNet','BE_UNet', 'UNet', 32, 2, False, True , 0.2, 1  , 16],
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

RESET_BENCHMARK = True # deletes old benchmark csv file

if RESET_BENCHMARK:
    TYPE_WRITE = 'w'
else: 
    TYPE_WRITE = 'a'

SIZE_LOOP = 100

name_benchmark_file = 'benchmark_architecture_100.csv'

RUN_BENCHMARK = True  # Set to False to skip the benchmark and only read the results file

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


            if NETWORK == 'U-Net':
                tag = f'{name_file}_NIF={NIF}_{N_CONV}_conv_CBAM={USE_CBAM}_aug={USE_AUGMENTATION if not USE_AUGMENTATION else int(100*AUGMENTATION_P)}%_portion={int(PORTION_DATA*100)}%_batch={BATCH_SIZE}'
            else:
                tag = f'{name_file}_NIF={NIF}_{N_CONV}_conv_{HIDDEN_LAYERS_MLP}_CBAM={USE_CBAM}_aug={USE_AUGMENTATION if not USE_AUGMENTATION else int(100*AUGMENTATION_P)}%_portion={int(PORTION_DATA*100)}%_batch={BATCH_SIZE}'

            RESULTS_DIR       = RESULTS_ROOT / NETWORK / tag
            ILLUSTRATIONS_DIR = BASE / 'Software' / 'OT_NN' / 'Pytorch_NN' / 'illustrations' / NETWORK / tag
            CHECKPOINT_PATH   = RESULTS_DIR / ('unet_' + name_file + '_checkpoint.pth')
            BEST_PATH         = RESULTS_DIR / ('unet_' + name_file + '_best.pth')


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
                model = UNetTopo(
                    nif=NIF,
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
                    window_Unet=5, 
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

                err_rel_c  = abs(c_FEM - c_Unet) / c_FEM
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

plot_FEM_error_c(list_benchmark, Tab_ratio_FEM, Tab_err_rel_c, 'Architecture')



#%%
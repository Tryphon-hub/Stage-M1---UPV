#%% Import libraries
import sys
import matlab.engine
from pathlib import Path
import re

user      = 'laptop'
# user      = 'server'

name_file = 'dataset'
name_data = 'dataset_test'
# NETWORK   = 'BE_Unet'
NETWORK   = 'U-net'



NIF = 32
N_CONV = 3
HIDDEN_LAYERS_MLP = [32,64]
USE_CBAM    = False
EMBED_OUT   = 128



if NETWORK=='BE_Unet':
    N_in=1
else:
    N_in=3

if user == 'laptop':
    BASE = Path(r'C:\Users\maxen\Documents\Stage')
elif user == 'server':
    BASE = Path(r'D:\Maxence\Stage-M1---UPV')

DATA_PATH       = BASE / 'HeavyFiles' / 'data' / (name_data + '.mat')
if NETWORK == 'U-net':
    RESULTS_DIR     = BASE / 'Software' / 'OT_NN' / 'Pytorch_NN' / 'results' / NETWORK / f'{name_file}_{N_CONV}_conv_CBAM={USE_CBAM}'
    ILLUSTRATIONS_DIR = BASE / 'Software' / 'OT_NN' / 'Pytorch_NN' / 'illustrations' / NETWORK / f'{name_file}_{N_CONV}_conv_CBAM={USE_CBAM}'
    CHECKPOINT_PATH = RESULTS_DIR / ('unet_' + name_file + '_checkpoint.pth')
    BEST_PATH       = RESULTS_DIR / ('unet_' + name_file + '_best.pth')
    TB_LOG_DIR      = RESULTS_DIR / ('runs_' + name_file) / ('unet_' + name_file)



else:
    RESULTS_DIR     = BASE / 'Software' / 'OT_NN' / 'Pytorch_NN' / 'results' / NETWORK / f'{name_file}_{N_CONV}_conv_{HIDDEN_LAYERS_MLP}'
    ILLUSTRATIONS_DIR = BASE / 'Software' / 'OT_NN' / 'Pytorch_NN' / 'illustrations' / NETWORK / f'{name_file}_{N_CONV}_conv_{HIDDEN_LAYERS_MLP}'
    CHECKPOINT_PATH = RESULTS_DIR / ('unet_' + name_file + '_checkpoint.pth')
    BEST_PATH       = RESULTS_DIR / ('unet_' + name_file + '_best.pth')
    TB_LOG_DIR      = RESULTS_DIR / ('runs_' + name_file) / ('unet_' + name_file)



sys.path.append(str(BASE / 'Software' / 'OT_NN' / 'Pytorch_NN'))
sys.path.append(str(BASE / 'Software' / 'OT_Functions'))
sys.path.append(str(BASE / 'Software' / 'OT_Software'))

import torch
import numpy as np
from model         import *
from dataset       import *
from topology_utils import *

#%% Constants
IMG_SIZE = 32
PENAL    = 3
RMIN     = 1.5
NGPpL    = 2
NGPpS    = 9
E        = 1000
NU       = 0.3




#%% Load model

if NETWORK=='BE_Unet':
    model = BE_UNetTopo(
        nif           = NIF,
        n_in          = N_in,          # ρ seul — tractions via BoundaryEmbedding
        n_out         = 3,
        use_cbam      = USE_CBAM,
        hidden_layers_MLP = HIDDEN_LAYERS_MLP,
        embed_out     = EMBED_OUT,
        N_conv=N_CONV,
    )
    
elif NETWORK=='U-net':
    model = UNetTopo(
        nif=32, 
        n_in=N_in, 
        n_out=3, 
        use_cbam=True,
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
n_params = sum(p.numel() for p in model.parameters())
print(f"Number of {NETWORK} parameters : {n_params:_}".replace('_',' '))

# n_total = sum(p.numel() for p in model.parameters())
# n_embed = sum(p.numel() for p in model.boundary_embedding.parameters())

# print(f"Total           : {n_total:_}".replace('_', ' '))
# print(f"dont embedding  : {n_embed:_}".replace('_', ' '))
# print(f"reste U-Net     : {n_total - n_embed:_}".replace('_', ' '))


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

mesh_path = str(BASE / 'Software' / 'OT_Software' / 'Square.msh')
eng.eval(f"MeshData = ReadGMSH('{mesh_path}');", nargout=0)
eng.eval("D = DHooks2D(1000, 0.3, 'Plane Stress');", nargout=0)




#%% Plot

# FEM_c, UNet_c = statistical_convergence(
#     List_List_iterations, 
#     IterData_FEM, 
#     NETWORK=NETWORK, 
#     PLOT=True, 
#     TYPE='std'
#     )






#%% One sample

ID_distrib=1
idx_FEM_sol = IterData_FEM.last_iteration_index[ID_distrib]
FEM_sample  = IterationSample(IterData_FEM, idx_FEM_sol)

ds_iter=IterationDataset(ds_filtre.get_series(ID_distrib))

List_iterations, List_count_FEM = run_topology_optimization(
        ds_filtre, 
        ID_distrib, 
        eng, model, 
        N_in = N_in,
        N_max_iterations = 100, 
        # RULE = 'Only UNet',
        # RULE = '10 Unet - 3 FEM',
        RULE = 'Decreasing compliance', 
        # RULE = 'Only FEM',
        # TYPE_FIRST = 'UNet',
        TYPE_FIRST = 'FEM',
        threshold = 0.0,
        N_end_FEM_iterations = 0,
        tol=2e-3,
        end_FEM=True
        )

FEM_c, UNet_c = visualize_convergence(
    List_iterations, 
    ds_iter, 
    List_count_FEM,
    NETWORK=NETWORK,
    PLOT=True,
    SCALE='linear'
    )


compare_NN_FEM(List_iterations[-1], FEM_sample)

density_evolution(List_iterations,List_count_FEM, 1)
#%% Import libraries
import sys
import matlab.engine
from pathlib import Path
import re

# user      = 'laptop'
user      = 'server'

name_file = 'dataset'
name_data = 'dataset_test'
# NETWORK   = 'BE_Unet'
NETWORK   = 'U-net'



NIF = 32
N_CONV = 2
HIDDEN_LAYERS_MLP = [32,64]
USE_CBAM    = True
EMBED_OUT   = 128



if NETWORK=='BE_Unet':
    N_in=1
else:
    N_in=3

if user == 'laptop':
    BASE = Path(r'C:\Users\maxen\Documents\Stage')
elif user == 'server':
    BASE = Path(r'D:\Maxence\Stage-M1---UPV')

DATA_PATH       = BASE / 'HeavyFiles' / 'data' / (name_file + '.mat')
if NETWORK == 'U-net':
    RESULTS_DIR     = BASE / 'Software' / 'OT_NN' / 'Pytorch_NN' / 'results' / NETWORK / f'{name_file}_{N_CONV}_conv'
    ILLUSTRATIONS_DIR = BASE / 'Software' / 'OT_NN' / 'Pytorch_NN' / 'illustrations' / NETWORK / f'{name_file}_{N_CONV}_conv'
    CHECKPOINT_PATH = RESULTS_DIR / ('unet_' + name_file + '_checkpoint.pth')
    BEST_PATH       = RESULTS_DIR / ('unet_' + name_file + '_best.pth')
    TB_LOG_DIR      = RESULTS_DIR / ('runs_' + name_file) / ('unet_' + name_file)



else:
    RESULTS_DIR     = BASE / 'Software' / 'OT_NN' / 'Pytorch_NN' / 'results' / NETWORK / f'{name_file}_{N_CONV}_conv_{HIDDEN_LAYERS_MLP}'
    ILLUSTRATIONS_DIR = BASE / 'Software' / 'OT_NN' / 'Pytorch_NN' / 'illustrations' / NETWORK / f'{name_file}_{N_CONV}_conv_{HIDDEN_LAYERS_MLP}'
    CHECKPOINT_PATH = RESULTS_DIR / ('unet_' + f'{name_file}_{N_CONV}_conv_{HIDDEN_LAYERS_MLP}' + '_checkpoint.pth')
    BEST_PATH       = RESULTS_DIR / ('unet_' + f'{name_file}_{N_CONV}_conv_{HIDDEN_LAYERS_MLP}' + '_best.pth')
    TB_LOG_DIR      = RESULTS_DIR / ('runs_' + f'{name_file}_{N_CONV}_conv_{HIDDEN_LAYERS_MLP}') / ('unet_' + name_file)



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


#%% Topology optimization loop

# List_List_count_FEM = []



# # Only Unet, Only FEM, Decreasing compliance, n Unet - m FEM
# for ID_distrib in range(10):
#     print(f"\n\nStarting topology optimization for distribution ID: {ID_distrib}")
#     List_iterations, List_count_FEM = run_topology_optimization(
#         ds_filtre, 
#         ID_distrib, 
#         eng, model, 
#         N_in = N_in,
#         N_max_iterations = 100, 
#         # RULE = '15 Unet - 5 FEM',
#         # RULE = 'Decreasing compliance', 
#         RULE = 'Only Unet',
#         TYPE_FIRST = 'FEM',
#         threshold = 0.02,
#         N_end_FEM_iterations = 0
#         )

#     List_List_iterations.append(List_iterations)
#     List_List_count_FEM.append(List_count_FEM)

#     idx_FEM_sol = IterData_FEM.last_iteration_index[ID_distrib]
#     FEM_sample  = IterationSample(IterData_FEM, idx_FEM_sol)


#     List_iterations[-1].plot_inputs()
#     FEM_sample.plot_inputs()

#     ds_iter=IterationDataset(ds_filtre.get_series(ID_distrib))

#     FEM_c, UNet_c = visualize_convergence(
#         List_iterations[:-1], 
#         ds_iter, 
#         List_count_FEM,
#         NETWORK=NETWORK,
#         PLOT=True,
#         )


#%% Mean density evolution

# List_Relative_Vol_Frac=[]
# List_mean_densities=[]
# for sample in List_iterations:
#     List_Relative_Vol_Frac.append(sample.Relative_Vol_Frac)
#     List_mean_densities.append(sample.Densities.numpy().mean())

# plt.figure()
# plt.plot(List_Relative_Vol_Frac, label='Relative Volume Fraction')
# plt.plot(List_mean_densities, label='Mean Density')
# plt.xlabel('Iteration')
# plt.ylabel('Value')
# plt.title('Evolution of Relative Volume Fraction and Mean Density')
# plt.legend()
# plt.grid()
# plt.ylim(0, 1) 
# plt.show()




#%% Plot

# FEM_c, UNet_c = statistical_convergence(
#     List_List_iterations, 
#     IterData_FEM, 
#     NETWORK=NETWORK, 
#     PLOT=True, 
#     TYPE='std'
#     )


# ds_iter=IterationDataset(ds_filtre.get_series(ID_distrib))

# for i in range(len(ds_iter)):
#     sample=IterationSample(ds_iter,i)
#     print('sample ',i)
#     sample.plot_inputs()

# for i,sample_i in enumerate(List_iterations):
#     print('sample :', i)
#     sample_i.plot_inputs()




#%% One sample

ID_distrib=0
idx_FEM_sol = IterData_FEM.last_iteration_index[ID_distrib]
FEM_sample  = IterationSample(IterData_FEM, idx_FEM_sol)

ds_iter=IterationDataset(ds_filtre.get_series(ID_distrib))

List_iterations, List_count_FEM = run_topology_optimization(
        ds_filtre, 
        ID_distrib, 
        eng, model, 
        N_in = N_in,
        N_max_iterations = 100, 
        # RULE = '10 Unet - 3 FEM',
        RULE = 'Decreasing compliance', 
        # RULE = 'Only UNet',
        # TYPE_FIRST = 'UNet',
        TYPE_FIRST = 'FEM',
        threshold = 0.02,
        N_end_FEM_iterations = 1,
        )



FEM_c, UNet_c = visualize_convergence(
    List_iterations[:-1], 
    ds_iter, 
    List_count_FEM,
    NETWORK=NETWORK,
    PLOT=True,
    )


compare_NN_FEM(List_iterations[-1], FEM_sample)
#%% Import libraries
import sys
import matlab.engine
from pathlib import Path
import re

user      = 'laptop'
name_file = 'dataset'
name_data = 'dataset_test'
NETWORK   = 'BE_Unet'
# NETWORK   = 'U-net'


if NETWORK=='BE_Unet':
    N_in=1
else:
    N_in=3

if user == 'laptop':
    BASE = Path(r'C:\Users\maxen\Documents\Stage')
elif user == 'server':
    BASE = Path(r'D:\Maxence\Stage-M1---UPV')

DATA_PATH       = BASE / 'HeavyFiles' / 'data' / (name_data + '.mat')
RESULTS_DIR     = BASE / 'Software' / 'OT_NN' / 'Pytorch_NN' / 'results'/ NETWORK 
BEST_PATH       = RESULTS_DIR / name_file / ('unet_' + name_file + '_best.pth')

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
    model = BE_UNetTopo(nif=32, n_in=N_in, n_out=3, use_cbam=True,embed_n1=32, embed_out=64)
elif NETWORK=='U-net':
    model = UNetTopo(nif=32, n_in=N_in, n_out=3, use_cbam=True)
else:
    raise ValueError("Invalid NETWORK value. Choose 'U-net' or 'BE_Unet'.")

state_dict = torch.load(
    BEST_PATH,
    map_location='cpu'
)

model.load_state_dict(state_dict)
model.eval()

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

List_count_FEM = []

ID_distrib = 0


# Only Unet, Only FEM, Decreasing compliance, n Unet - m FEM
List_iterations, count_FEM = run_topology_optimization(
    ds_filtre, 
    ID_distrib, 
    eng, model, 
    N_in=N_in,
    N_max_iterations = 100, 
    RULE=' ', 
    )


List_List_iterations.append(List_iterations)
List_count_FEM.append(count_FEM)

#%% Visualize results


idx_FEM_sol = IterData_FEM.last_iteration_index[ID_distrib]
FEM_sample  = IterationSample(IterData_FEM, idx_FEM_sol)


List_iterations[-1].plot_inputs()
FEM_sample.plot_inputs()


#%% Mean density evolution
List_Relative_Vol_Frac=[]
List_mean_densities=[]
for sample in List_iterations:
    List_Relative_Vol_Frac.append(sample.Relative_Vol_Frac)
    List_mean_densities.append(sample.Densities.numpy().mean())

plt.figure()
plt.plot(List_Relative_Vol_Frac, label='Relative Volume Fraction')
plt.plot(List_mean_densities, label='Mean Density')
plt.xlabel('Iteration')
plt.ylabel('Value')
plt.title('Evolution of Relative Volume Fraction and Mean Density')
plt.legend()
plt.grid()
plt.show()




#%% Plot

# FEM_c, UNet_c = statistical_convergence(
#     List_List_iterations, 
#     IterData_FEM, 
#     NETWORK=NETWORK, 
#     PLOT=True, 
#     TYPE='std'
#     )




ds_iter=IterationDataset(ds_filtre.get_series(ID_distrib))

# for i in range(len(ds_iter)):
#     sample=IterationSample(ds_iter,i)
#     print('sample ',i)
#     sample.plot_inputs()



FEM_c, UNet_c = visualize_convergence(
    List_iterations, 
    ds_iter, 
    NETWORK=NETWORK, 
    )

#%%
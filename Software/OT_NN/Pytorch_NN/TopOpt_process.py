#%% Import libraries
import sys
import matlab.engine
from pathlib import Path

user      = 'laptop'
name_file = 'dataset_macro'
NETWORK   = 'BE_Unet'
# NETWORK   = 'U-net'

ID_distrib = 53


if NETWORK=='BE_Unet':
    N_in=1
else:
    N_in=3

if user == 'laptop':
    BASE = Path(r'C:\Users\maxen\Documents\Stage')
elif user == 'server':
    BASE = Path(r'D:\Maxence\Stage-M1---UPV')



DATA_PATH       = BASE / 'HeavyFiles' / 'data' / (name_file + '.mat')
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
ds_iter = IterationDataset(ds_filtre.get_series(ID_distrib))


#%% Select one sample
IDX=0

sample = IterationSample(ds_iter, IDX)

#%% Start MATLAB engine
eng = matlab.engine.start_matlab()
eng.addpath(str(BASE / 'Software' / 'OT_Functions'))
eng.addpath(str(BASE / 'Software' / 'OT_Software'))

mesh_path = str(BASE / 'Software' / 'OT_Software' / 'Square.msh')
eng.eval(f"MeshData = ReadGMSH('{mesh_path}');", nargout=0)
eng.eval("D = DHooks2D(1000, 0.3, 'Plane Stress');", nargout=0)

#%% Run topology optimization
List_Relative_Vol_Frac=[sample.Relative_Vol_Frac]
List_mean_densities = [sample.Densities.numpy().mean()]

next_sample      = GenTopology(sample, eng, model, TYPE='UNet',N_in=N_in)

List_Relative_Vol_Frac.append(next_sample.Relative_Vol_Frac)
List_mean_densities.append(next_sample.Densities.numpy().mean())

List_iterations  = [sample, next_sample]
i                = 1
N_max_iterations = 100

while i < N_max_iterations and not is_converged(sample, next_sample, tol=1e-4):
    sample      = next_sample
    next_sample = GenTopology(sample, eng, model, TYPE='UNet',N_in=N_in)
    List_Relative_Vol_Frac.append(next_sample.Relative_Vol_Frac)
    List_mean_densities.append(next_sample.Densities.numpy().mean())
    List_iterations.append(next_sample)
    i += 1

#%% Visualize results
List_iterations[-1].plot_inputs()

idx_FEM_sol = ds_iter.last_iteration_index[IDX]
FEM_sample  = IterationSample(ds_iter, idx_FEM_sol)
FEM_sample.plot_inputs()


#%% Mean density evolution

plt.figure()
plt.plot(List_Relative_Vol_Frac, label='Relative Volume Fraction')
plt.plot(List_mean_densities, label='Mean Density')
plt.xlabel('Iteration')
plt.ylabel('Value')
plt.title('Evolution of Relative Volume Fraction and Mean Density')
plt.legend()
plt.grid()
plt.show()


#%% Compliance convergence

visualize_convergence(List_iterations, ds_iter)


#%% Import libraries
import sys
import matlab.engine
from pathlib import Path

user      = 'laptop'
name_file = 'dataset_macro'
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

sys.path.append(str(BASE / 'Software' / 'OT_NN' / NETWORK))
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
    state_dict = torch.load(
        BASE / 'Software' / 'OT_NN' / NETWORK / 'results' / name_file / ('unet_'+name_file+'_best.pth'),
        map_location='cpu'
    )
else:
    model = UNetTopo(nif=32, n_in=N_in, n_out=3, use_cbam=True)
    state_dict = torch.load(
        BASE / 'Software' / 'OT_NN' / NETWORK / 'results' / name_file / ('unet_'+name_file+'_best.pth'),
        map_location='cpu'
    )

model.load_state_dict(state_dict)
model.eval()

#%% Load dataset
data    = load_mat(BASE / 'HeavyFiles' / 'data' / (name_file + '.mat'))
ds_base = Dataset_TopOpt(data)
ds_iter = IterationDataset(ds_base)

#%% Check dataset

List_index=[]
List_last_j=[]
ref=0
for i,j in ds_iter.index:
    if i>ref:
        ref=i
        List_index.append(i)
        List_last_j.append(j-1)



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
next_sample      = GenTopology(sample, eng, model, TYPE='UNet',N_in=N_in)
List_iterations  = [sample, next_sample]
i                = 1
N_max_iterations = 100

while i < N_max_iterations and not is_converged(sample, next_sample):
    sample      = next_sample
    next_sample = GenTopology(sample, eng, model, TYPE='UNet',N_in=N_in)
    List_iterations.append(next_sample)
    i += 1

#%% Visualize results
List_iterations[-1].plot_inputs()

idx_FEM_sol = ds_iter.last_iteration_index[IDX]
FEM_sample  = IterationSample(ds_iter, idx_FEM_sol)
FEM_sample.plot_inputs()

#%% Plot outputs
print("Plotting UNet outputs")

for i in range(len(List_iterations)-2):
    print(f"Iteration {i+1}/{len(List_iterations)-2}")
    List_iterations[i].plot_inputs()
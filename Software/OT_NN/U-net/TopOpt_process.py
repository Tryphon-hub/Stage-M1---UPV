#%% Import libraries
import sys

user='server'


if user=='laptop':
    path=r'C:\Users\maxen\Documents\Stage'
elif user=='server':
    path=r'D:\Maxence\Stage-M1---UPV'
    
sys.path.append(path+r'\Software\OT_NN')
sys.path.append(path+r'\Software\OT_Software')
sys.path.append(path+r'\Software\OT_Functions')





import torch
import numpy as np
from model import UNetTopo
from dataset import Dataset_TopOpt, IterationDataset, IterationSample, load_mat
from pathlib import Path
import scipy.io
from topology_utils import *

#%% Constants
IMG_SIZE = 32
PENAL    = 3
RMIN     = 1.5
NGPpL    = 2   # number of 1D integration points
NGPpS    = 9   # number of 2D integration points
E        = 1000
NU       = 0.3
NGPpL = 2   # number of integration points 1D
NGPpS = 9   # number of integration points 2D


#%% Load dataset

data=load_mat(path+r'\HeavyFiles\data\dataset_macro.mat')


ds_base  = Dataset_TopOpt(data)
ds_iter  = IterationDataset(ds_base)

#%% Select one sample

IDX    = 25
sample = IterationSample(ds_iter, IDX)


# ── Start MATLAB engine ────────────────────────────────────────────
eng = matlab.engine.start_matlab()

eng.addpath(path+r'\Software\OT_Functions')
eng.addpath(path+r'\Software\OT_Software')

# ── Load mesh and material matrix ──────────────────────────────────
eng.eval("MeshData = ReadGMSH(path+r'\Software\OT_Software\Square.msh');", nargout=0)
eng.eval("D = DHooks2D(1000, 0.3, 'Plane Stress');", nargout=0)

# ── Load U-Net model ───────────────────────────────────────────────
model = UNetTopo(nif=32, n_in=3, n_out=3, use_cbam=True)
state_dict = torch.load(
    path+r'\Software\OT_NN\U-net\results\unet_topo_best.pth',
    map_location='cpu'
)

model.load_state_dict(state_dict)
model.eval()

# ── Load dataset and select initial sample ─────────────────────────
data    = load_mat(path+r'\HeavyFiles\data\dataset_test.mat')
ds_base = Dataset_TopOpt(data)
ds_iter = IterationDataset(ds_base)
sample  = IterationSample(ds_iter, idx=IDX)  # first iteration of first sample
next_sample = GenTopology(sample, eng, model, TYPE='UNet')  # Generate first topology using U-Net predictions
i=1
N_max_iterations=100
# ── Run one topology optimization process ────────────────────────

List_iterations=[sample, next_sample]

while i<N_max_iterations and not is_converged(sample, next_sample):
    sample = next_sample
    next_sample = GenTopology(sample, eng, model, TYPE='UNet')
    List_iterations.append(next_sample)
    i+=1


#%% Visualize results

List_iterations[-1].plot_inputs()

idx_FEM_sol = ds_iter.last_iteration_index[IDX]
FEM_sample = IterationSample(ds_iter, idx_FEM_sol)
FEM_sample.plot_inputs()

#%% 
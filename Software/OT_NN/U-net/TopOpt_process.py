#%% Import libraries
import torch
import numpy as np
from model import UNetTopo
from dataset import Dataset_TopOpt, IterationDataset, IterationSample, load_mat
from pathlib import Path
import scipy.io

#%% Load model
model = UNetTopo(nif=32, n_in=3, n_out=3, use_cbam=True)
state_dict = torch.load(
    r'C:\Users\maxen\Documents\Stage\Software\OT_NN\U-net\results\unet_topo_best.pth',
    map_location='cpu'
)
model.load_state_dict(state_dict)
model.eval()

#%% Load dataset
data     = load_mat(r'C:\Users\maxen\Documents\Stage\HeavyFiles\data\dataset_test.mat')
ds_base  = Dataset_TopOpt(data)
ds_iter  = IterationDataset(ds_base)

#%% Select one sample
idx    = 0  
sample = IterationSample(ds_iter, idx)


#%% Build network input
img_size = 32

# Channel 1 : current density field
rho = sample.Densities.squeeze().numpy().reshape(img_size, img_size)  # (32, 32)

# Channels 2-3 : traction fields tx, ty interpolated on the boundary
tx,ty = sample.get_traction_distribution()  # 2 x (1, 32, 32)


# Stack channels and add batch dimension
input_tensor = np.stack([rho, tx, ty], axis=0)              # (3, 32, 32)
input_tensor = torch.tensor(input_tensor).float().unsqueeze(0)  # (1, 3, 32, 32)

#%% Predict stress fields
with torch.no_grad():
    output = model(input_tensor)  # (1, 3, 32, 32)

sigma_x = output[0, 0].numpy()  # (32, 32) — normal stress σx
sigma_y = output[0, 1].numpy()  # (32, 32) — normal stress σy
tau_xy  = output[0, 2].numpy()  # (32, 32) — shear stress τxy


#%% 
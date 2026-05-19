''' 
Adaptated from GenTopology.mat by Maxence Barberet-Pinto
------------------------------------------------------
This script uses a trained U-Net to predict stress fields
and return the density image for the next topology optimization iteration.
'''

#%% Libraries
import torch
import numpy as np
import matlab.engine
from model import UNetTopo
from pathlib import Path
from dataset import Dataset_TopOpt, IterationDataset, IterationSample, load_mat
import scipy.io
import matplotlib.pyplot as plt

#%% Constants
IMG_SIZE = 32
PENAL    = 3
RMIN     = 1.5

#%% Functions

def predict_stress(model, sample):
    """
    Predict stress fields using the U-Net.
    sample : IterationSample containing Densities and Tractions
    Returns sigma_x, sigma_y, tau_xy and updates sample.UNet_Stress.
    """
    # Channel 1 : current density field
    rho = sample.Densities.squeeze().numpy().reshape(IMG_SIZE, IMG_SIZE)  # (32, 32)

    # Channels 2-3 : traction fields tx, ty interpolated on the boundary
    tx_ty = sample.get_traction_distribution()  # (2, 32, 32)
    tx, ty = tx_ty[0], tx_ty[1]

    # Stack channels and add batch dimension
    input_tensor = np.stack([rho, tx, ty], axis=0)                    # (3, 32, 32)
    input_tensor = torch.tensor(input_tensor).float().unsqueeze(0)    # (1, 3, 32, 32)

    # Predict stress fields
    with torch.no_grad():
        output = model(input_tensor)  # (1, 3, 32, 32)

    sigma_x = output[0, 0].numpy().flatten()  # (NumEls,) — normal stress σx
    sigma_y = output[0, 1].numpy().flatten()  # (NumEls,) — normal stress σy
    tau_xy  = output[0, 2].numpy().flatten()  # (NumEls,) — shear stress τxy

    # Store predicted stress in sample

    fill=np.zeros_like(sigma_x)  # (NumEls,) — zero for missing components σz, τxz, τyz
    sample.UNet_Stress = torch.tensor(
        np.column_stack([sigma_x, sigma_y, fill, tau_xy, fill, fill])
    ).float()  # (NumEls, 6)

    return sigma_x, sigma_y, tau_xy


def GenTopology(sample: IterationSample, eng, model) -> IterationSample:
    """
    Compute one topology optimization iteration using the U-Net for stress prediction.
    Takes an IterationSample, returns the updated IterationSample.

    Parameters
    ----------
    sample : IterationSample — current iteration state
    eng    : matlab.engine   — MATLAB engine instance
    model  : UNetTopo        — trained U-Net model

    Returns
    -------
    next_sample : IterationSample — next iteration state
    """
    # Extract current density
    Rel_Density = sample.Densities.squeeze().numpy().flatten()  # (NumEls,)

    # ── Predict stress fields with U-Net ───────────────────────────────
    sigma_x, sigma_y, tau_xy = predict_stress(model, sample)
    Stress = sample.UNet_Stress.numpy()  # (NumEls, 6) — σx, σy, τxy

    # ── Pass variables to MATLAB workspace ────────────────────────────
    eng.workspace['Rel_Density'] = matlab.double(Rel_Density.tolist())
    eng.workspace['Stress_py']   = matlab.double(Stress.tolist())

    # ── Objective function and sensitivities ──────────────────────────
    eng.eval(
        f"[c, dc, ce, InfVol] = Opt_Stress(Rel_Density, Stress_py, D, {PENAL}, MeshData, true, true, 2, 4);",
        nargout=0
    )
    c      = float(eng.workspace['c'])
    dc     = np.array(eng.workspace['dc']).flatten()
    InfVol = np.array(eng.workspace['InfVol']).flatten()
    Vol    = float(np.sum(InfVol))

    # ── Sensitivity filtering ──────────────────────────────────────────
    CharSize = (Vol / IMG_SIZE**2) ** 0.5
    Rmin_aux = RMIN * CharSize

    eng.workspace['Rel_Density'] = matlab.double(Rel_Density.reshape(-1, 1).tolist())
    eng.workspace['dc']          = matlab.double(dc.reshape(-1, 1).tolist())
    eng.workspace['InfVol']      = matlab.double(InfVol.reshape(-1, 1).tolist())
    eng.workspace['Rmin_aux']    = float(Rmin_aux)

    eng.eval("dc     = Filter(Rel_Density, dc, Rmin_aux, MeshData, true, 2);", nargout=0)
    eng.eval("InfVol = Filter(Rel_Density, InfVol, Rmin_aux, MeshData, true, 2);", nargout=0)

    dc     = np.array(eng.workspace['dc']).flatten()
    InfVol = np.array(eng.workspace['InfVol']).flatten()
    InfVol = InfVol * Vol / np.sum(InfVol)

    # ── Optimality criteria update ─────────────────────────────────────
    eng.workspace['dc']     = matlab.double(dc.tolist())
    eng.workspace['InfVol'] = matlab.double(InfVol.tolist())
    eng.workspace['VolFrac']= float(sample.Relative_Vol_Frac)

    eng.eval("New_Rel_Density = OC(Rel_Density, dc, InfVol, VolFrac);", nargout=0)
    New_Rel_Density = np.array(eng.workspace['New_Rel_Density']).flatten()

    # ── Build next IterationSample ─────────────────────────────────────
    next_sample                   = IterationSample.__new__(IterationSample)
    next_sample.Tractions         = sample.Tractions
    next_sample.Densities         = torch.tensor(New_Rel_Density).float().unsqueeze(0)
    next_sample.Relative_Vol_Frac = sample.Relative_Vol_Frac
    next_sample.Stress            = torch.tensor(Stress).float()
    next_sample.c                 = torch.tensor(float(c)).float()
    next_sample.FEMc              = torch.tensor(0.0).float()
    next_sample.NumIts            = sample.NumIts
    next_sample.ItsFull           = sample.ItsFull
    next_sample.TEnd              = sample.TEnd

    return next_sample


#%% Example usage
# if __name__ == '__main__':

#%% ── Start MATLAB engine ────────────────────────────────────────────
eng = matlab.engine.start_matlab()
eng.addpath(r'C:\Users\maxen\Documents\Stage\Software\OT_Functions')
eng.addpath(r'C:\Users\maxen\Documents\Stage\Software\OT_Software')

#%% ── Load mesh and material matrix ──────────────────────────────────
eng.eval("MeshData = ReadGMSH('C:\\Users\\maxen\\Documents\\Stage\\Software\\OT_Software\\Square.msh');", nargout=0)
eng.eval("D = DHooks2D(1000, 0.3, 'Plane Stress');", nargout=0)

#%% ── Load U-Net model ───────────────────────────────────────────────
model = UNetTopo(nif=32, n_in=3, n_out=3, use_cbam=True)
state_dict = torch.load(
    r'C:\Users\maxen\Documents\Stage\Software\OT_NN\U-net\results\unet_topo_best.pth',
    map_location='cpu'
)
model.load_state_dict(state_dict)
model.eval()

#%% ── Load dataset and select initial sample ─────────────────────────
data    = load_mat(r'C:\Users\maxen\Documents\Stage\HeavyFiles\data\dataset_test.mat')
ds_base = Dataset_TopOpt(data)
ds_iter = IterationDataset(ds_base)
sample  = IterationSample(ds_iter, idx=0)  # first iteration of first sample

#%% ── Run one topology optimization iteration ────────────────────────
next_sample = GenTopology(sample, eng, model)

print(f'Compliance : {next_sample.c.item():.4f}')
print(f'Density range : [{next_sample.Densities.min():.3f}, {next_sample.Densities.max():.3f}]')

#%%

sample.plot_inputs()
sample.plot_outputs('FEM')
sample.plot_outputs('UNet')

# %%

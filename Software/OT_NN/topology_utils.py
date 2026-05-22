''' 
Adaptated from GenTopology.mat by Maxence Barberet-Pinto
------------------------------------------------------
This script uses a trained U-Net to predict stress fields
and return the density image for the next topology optimization iteration.
'''

#%% Libraries

import sys

sys.path.append(r'C:\Users\maxen\Documents\Stage\Software\OT_NN\U-net')
sys.path.append(r'C:\Users\maxen\Documents\Stage\Software\OT_Software')
sys.path.append(r'C:\Users\maxen\Documents\Stage\Software\OT_Functions')

import torch
import numpy as np
import matlab.engine
from model import *
from pathlib import Path
from dataset import Dataset_TopOpt, IterationDataset, IterationSample, load_mat
import scipy.io
import matplotlib.pyplot as plt

#%% Constants
IMG_SIZE = 32
PENAL    = 3
RMIN     = 1.5

#%% Functions

def predict_stress(model, sample, N_in=3):
    """
    Predict stress fields using the U-Net.
    Handles both UNetTopo (N_in=3) and BE_UNetTopo (N_in=1) architectures.
    """
    # Channel 1 : current density field
    rho = sample.Densities.squeeze().numpy().reshape(IMG_SIZE, IMG_SIZE)

    # BE_UNetTopo — ρ seul + nodes
    if N_in == 1:
        rho_tensor = torch.tensor(rho).float().unsqueeze(0).unsqueeze(0)  # [1, 1, 32, 32]

        T     = sample.Tractions.squeeze().numpy()   # [2, 8]
        nodes = torch.tensor(T.reshape(1, 16)).float()  # [1, 16]

        with torch.no_grad():
            output = model(rho_tensor, nodes)   # [1, 3, 32, 32]

    # UNetTopo — ρ + tx + ty
    else:
        tx_ty = sample.get_traction_distribution()   # [2, 32, 32]
        tx, ty = tx_ty[0], tx_ty[1]

        input_tensor = np.stack([rho, tx, ty], axis=0)
        input_tensor = torch.tensor(input_tensor).float().unsqueeze(0)   # [1, 3, 32, 32]

        with torch.no_grad():
            output = model(input_tensor)   # [1, 3, 32, 32]

    sigma_x = output[0, 0].numpy().flatten()
    sigma_y = output[0, 1].numpy().flatten()
    tau_xy  = output[0, 2].numpy().flatten()

    fill = np.zeros_like(sigma_x)
    sample.UNet_Stress = torch.tensor(
        np.column_stack([sigma_x, sigma_y, fill, tau_xy, fill, fill])
    ).float()

    return sigma_x, sigma_y, tau_xy


def predict_stress_FEM(eng, sample):
    """
    Compute stress fields using the full Finite Element Method (SIMP method).
    sample : IterationSample containing Densities and Tractions
    Updates sample.FEM_Stress with the computed stress tensor.
    """
    # Extract current state
    Rel_Density = sample.Densities.squeeze().numpy().flatten()  # (NumEls,)
    Tractions   = sample.Tractions.squeeze().numpy()            # (2, 8)

    # Pass variables to MATLAB workspace
    eng.workspace['Rel_Density'] = matlab.double(Rel_Density.reshape(-1, 1).tolist())
    eng.workspace['Tractions']   = matlab.double(Tractions.tolist())

    # Solve finite element system KU = F
    eng.eval(
        f"Sol = SolveFE(MeshData, Rel_Density.^{PENAL}, {NGPpS}, {NGPpL}, D, Tractions, true, true);",
        nargout=0
    )

    # Evaluate stress fields
    eng.eval(
        "Stress_FEM = EvalStress(MeshData.Surf.Topology, MeshData.XYZ, Rel_Density.^1, D, Sol, 2, true, true, 'Plane Stress', 1000, 0.3);",
        nargout=0
    )

    # Retrieve stress from MATLAB
    Stress_FEM = np.array(eng.workspace['Stress_FEM'])  # (NumEls, 6)

    # Store FEM stress in sample
    sample.FEM_Stress = torch.tensor(Stress_FEM).float()  # (NumEls, 6)

    return Stress_FEM


def GenTopology(sample: IterationSample, eng, model, TYPE, N_in=3) -> IterationSample:
    """
    Compute one topology optimization iteration using the U-Net for stress prediction.
    Takes an IterationSample, returns the updated IterationSample.

    Parameters
    ----------
    sample : IterationSample — current iteration state
    eng    : matlab.engine   — MATLAB engine instance
    model  : UNetTopo        — trained U-Net model
    TYPE   : str             — type of optimization (e.g., 'UNet', 'FEM')
    N_in   : int             — number of input channels for the U-Net
    Returns
    -------
    next_sample : IterationSample — next iteration state
    """
    # Extract current density
    Rel_Density = sample.Densities.squeeze().numpy().flatten()  # (NumEls,)

    # ── Predict stress fields with U-Net ───────────────────────────────
    if TYPE == 'UNet':
        predict_stress(model, sample,N_in=N_in)
        Stress = sample.UNet_Stress.numpy()  # (NumEls, 6) — σx, σy, τxy
    elif TYPE == 'FEM':
        Stress = predict_stress_FEM(eng, sample)  # (NumEls, 6) — σx, σy, τxy
    else:
        raise ValueError("Invalid TYPE. Must be 'UNet' or 'FEM'.")

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

    eng.eval("InfVol = InfVol(:);",      nargout=0) # reshape to column vector
    eng.eval("dc     = dc(:);",          nargout=0)
    eng.eval("Rel_Density = Rel_Density(:);", nargout=0)
    eng.eval("New_Rel_Density = OC(Rel_Density, dc, InfVol, VolFrac);", nargout=0)

    New_Rel_Density = np.array(eng.workspace['New_Rel_Density']).flatten()
    
    # ── Build next IterationSample ─────────────────────────────────────
    next_sample                   = IterationSample.__new__(IterationSample)
    next_sample.Tractions         = sample.Tractions
    next_sample.Densities         = torch.tensor(New_Rel_Density).float().unsqueeze(0)
    next_sample.Relative_Vol_Frac = sample.Relative_Vol_Frac
    next_sample.FEM_Stress        = torch.tensor(Stress).float()
    next_sample.UNet_Stress       = None # will be computed in the next iteration
    next_sample.c                 = torch.tensor(float(c)).float()
    next_sample.FEMc              = torch.tensor(0.0).float()
    next_sample.NumIts            = sample.NumIts
    next_sample.ItsFull           = sample.ItsFull
    next_sample.TEnd              = sample.TEnd

    return next_sample


def is_converged(sample_a: IterationSample, sample_b: IterationSample, tol=1e-3) -> bool:
    '''
    Check convergence between two IterationSamples based on compliance change.
    '''
    c_a = sample_a.c.item()
    c_b = sample_b.c.item()
    if abs(c_b - c_a) / abs(c_a) < tol:
        return True
    else:
        return False
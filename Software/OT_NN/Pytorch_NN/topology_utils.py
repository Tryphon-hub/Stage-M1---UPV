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
    eng.eval("InfVol = Filter(ones(size(Rel_Density)), InfVol, Rmin_aux, MeshData, true, 2);", nargout=0)

    dc     = np.array(eng.workspace['dc']).flatten()
    InfVol = np.array(eng.workspace['InfVol']).flatten()
    # ── CRITICAL: Rescale InfVol to maintain volume constraint ─────────
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
    

###############################################################################
#                       Error quantification                                  #
###############################################################################


def extract_stress_maps(y):
    sx  = y[0, 0].cpu().numpy()   # σx  [32, 32]
    sy  = y[0, 1].cpu().numpy()   # σy  [32, 32]
    txy = y[0, 2].cpu().numpy()   # τxy [32, 32]
    return sx, sy, txy


def visualize_in_out(y_true, y_pred):
        
    # Inputs
    rho   = x[0, 0].cpu().numpy()        # density        [32, 32]
    tx    = x[0, 1].cpu().numpy()        # traction tx    [32, 32]
    ty    = x[0, 2].cpu().numpy()        # traction ty    [32, 32]


    # Ground truth outputs
    sx_true, sy_true, txy_true = extract_stress_maps(y_true)

    # Predicted outputs
    sx_pred, sy_pred, txy_pred = extract_stress_maps(y_pred)


    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    components = [
        (sx_true,  sx_pred,  'σx'),
        (sy_true,  sy_pred,  'σy'),
        (txy_true, txy_pred, 'τxy'),
    ]

    # Shared scale across all components
    vmin = min(sx_true.min(), sx_pred.min(), sy_true.min(), sy_pred.min(),
            txy_true.min(), txy_pred.min())
    vmax = max(sx_true.max(), sx_pred.max(), sy_true.max(), sy_pred.max(),
            txy_true.max(), txy_pred.max())

    for col, (true, pred, name) in enumerate(components):
        axes[0, col].imshow(true, cmap='RdBu', origin='lower', vmin=vmin, vmax=vmax)
        axes[0, col].set_title(f'GT {name}')
        axes[0, col].axis('off')

        im = axes[1, col].imshow(pred, cmap='RdBu', origin='lower', vmin=vmin, vmax=vmax)
        axes[1, col].set_title(f'Pred {name}')
        axes[1, col].axis('off')

    # Single colorbar on the right of the entire figure
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])   # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax)

    plt.suptitle(f'Distribution i={i}, iteration j={j}', fontsize=14)
    plt.show()


#%%  Error maps

def visualize_error(y_true, y_pred, TYPE):

    # Ground truth outputs
    sx_true, sy_true, txy_true = extract_stress_maps(y_true)

    # Predicted outputs
    sx_pred, sy_pred, txy_pred = extract_stress_maps(y_pred)

    if TYPE=='MAE':
        err_sx  = np.abs(sx_true - sx_pred)
        err_sy  = np.abs(sy_true - sy_pred)
        err_txy = np.abs(txy_true - txy_pred)
    
    if TYPE=='MSE':
        err_sx  = (sx_true - sx_pred) ** 2
        err_sy  = (sy_true - sy_pred) ** 2
        err_txy = (txy_true - txy_pred) ** 2

    if TYPE=='SMAPE':
        err_sx  = 2 * np.abs(sx_true - sx_pred) / (np.abs(sx_true) + np.abs(sx_pred) + 1e-6)
        err_sy  = 2 * np.abs(sy_true - sy_pred) / (np.abs(sy_true) + np.abs(sy_pred) + 1e-6)
        err_txy = 2 * np.abs(txy_true - txy_pred) / (np.abs(txy_true) + np.abs(txy_pred) + 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    components = [
        ( err_sx,  'σx'),
        ( err_sy,  'σy'),
        ( err_txy, 'τxy'),
    ]

    vmin = 0
    vmax = max( err_sx.max(),  err_sy.max(),  err_txy.max())

    for col, (err, name) in enumerate(components):
        im = axes[col].imshow(err, cmap='hot', origin='lower', vmin=vmin, vmax=vmax)
        axes[col].set_title(f'error {name}')
        axes[col].axis('off')

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.suptitle(f'{TYPE} — Distribution i={i}, iteration j={j}', fontsize=14)
    plt.show()


# %% Error calculation

def ErrorMetrics_1D(y_true, y_pred, TYPE):
    sx=y_true[0, 0].cpu().numpy()        # σx  [32, 32]
    sy=y_true[0, 1].cpu().numpy()        # σy  [32, 32]
    txy=y_true[0, 2].cpu().numpy()       # τxy [32, 32]

    sx_p=y_pred[0, 0].cpu().numpy()      # σx  [32, 32]
    sy_p=y_pred[0, 1].cpu().numpy()      # σy  [32, 32]
    txy_p=y_pred[0, 2].cpu().numpy()     # τxy [32, 32]


    if TYPE=='MSE':
        err_sx = np.mean((sx - sx_p) ** 2)
        err_sy = np.mean((sy - sy_p) ** 2)
        err_txy = np.mean((txy - txy_p) ** 2)
        return err_sx, err_sy, err_txy
    
    elif TYPE=='MAE':
        err_sx = np.mean(np.abs(sx - sx_p))
        err_sy = np.mean(np.abs(sy - sy_p))
        err_txy = np.mean(np.abs(txy - txy_p))
        return err_sx, err_sy, err_txy
    
    elif TYPE=='SMAPE':
        err_sx = 2 * np.mean(np.abs(sx - sx_p) / (np.abs(sx) + np.abs(sx_p) + 1e-6))
        err_sy = 2 * np.mean(np.abs(sy - sy_p) / (np.abs(sy) + np.abs(sy_p) + 1e-6))
        err_txy = 2 * np.mean(np.abs(txy - txy_p) / (np.abs(txy) + np.abs(txy_p) + 1e-6))
        return err_sx, err_sy, err_txy

    elif TYPE=='R2':
        err_sx = 1 - np.sum((sx - sx_p) ** 2) / np.sum((sx - np.mean(sx)) ** 2)
        err_sy = 1 - np.sum((sy - sy_p) ** 2) / np.sum((sy - np.mean(sy)) ** 2)
        err_txy = 1 - np.sum((txy - txy_p) ** 2) / np.sum((txy - np.mean(txy)) ** 2)
        return err_sx, err_sy, err_txy

# %% Error calculation with kernel

def convolve(img, kernel: int, pad: bool, strides: (int, int)):
    """
    Applies a 2D convolution with a ones kernel.

    Parameters
    ----------
    img         : np.ndarray [H, W]  — input image
    kernel      : np.ndarray         — square kernel
    pad         : bool               — if True, pads input to preserve spatial dimensions
    strides     : (int, int)         — (stride_h, stride_w)

    Returns
    -------
    np.ndarray — output feature map
    """
    
    kernel_size = kernel.shape[0]

    kH, kW = kernel.shape
    sH, sW = strides

    if pad:
        pad_size = kernel_size // 2
        img = np.pad(img, pad_size, mode='constant', constant_values=0)

    H_pad, W_pad = img.shape
    H_out = (H_pad - kH) // sH + 1
    W_out = (W_pad - kW) // sW + 1

    out = np.zeros((H_out, W_out), dtype=np.float32)

    for i in range(H_out):
        for j in range(W_out):
            region    = img[i*sH : i*sH+kH, j*sW : j*sW+kW]
            out[i, j] = np.sum(region * kernel)

    return out


def ErrorMetrics_Kernel(y_true, y_pred, kernel_size:int, pad:bool, strides:(int, int), TYPE:str,plot=True):

    # Ground truth outputs
    sx_true, sy_true, txy_true = extract_stress_maps(y_true)

    # Predicted outputs
    sx_pred, sy_pred, txy_pred = extract_stress_maps(y_pred)


    if TYPE=='MAE':
        err_sx  = np.abs(sx_true - sx_pred)
        err_sy  = np.abs(sy_true - sy_pred)
        err_txy = np.abs(txy_true - txy_pred)
    
    if TYPE=='MSE':
        err_sx  = (sx_true - sx_pred) ** 2
        err_sy  = (sy_true - sy_pred) ** 2
        err_txy = (txy_true - txy_pred) ** 2

    if TYPE=='SMAPE':
        err_sx  = 2 * np.abs(sx_true - sx_pred) / (np.abs(sx_true) + np.abs(sx_pred) + 1e-6)
        err_sy  = 2 * np.abs(sy_true - sy_pred) / (np.abs(sy_true) + np.abs(sy_pred) + 1e-6)
        err_txy = 2 * np.abs(txy_true - txy_pred) / (np.abs(txy_true) + np.abs(txy_pred) + 1e-6)

    kernel=np.ones((kernel_size, kernel_size)) # / (kernel_size ** 2)
    
    err_sx_k  = convolve(err_sx, kernel, pad, strides)
    err_sy_k  = convolve(err_sy, kernel, pad, strides)
    err_txy_k = convolve(err_txy, kernel, pad, strides)


    if plot==True:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        components = [
            ( err_sx_k,  'σx'),
            ( err_sy_k,  'σy'),
            ( err_txy_k, 'τxy'),
        ]

        vmin = 0
        vmax = max( err_sx_k.max(),  err_sy_k.max(),  err_txy_k.max())

        for col, (err, name) in enumerate(components):
            im = axes[col].imshow(err, cmap='hot', origin='lower', vmin=vmin, vmax=vmax)
            axes[col].set_title(f'error {name}')
            axes[col].axis('off')

        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax)

        plt.suptitle(f'{TYPE} — kernel {kernel_size}x{kernel_size} — Distribution i={i}, iteration j={j}', fontsize=14)
        plt.show()

        
    return np.mean(err_sx_k), np.mean(err_sy_k), np.mean(err_txy_k)   


# %% Convergence study
def visualize_convergence(List_Iterations_Unet, IterationDataset_FEM):
    f_text=1.25 # text size multiplicator 


    FEM_c=[]
    UNet_c=[]

    for i in range(len(IterationDataset_FEM)):
        sample=IterationSample(IterationDataset_FEM, i)
        FEM_c.append([i,sample.c])

    for i,sample in enumerate(List_Iterations_Unet):
        UNet_c.append([i, sample.c])

    FEM_c=np.array(FEM_c)
    UNet_c=np.array(UNet_c)

    plt.figure(figsize=(10, 6))
    plt.plot(FEM_c[:, 0], FEM_c[:, 1], 'o-', linewidth=2.5, markersize=7, label='FEM')
    plt.plot(UNet_c[:, 0], UNet_c[:, 1], 's-', linewidth=2.5, markersize=7, label='U-Net')
    plt.xlabel('Iterations', fontsize=f_text*14,)
    plt.ylabel('Compliance', fontsize=f_text*14, )
    plt.title('Compliance Convergence: FEM vs U-Net', fontsize=f_text*16, )
    plt.legend(fontsize=f_text*13)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=f_text*12)
    plt.yticks(fontsize=f_text*12)
    plt.tight_layout()
    plt.show()

    return FEM_c, UNet_c
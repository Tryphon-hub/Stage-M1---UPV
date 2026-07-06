#%% Libraries

import sys
import re

import tkinter as tk
import time
import threading


from zmq import TYPE

sys.path.append(r'C:\Users\maxen\Documents\Stage\Software\OT_NN\U-net')
sys.path.append(r'C:\Users\maxen\Documents\Stage\Software\OT_Software')
sys.path.append(r'C:\Users\maxen\Documents\Stage\Software\OT_Functions')

import torch
import numpy as np
import matlab.engine
from model import *
from pathlib import Path
from dataset import *
import scipy.io
import matplotlib.pyplot as plt

#%% Constants
IMG_SIZE = 32
PENAL    = 3
RMIN     = 1.5
NGPpS = 9   # number of 2D integration points
NGPpL = 2   # number of 1D integration points
PENAL = 3
RMIN  = 1.5

#%% Stress prediction and Topology generation

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
        predict_stress(model, sample, N_in=N_in)
        Stress = sample.UNet_Stress.numpy()

    elif TYPE == 'FEM':
        Stress = predict_stress_FEM(eng, sample)
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

    sample.c = torch.tensor(float(c)).float() # sample compliance update

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
    next_sample.c                 = torch.tensor(1.0).float() # will be updated in the next iteration
    next_sample.FEMc              = torch.tensor(0.0).float()
    next_sample.NumIts            = sample.NumIts
    next_sample.ItsFull           = sample.ItsFull
    next_sample.TEnd              = sample.TEnd
    next_sample.Ener              = sample.Ener
    return next_sample




#%% Convergence criteria

def is_converged_compliance(sample_a: IterationSample, sample_b: IterationSample, tol=1e-3) -> bool:
    '''
    Check convergence between two IterationSamples based on compliance change.
    '''
    c_a = sample_a.c.item()
    c_b = sample_b.c.item()
    if abs(c_b - c_a) / abs(c_a) < tol:
        return True
    else:
        return False
    
def is_converged_window(List_iterations, window=5, tol=1e-3):
    """
    Check convergence by comparing the latest compliance to the mean of the
    previous `window` iterations. This smooths out the iteration-to-iteration
    noise of the U-Net predictions compared to a raw two-point comparison.

    Parameters
    ----------
    List_iterations : list[IterationSample] — optimization history.
    window          : int   — number of previous iterations averaged.
    tol             : float — relative tolerance on the compliance.

    Returns
    -------
    bool — True if the relative deviation is below `tol`.
    """
    # Not enough history yet to fill the window plus the current point.
    if len(List_iterations) < window + 1:
        return False
    c_values = np.array([s.c.item() for s in List_iterations[-(window+1):]])
    c_window = c_values[:-1]  # the `window` previous compliances
    c_last = c_values[-1]
    mean_c = c_window.mean()
    return abs(c_last - mean_c) / abs(mean_c) < tol


def is_converged_density(sample_a, sample_b, tol=0.01):
    """
    Check convergence based on the largest per-element density change between
    two successive iterations.

    Parameters
    ----------
    sample_a, sample_b : IterationSample — consecutive iterations.
    tol                : float — tolerance on the maximum absolute change.

    Returns
    -------
    bool — True if the maximum density change is below `tol`.
    """
    change = (sample_b.Densities - sample_a.Densities).abs().max().item()
    return change < tol


def is_converged_combined(List_iterations, window=5, tol_c=1e-3, tol_rho=0.01, enabled=True):
    """
    Combined convergence test used by the main optimization loop.

    When `enabled` is True, convergence requires BOTH a stable compliance
    (windowed criterion) AND a stable density field. When `enabled` is False,
    it falls back to the simpler two-point compliance criterion (used once the
    optimization has switched to FEM-only mode, where the signal is less noisy).

    Parameters
    ----------
    List_iterations : list[IterationSample] — optimization history.
    window          : int   — window size for the compliance criterion.
    tol_c           : float — relative tolerance on the compliance.
    tol_rho         : float — tolerance on the density change.
    enabled         : bool  — enable the combined (compliance + density) test.

    Returns
    -------
    bool — True if converged.
    """
    conv_c = is_converged_window(List_iterations, window, tol_c)
    conv_rho = is_converged_density(List_iterations[-2], List_iterations[-1], tol_rho)

    if enabled:
        return conv_c and conv_rho
    else:
        return is_converged_compliance(List_iterations[-1], List_iterations[-2], tol_c)


def is_converged_trend(List_iterations, window=5, tol=1e-4):
    """
    Check convergence from the slope of a linear fit of the last `window`
    compliances: a near-flat trend (relative slope below `tol`) signals
    convergence.

    Parameters
    ----------
    List_iterations : list[IterationSample] — optimization history.
    window          : int   — number of points used for the linear fit.
    tol             : float — relative tolerance on the slope.

    Returns
    -------
    bool — True if the normalized slope magnitude is below `tol`.
    """
    if len(List_iterations) < window:
        return False
    c_values = np.array([s.c.item() for s in List_iterations[-window:]])
    x = np.arange(window)
    slope = np.polyfit(x, c_values, 1)[0]  # first-order coefficient = slope
    return abs(slope) / abs(c_values.mean()) < tol


def is_converged_std(List_iterations, window=5, tol=1e-3):
    """
    Check convergence from the coefficient of variation (std / mean) of the
    compliance over the last `window` iterations.

    Parameters
    ----------
    List_iterations : list[IterationSample] — optimization history.
    window          : int   — number of points used.
    tol             : float — tolerance on the relative standard deviation.

    Returns
    -------
    bool — True if the relative std is below `tol`.
    """
    if len(List_iterations) < window:
        return False
    c_values = np.array([s.c.item() for s in List_iterations[-window:]])
    return c_values.std() / abs(c_values.mean()) < tol


def is_increasing_trend(List_iterations, window=5, threshold=0.0):
    """
    Detects whether compliance has been trending upward over the last `window` iterations,
    using the slope of a linear regression on the last `window+1` compliance values.
    """
    if len(List_iterations) < window + 1:
        return False
    c_values = np.array([s.c.item() for s in List_iterations[-(window+1):]])
    x = np.arange(window + 1)
    slope = np.polyfit(x, c_values, 1)[0]
    mean_c = c_values.mean()
    return (slope / abs(mean_c)) > threshold

###############################################################################
#%%  Error quantification                                  #

def extract_stress_maps(y):
    """
    Split a network stress output tensor into its three 2D component maps.

    Parameters
    ----------
    y : torch.Tensor [1, 3, 32, 32] — stacked stress channels (σx, σy, τxy).

    Returns
    -------
    tuple(np.ndarray, np.ndarray, np.ndarray) — the (σx, σy, τxy) maps [32, 32].
    """
    sx  = y[0, 0].cpu().numpy()   # σx  [32, 32]
    sy  = y[0, 1].cpu().numpy()   # σy  [32, 32]
    txy = y[0, 2].cpu().numpy()   # τxy [32, 32]
    return sx, sy, txy


def visualize_in_out(y_true, y_pred):
    """
    Plot a 2x3 grid comparing ground-truth and predicted stress fields
    (σx, σy, τxy) on a shared color scale, with a single common colorbar.

    Parameters
    ----------
    y_true : torch.Tensor [1, 3, 32, 32] — reference stress fields.
    y_pred : torch.Tensor [1, 3, 32, 32] — predicted stress fields.

    Returns
    -------
    None — displays a matplotlib figure.
    """
    # Inputs
    # rho   = x[0, 0].cpu().numpy()        # density        [32, 32]
    # tx    = x[0, 1].cpu().numpy()        # traction tx    [32, 32]
    # ty    = x[0, 2].cpu().numpy()        # traction ty    [32, 32]


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


def plot_error(y_true, y_pred, TYPE):
    """
    Plot the per-pixel error map between predicted and ground-truth stress
    fields for each component (σx, σy, τxy).

    Parameters
    ----------
    y_true : torch.Tensor [1, 3, 32, 32] — reference stress fields.
    y_pred : torch.Tensor [1, 3, 32, 32] — predicted stress fields.
    TYPE   : str — error type: 'MAE', 'MSE' or 'SMAPE'.

    Returns
    -------
    None — displays a matplotlib figure.
    """
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

    plt.show()


# %% Error calculation

def ErrorMetrics_1D(y_true, y_pred, TYPE):
    """
    Compute a scalar error metric per stress component (σx, σy, τxy) over the
    whole field.

    Parameters
    ----------
    y_true : torch.Tensor [1, 3, 32, 32] — reference stress fields.
    y_pred : torch.Tensor [1, 3, 32, 32] — predicted stress fields.
    TYPE   : str — metric: 'MSE', 'MAE', 'SMAPE' or 'R2'.

    Returns
    -------
    tuple(float, float, float) — the (σx, σy, τxy) metric values.
    """
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
    """
    Compute a spatially-smoothed error per stress component by convolving the
    per-pixel error map with a ones kernel, then averaging. This highlights
    localized error clusters rather than isolated pixels.

    Parameters
    ----------
    y_true      : torch.Tensor [1, 3, 32, 32] — reference stress fields.
    y_pred      : torch.Tensor [1, 3, 32, 32] — predicted stress fields.
    kernel_size : int   — side length of the square averaging kernel.
    pad         : bool  — pad the error map to preserve spatial size.
    strides     : (int, int) — (stride_h, stride_w) of the convolution.
    TYPE        : str   — per-pixel error type: 'MAE', 'MSE' or 'SMAPE'.
    plot        : bool  — display the smoothed error maps.

    Returns
    -------
    tuple(float, float, float) — mean smoothed error for (σx, σy, τxy).
    """
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


#%% TopOpt process

def _run_while_loop(sample, next_sample, i, List_iterations, List_Relative_Vol_Frac, 
                     List_mean_densities, List_count_FEM, count_unet,
                     eng, model, N_in, N_max_iterations,
                     match_FEM, match_Periodic, n_unet, m_fem, match_decreasing, threshold,
                     N_end_FEM_iterations=0, window_Unet=5, window_FEM=2, tol_c=1e-3, tol_rho=0.01, 
                     end_FEM='False'):
    """
    Run the main optimization loop until max iterations, convergence, or final compliance reached.
    Mutates and returns the tracking lists and state variables.
    """
    NEXT_TYPE = 'UNet'
    
    if match_FEM:
        enabled_rho_criteria = False
        window = window_FEM
    else:
        fem_mode  = False   # True while correcting a compliance increase with FEM steps
        enabled_rho_criteria = True
        window = window_Unet

        

    while (i < N_max_iterations
           and not is_converged_combined(List_iterations, window=window, tol_c=tol_c, tol_rho=tol_rho, enabled=enabled_rho_criteria)
        ):
        print(abs(List_iterations[-1].c.item()-List_iterations[-2].c.item())/List_iterations[-1].c.item())

        NEXT_TYPE = 'UNet' # Default setting

        if match_FEM:
            NEXT_TYPE = 'FEM'
            List_count_FEM.append(i)

        elif match_Periodic and count_unet >= n_unet:
            
            for j in range(m_fem-1):
                # Compute sample and add its properties to the lists
                next_sample = GenTopology(sample, eng, model, TYPE='FEM', N_in=N_in)
                List_Relative_Vol_Frac.append(sample.Relative_Vol_Frac)
                List_mean_densities.append(sample.Densities.numpy().mean())
                List_iterations.append(sample)
                List_count_FEM.append(i)
                i += 1

                # replace sample by next_sample for next iteration
                sample = next_sample

            NEXT_TYPE = 'FEM'
            List_count_FEM.append(i)
            count_unet = 0


        elif match_decreasing:
            increase = is_increasing_trend(List_iterations, window, threshold)

            if fem_mode:
                # Already in FEM correction mode: stay in FEM until the outer
                # while loop's convergence check ends the optimization.
                NEXT_TYPE = 'FEM'
                List_count_FEM.append(i)

            elif increase:
                # Compliance just increased: switch to FEM mode permanently.
                # Cancel the last (U-Net) step that caused the increase, and
                # recompute the previous iteration with FEM. The common section
                # below then commits the FEM-corrected candidate.
                del List_iterations[-1]
                del List_Relative_Vol_Frac[-1]
                del List_mean_densities[-1]
                i -= 1
                sample = List_iterations[-1]
                next_sample = GenTopology(sample, eng, model, TYPE='FEM', N_in=N_in)
                sample = next_sample
                List_count_FEM.append(i - 1)
                List_count_FEM.append(i)

                # reduce window size : FEM convergence has less noise
                window               =   window_FEM
                enabled_rho_criteria =   False

                NEXT_TYPE = 'FEM'
                fem_mode = True

            else:
                NEXT_TYPE = 'UNet'

        else:
            NEXT_TYPE = 'UNet'
            count_unet += 1


        # Common actualisation for both UNet and FEM iterations
        
        next_sample = GenTopology(sample, eng, model, TYPE=NEXT_TYPE, N_in=N_in)   
        
        List_Relative_Vol_Frac.append(sample.Relative_Vol_Frac)
        List_mean_densities.append(sample.Densities.numpy().mean())
        List_iterations.append(sample)
        sample=next_sample
        i += 1

        if (is_converged_combined(List_iterations, window=window, tol_c=tol_c, tol_rho=tol_rho)
            and i-1 not in List_count_FEM   # NEXT_TYPE == 'UNet'
            and end_FEM
        ): 
            # U-Net has converged, so a FEM iteration are done
            next_sample = GenTopology(sample, eng, model, TYPE='FEM', N_in=N_in)   

            List_Relative_Vol_Frac.append(sample.Relative_Vol_Frac)
            List_mean_densities.append(sample.Densities.numpy().mean())
            List_iterations.append(sample)
            List_count_FEM.append(i)

            sample=next_sample
            i += 1
            
            # U-Net has reached its capacities, next iterations will be FEM
            match_decreasing = False
            match_Periodic = False
            match_FEM = True
            window = window_FEM
            enabled_rho_criteria = False

    # Final FEM iterations
    
    for j in range(N_end_FEM_iterations):
        next_sample = GenTopology(sample, eng, model, TYPE='FEM', N_in=N_in)
        List_Relative_Vol_Frac.append(sample.Relative_Vol_Frac)
        List_mean_densities.append(sample.Densities.numpy().mean())
        List_iterations.append(sample)
        List_count_FEM.append(i)
        
        sample=next_sample
        i += 1
    
    

    return sample, next_sample, i, count_unet


# Full process

def run_topology_optimization(sample, eng, model, N_in=3, N_max_iterations=100,
                               RULE=' ', TYPE_FIRST='FEM', threshold=0.05, N_end_FEM_iterations=0,
                               window_Unet=5, window_FEM=2, tol_c=1e-3, tol_rho=0.01,
                               end_FEM=False,):
    """
    Run a full hybrid topology optimization starting from `sample`, mixing
    U-Net and FEM stress evaluations according to `RULE`.

    The `RULE` string selects the hybrid strategy:
      - 'Only FEM'              : every iteration uses FEM.
      - '<n> Unet - <m> FEM'    : periodic pattern of n U-Net then m FEM steps.
      - 'Decreasing compliance' : U-Net until the compliance starts rising,
                                  then switch permanently to FEM.
      - anything else           : U-Net only.

    The first iteration uses `TYPE_FIRST`; the core loop is delegated to
    `_run_while_loop`, and `N_end_FEM_iterations` extra FEM steps can be
    appended at the end.

    Parameters
    ----------
    sample            : IterationSample — initial state.
    eng               : matlab.engine — MATLAB engine.
    model             : trained network for stress prediction.
    N_in              : int — number of U-Net input channels (1 or 3).
    N_max_iterations  : int — iteration cap.
    RULE              : str — hybrid strategy selector (see above).
    TYPE_FIRST        : str — 'UNet' or 'FEM' for the first step.
    threshold         : float — slope threshold for the increasing-trend test.
    N_end_FEM_iterations : int — extra FEM steps appended at convergence.
    window_Unet/window_FEM : int — convergence window sizes per mode.
    tol_c, tol_rho    : float — compliance / density convergence tolerances.
    end_FEM           : bool — append a FEM step when the U-Net converges.

    Returns
    -------
    tuple(list[IterationSample], list[int])
        List_iterations : every computed iteration.
        List_count_FEM  : indices of the iterations computed with FEM.
    """
    List_count_FEM = []
    count_unet = 0

    # Detect a periodic "n Unet - m FEM" rule and extract n and m.
    match_Periodic = re.match(r'(\d+) Unet - (\d+) FEM', RULE)
    n_unet, m_fem = None, None

    if match_Periodic:
        n_unet = int(match_Periodic.group(1))
        m_fem  = int(match_Periodic.group(2))
        count_unet = 0
        

    match_FEM = re.match('Only FEM', RULE)

    match_decreasing = re.match('Decreasing compliance', RULE)

    # sample updated with stress and compliance
    # next_sample density updated, but stress and compliance not computed yes
    next_sample = GenTopology(sample, eng, model, TYPE=TYPE_FIRST, N_in=N_in)


    # Lists Initialisation
    List_Relative_Vol_Frac = [sample.Relative_Vol_Frac]
    List_mean_densities = [sample.Densities.numpy().mean()]

    List_iterations = [sample]
    i = 1 

    # Count FEM steps
    if TYPE_FIRST == 'FEM':
        List_count_FEM.append(0)

    if match_FEM: 
        NEXT_TYPE = 'FEM' 
        List_count_FEM.append(1)
    else: 
        NEXT_TYPE = 'UNet'

    # replace sample with next_sample
    sample = next_sample

    # sample updated with stress and compliance. Uptate also applies to List_iterations
    # next_sample density updated, but stress and compliance not computed yes
    next_sample = GenTopology(sample, eng, model, TYPE=NEXT_TYPE, N_in=N_in)    

    List_Relative_Vol_Frac.append(sample.Relative_Vol_Frac)
    List_mean_densities.append(sample.Densities.numpy().mean())
    List_iterations.append(sample)
    i += 1

    sample = next_sample

    # loop
    sample, next_sample, i, count_unet = _run_while_loop(
        sample, next_sample, i, List_iterations, List_Relative_Vol_Frac, 
        List_mean_densities, List_count_FEM, count_unet,
        eng, model, N_in, N_max_iterations,
        match_FEM, match_Periodic, n_unet, m_fem, match_decreasing, threshold,
        N_end_FEM_iterations,
        window_Unet=window_Unet, window_FEM=window_FEM, 
        tol_c=tol_c, tol_rho=tol_rho, 
        end_FEM=end_FEM,
    )

    # If last FEM step causes a compliance augmentation, the solution has not been reached
    # a new loop is applied, with only FEM because the solution is close
    # if (not is_converged_compliance(List_iterations[-1], List_iterations[-2], tol=tol)
    #     and end_FEM): 

    #     #Only FEM
    #     match_FEM=True
    #     match_decreasing=False
    #     match_Periodic=False

    #     sample, next_sample, i, count_unet = _run_while_loop(
    #         sample, next_sample, i, List_iterations, List_Relative_Vol_Frac, 
    #         List_mean_densities, List_count_FEM, count_unet,
    #         eng, model, N_in, N_max_iterations, final_compliance,
    #         match_FEM, match_Periodic, n_unet, m_fem, match_decreasing, threshold,
    #         N_end_FEM_iterations, tol=tol, end_FEM=end_FEM
    #     )


    return List_iterations, List_count_FEM


# %% Convergence study
def visualize_convergence(List_Iterations_Unet, IterationDataset_FEM, List_count_FEM, NETWORK:str, PLOT=True,SCALE='linear'):
    """
    Plot the compliance convergence of a single optimization: the full-FEM
    reference curve against the hybrid U-Net/FEM run, with the FEM steps of the
    hybrid run highlighted. Both curves are normalized by the initial FEM
    compliance.

    Parameters
    ----------
    List_Iterations_Unet : list[IterationSample] — hybrid run history.
    IterationDataset_FEM : IterationDataset — full-FEM reference run.
    List_count_FEM       : list[int] — indices of FEM steps in the hybrid run.
    NETWORK              : str — network name, used in labels.
    PLOT                 : bool — kept for API symmetry (figure always drawn).
    SCALE                : str — y-axis scale ('linear' or 'log').

    Returns
    -------
    tuple(np.ndarray, np.ndarray) — the normalized (FEM_c, UNet_c) curves,
        each as an array of [iteration, compliance] rows.
    """
    f_text=1.25 # text size multiplicator


    FEM_c=[]
    UNet_c=[]

    for i in range(len(IterationDataset_FEM)):
        sample=IterationSample(IterationDataset_FEM, i)
        FEM_c.append([i,sample.c])

    FEM_step_c=[]

    for i,sample in enumerate(List_Iterations_Unet):
        UNet_c.append([i, sample.c.item()])
        if i in List_count_FEM:
            FEM_step_c.append((i, sample.c.item()))

    FEM_c=np.array(FEM_c)

    c0_FEM = FEM_c[0, 1]

    FEM_c[:, 1] = FEM_c[:, 1] / c0_FEM # normalize by initial FEM compliance
    

    UNet_c=np.array(UNet_c)
    UNet_c[:, 1] = UNet_c[:, 1] / c0_FEM # normalize by initial compliance

    MARKERSIZE = 7

    plt.figure(figsize=(10, 6))
    plt.plot(FEM_c[:, 0], FEM_c[:, 1], 'o-', linewidth=2.5, markersize=MARKERSIZE, label=f'FEM: {len(IterationDataset_FEM)} steps')
    plt.plot(UNet_c[:, 0], UNet_c[:, 1], 's-', linewidth=2.5, markersize=MARKERSIZE, label=f'{NETWORK}: {len(List_Iterations_Unet)-len(List_count_FEM)} steps')
    
    # last FEM compliance
    plt.plot( (0, FEM_c[-1,0]) , (FEM_c[-1,1], FEM_c[-1,1]), color='tab:blue', linestyle='-', linewidth=2, dashes=(3, 2) )
    plt.plot((0, UNet_c[-1,0]), (UNet_c[-1,1], UNet_c[-1,1]), color='tab:orange', linewidth=2, dashes=(6, 4))


    if len(FEM_step_c)>0:
        FEM_step_c=np.array(FEM_step_c)
        FEM_step_c[:, 1] = FEM_step_c[:, 1] / c0_FEM # normalize by initial FEM compliance
        plt.plot(FEM_step_c[:, 0], FEM_step_c[:, 1], 'rs', markersize=MARKERSIZE, label = f'Hybrid strategy: {len(List_count_FEM)} FEM steps')
        
    plt.xlabel('Iterations', fontsize=f_text*14,)
    plt.ylabel(f'$c/c_{{0,FEM}}$', fontsize=f_text*14, )
    
    # plt.yscale(SCALE)
    # plt.ylim(0,1.1)
    plt.title(f'Compliance convergence: full-FEM vs Hybrid {NETWORK} strategy', fontsize=f_text*16, )
    plt.legend(fontsize=f_text*13)
    plt.grid(True, alpha=0.3)

    # Set x-axis ticks: integers only, step of 5
    max_iter = max(FEM_c[-1, 0], UNet_c[-1, 0])
    plt.xticks(range(0, int(max_iter) + 5, 5), fontsize=f_text*12)
    plt.yticks(fontsize=f_text*12)
    plt.tight_layout()
    plt.show()

    return FEM_c, UNet_c


def statistical_convergence(List_List_Iterations_UNet, IterData_FEM:IterationDataset, NETWORK='U-Net', PLOT=True, TYPE='std'):
    '''
    Returns the mean evolution of the compliance.
    '''
    IterData_Unet=list_to_IterationDataset(List_List_Iterations_UNet[0])

    for i in range(1, len(List_List_Iterations_UNet)):
        IterData_Unet += list_to_IterationDataset(List_List_Iterations_UNet[i])


    Variation_c = []

    for IterData in [IterData_FEM, IterData_Unet]: 
        c_array = IterData.dataset.c  # object array (N,), each element is (1, n_iter)

        N_max = max(c_array[i].flatten().shape[0] for i in range(len(c_array)))

        dict_c = {j: [] for j in range(N_max)}

        for i in range(len(c_array)):
            c_i = c_array[i].flatten()
            c0  = c_i.max()  # compliance at index 1
            for j in range(len(c_i)):
                dict_c[j].append(float(c_i[j]/c0))

        tab_c = []
        for key in dict_c.keys():
            mean = np.mean(dict_c[key])
            std  = np.std(dict_c[key])
            number = len(dict_c[key])
            tab_c.append((mean, std, number))

        Variation_c.append(tab_c)

    FEM_c, UNet_c = Variation_c

    if PLOT:
        fig, ax = plt.subplots(figsize=(10, 5))

        labels = ['FEM', NETWORK]
        colors = ['tab:blue', 'tab:orange']

        for k, (tab_c, label, color) in enumerate(zip(Variation_c, labels, colors)):
            means = [mean for mean, std, number in tab_c[2:]]
            stds  = [std  for mean, std, number in tab_c[2:]]
            LENS = [number for mean, std, number in tab_c[2:]]

            LENS = [n / max(LENS) for n in LENS]

            ax.plot(means, label=label, color=color)

            if TYPE == 'std':
                ax.fill_between(range(len(means)),
                                [m - s for m, s in zip(means, stds)],
                                [m + s for m, s in zip(means, stds)],
                                alpha=0.3, color=color)
            elif TYPE == 'lenght':
                ax.fill_between(range(len(means)),
                                [m - l/len(Variation_c) for m, l in zip(means, LENS)],
                                [m + l/len(Variation_c) for m, l in zip(means, LENS)],
                                alpha=0.3, color=color)


        ax.set_xlabel('Iteration')
        ax.set_ylabel('c / c_max')
        ax.set_title(f'Compliance convergence - mean and {TYPE} across distributions')
        ax.legend()
        ax.grid()
        ax.set_xlim(0, 100)
        plt.tight_layout()
        plt.show()

    return FEM_c, UNet_c


def density_evolution(List_iterations, List_count_FEM, step=5):
    """
    Display the evolution of density distributions across iterations.
    Tractions arrows are only shown on the first subplot.
    """
    indices = list(range(0, len(List_iterations), step))
    n_plots = len(indices)

    n_cols = int(np.ceil(np.sqrt(n_plots)))
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
    axes = np.array(axes).reshape(-1)  # flatten for easy indexing

    scale_force = 10
    cadre       = int(scale_force)

    for plot_idx, iter_idx in enumerate(indices):
        ax = axes[plot_idx]
        sample = List_iterations[iter_idx]

        topo     = sample.Densities.squeeze().numpy()
        img_size = int(np.sqrt(len(topo)))
        img      = topo.reshape(img_size, img_size)

        ax.imshow(img, cmap='gray_r', origin='lower',
                  extent=[0, img_size, 0, img_size], vmin=0, vmax=1)
        ax.set_aspect('equal')
        ax.axis('off')

        if iter_idx in List_count_FEM:
            TYPE = 'FEM'
        else:
            TYPE = 'U-Net'
        ax.set_title(f'i={iter_idx} : {TYPE}', fontsize=16)

        if plot_idx == 0:
            # Tractions arrows only on the first subplot
            ax.set_xlim(-cadre, img_size + cadre)
            ax.set_ylim(-cadre, img_size + cadre)

            T_scale = sample.Tractions.squeeze().numpy() * scale_force
            T_scale = T_scale.T

            Points = np.array([
                [0,        img_size],
                [img_size, img_size],
                [img_size, img_size],
                [img_size, 0       ],
                [img_size, 0       ],
                [0,        0       ],
                [0,        0       ],
                [0,        img_size],
            ], dtype=float)

            for k in range(8):
                sx, sy = Points[k]
                tx, ty = T_scale[k]
                ax.quiver(sx, sy, tx, ty, angles='xy', scale_units='xy', scale=1,
                          color='r', linewidth=1, headwidth=2)

    # Hide unused subplots
    for plot_idx in range(n_plots, len(axes)):
        axes[plot_idx].axis('off')

    plt.tight_layout()
    plt.show()

#%% Compare NN and FEM results for a given force distribution

def compare_NN_FEM(sample_NN, sample_FEM):
    """
    Display the optimized densities of a U-Net run and a FEM run side by side,
    each overlaid with its boundary traction arrows, sharing a common density
    colorbar and a reference force-scale arrow.

    Parameters
    ----------
    sample_NN  : IterationSample — U-Net optimized result.
    sample_FEM : IterationSample — FEM optimized result.

    Returns
    -------
    None — displays a matplotlib figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    scale_force = 10
    cadre       = int(scale_force)

    def plot_density(ax, sample, title):
        """Draw one sample's density map plus its traction arrows on `ax`;
        returns the AxesImage so the caller can build a shared colorbar."""
        topo     = sample.Densities.squeeze().numpy()
        img_size = int(np.sqrt(len(topo)))
        img      = topo.reshape(img_size, img_size)

        im = ax.imshow(img, cmap='gray_r', origin='lower',
                       extent=[0, img_size, 0, img_size], vmin=0, vmax=1)
        ax.set_xlim(-cadre, img_size + cadre)
        ax.set_ylim(-cadre, img_size + cadre)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=24)

        T_scale = sample.Tractions.squeeze().numpy() * scale_force
        T_scale = T_scale.T

        Points = np.array([
            [0,        img_size],
            [img_size, img_size],
            [img_size, img_size],
            [img_size, 0       ],
            [img_size, 0       ],
            [0,        0       ],
            [0,        0       ],
            [0,        img_size],
        ], dtype=float)

        for k in range(8):
            sx, sy = Points[k]
            tx, ty = T_scale[k]
            ax.quiver(sx, sy, tx, ty, angles='xy', scale_units='xy', scale=1,
                      color='r', linewidth=1, headwidth=2)

        return im

    im = plot_density(axes[0], sample_NN,  'U-Net density')
    plot_density(axes[1], sample_FEM, 'FEM density')

    # Common reference arrow on axes[0]
    pos_x=40
    pos_y=0
    axes[0].quiver(-cadre + 1 + pos_x, -cadre + 2 + pos_y, scale_force, 0,
                   angles='xy', scale_units='xy', scale=1,
                   color='r', linewidth=1, headwidth=2)
    axes[0].text(-cadre - 2 + pos_x, -cadre + 4 + pos_y, '1 force unit', fontsize=20, color='r')


    # Common colorbar
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.48, 0.2, 0.02, 0.6])  # [left, bottom, width, height]
    cb = fig.colorbar(im, cax=cbar_ax, ticks=np.arange(0, 1.2, 0.2))
    cb.mappable.set_clim(0, 1)
    cb.ax.tick_params(labelsize=16)

    plt.tight_layout()
    plt.show()


#%% Strategy comparison
def plot_FEM_error_c(list_benchmark, Tab_ratio_FEM, Tab_err_rel_c, TYPE_BENCHMARK='Hybrid'):
    """
    Tab_ratio_FEM : (n_configs, SIZE_LOOP)
    Tab_err_rel_c  : (n_configs, SIZE_LOOP)
    """
    n = len(list_benchmark)
    x = np.arange(n)
    width = 0.35

    FONT = 18
    FONT_SIZE = 13  # font size of the parameter table cells

    # Aggregate over SIZE_LOOP
    mean_FEM     = (Tab_ratio_FEM*100).mean(axis=1)
    std_FEM      = (Tab_ratio_FEM*100).std(axis=1)
    mean_err_pct = (Tab_err_rel_c * 100).mean(axis=1)
    std_err_pct  = (Tab_err_rel_c * 100).std(axis=1)

    # Build x-axis labels and optional parameter table
    if TYPE_BENCHMARK == 'Hybrid':
        labels = [f"{b[0]}\n{b[1]} start" for b in list_benchmark]
        table_data = None
    else:
        labels = [f"Config {i+1}" for i in range(n)]
        param_names = ['NIF', 'N_conv', 'CBAM', 'aug', 'p_aug', 'portion', 'bs']
        aug_idx = param_names.index('aug')

        def cell(b, k):
            # Hide p_aug value when aug is False
            if param_names[k] == 'p_aug' and not b[3 + aug_idx]:
                return ''
            return str(b[3 + k])

        table_data = [[cell(b, k) for b in list_benchmark] for k in range(len(param_names))]

    fig, ax1 = plt.subplots(figsize=(max(10, n * 1.5), 7 if table_data else 6))

    # FEM iterations — left axis
    bars1 = ax1.bar(x - width/2, mean_FEM, width, yerr=std_FEM,
                     capsize=4, color='tab:blue', alpha=0.8)
    ax1.set_ylabel(r'Ratio of FEM iterations: $N_{Hybrid}/N_{FEM}$ (%)', fontsize=13, color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Relative error (%) — right axis
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, mean_err_pct, width, yerr=std_err_pct,
                     capsize=4, color='tab:orange', alpha=0.8)
    ax2.set_ylabel('Relative compliance error (%)', fontsize=13, color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    # Bar value labels
    for bar, val in zip(bars1, mean_FEM):
        ax1.text(bar.get_x() + bar.get_width() * 0.55, bar.get_height() + 0.01,
                 f'{val:.1f}%', ha='left', va='bottom', fontsize=9, color='tab:blue')
    for bar, val in zip(bars2, mean_err_pct):
        ax2.text(bar.get_x() + bar.get_width() * 0.55, bar.get_height() + 0.001,
                 f'{val:.2f}%', ha='left', va='bottom', fontsize=9, color='tab:orange')

    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)

    ax1.set_xticks(x)
    rotation = 30 if TYPE_BENCHMARK == 'Hybrid' else 0
    ax1.set_xticklabels(labels, fontsize=13, rotation=rotation, ha='right' if rotation else 'center')

    ax1.legend([bars1, bars2], ['Ratio of FEM iterations', 'Relative error (%)'],
                fontsize=11, loc='upper left')

    if table_data is not None:
        table = ax1.table(
            cellText=table_data,
            rowLabels=param_names,
            colLabels=labels,
            cellLoc='center',
            rowLoc='center',
            loc='bottom',
            bbox=[0, -0.6, 1, 0.55]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(FONT_SIZE)
        ax1.set_xticklabels([])
        ax1.set_xlabel('')

    plt.title(f'Hybrid strategies comparison — {len(Tab_ratio_FEM[0])} traction distributions',
              fontsize=FONT)
    plt.tight_layout()
    plt.show()


#%% Window showing the progress of the process
class ProgressWindow:
    """
    Small Tkinter window showing benchmark progress (step count + elapsed time).

    The Tk root is created and updated from the GUI thread (see `run_window`),
    while the worker thread only mutates counters via `increment()` / `close()`.
    """

    def __init__(self, total):
        """Initialize the counters; the Tk window itself is built later in the
        GUI thread by `_setup()`.

        Parameters
        ----------
        total : int — total number of steps to reach 100%.
        """
        self.total = total
        self.step  = 0
        self.start = time.time()
        self.running = True
        self.root = None  # created inside the GUI thread by _setup()

    def _setup(self):
        """Build the Tk window and its labels, then start the refresh loop.
        Must be called from the GUI thread."""
        self.root = tk.Tk()
        self.root.title("Benchmark progress")
        self.root.geometry("300x120")
        self.root.resizable(False, False)

        self.label_step = tk.Label(self.root, text=f"Step: 0 / {self.total}", font=("Arial", 14))
        self.label_step.pack(pady=10)

        self.label_time = tk.Label(self.root, text="Elapsed: 00:00:00", font=("Arial", 14))
        self.label_time.pack(pady=5)

        self._update()

    def _update(self):
        """Refresh the step/time labels every 500 ms, and destroy the window
        once `running` has been cleared by `close()`."""
        if not self.running:
            self.root.destroy()
            return
        elapsed = int(time.time() - self.start)
        h, rem = divmod(elapsed, 3600)
        m, s   = divmod(rem, 60)
        self.label_time.config(text=f"Elapsed: {h:02d}:{m:02d}:{s:02d}")
        self.label_step.config(text=f"Step: {self.step} / {self.total}")
        self.root.after(500, self._update)

    def increment(self):
        """Advance the step counter by one (called from the worker thread)."""
        self.step += 1

    def close(self):
        """Request the window to close. Thread-safe: only clears a flag; the
        actual `root.destroy()` happens in `_update()` on the GUI thread."""
        # Signal the GUI thread to destroy the window; destruction happens in _update()
        self.running = False

def run_window(win):
    """
    GUI-thread entry point for a ProgressWindow: build the window and enter the
    Tk main loop. Intended to be the target of a daemon thread.

    Parameters
    ----------
    win : ProgressWindow — the progress window to run.

    Returns
    -------
    None
    """
    win._setup()
    win.root.mainloop()


#%% sMAPE benchmark across models

# Column layout of each list_benchmark entry (mirrors the benchmark scripts).
BENCHMARK_CONFIG_COLUMNS = ['Strategy', 'Model', 'First step', 'NIF', 'N_conv',
                            'use cbam', 'use augmentation', 'probability of augmentation',
                            'dataset portion', 'batch size']


def _build_model_from_config(bench, RESULTS_ROOT, name_file,
                             hidden_layers_MLP=(32, 64), embed_out=128):
    """
    Rebuild and load a trained model from one `list_benchmark` entry.
    Mirrors the model-loading block of the benchmark scripts.

    Parameters
    ----------
    bench             : list — one `list_benchmark` row (see BENCHMARK_CONFIG_COLUMNS).
    RESULTS_ROOT      : Path — root of the trained-model results tree.
    name_file         : str  — dataset tag used when the models were trained.
    hidden_layers_MLP : tuple — MLP layout for BE_UNet (unused for U-Net).
    embed_out         : int   — embedding dim for BE_UNet (unused for U-Net).

    Returns
    -------
    tuple(nn.Module, str, int) — (model in eval mode, NETWORK, N_in).
    """
    (STRATEGY, NETWORK, FIRST_STEP, NIF, N_CONV, USE_CBAM,
     USE_AUGMENTATION, AUGMENTATION_P, PORTION_DATA, BATCH_SIZE) = bench

    N_in = 1 if NETWORK == 'BE_UNet' else 3

    if NETWORK == 'U-Net':
        tag = (f'{name_file}_NIF={NIF}_{N_CONV}_conv_CBAM={USE_CBAM}'
               f'_aug={USE_AUGMENTATION}_portion={int(PORTION_DATA*100)}%_batch={BATCH_SIZE}')
    else:
        tag = (f'{name_file}_NIF={NIF}_{N_CONV}_conv_{list(hidden_layers_MLP)}_CBAM={USE_CBAM}'
               f'_aug={USE_AUGMENTATION}_portion={int(PORTION_DATA*100)}%_batch={BATCH_SIZE}')

    BEST_PATH = RESULTS_ROOT / NETWORK / tag / ('unet_' + name_file + '_best.pth')

    if NETWORK == 'BE_UNet':
        model = BE_UNetTopo(nif=NIF, n_in=N_in, n_out=3, use_cbam=USE_CBAM,
                            hidden_layers_MLP=list(hidden_layers_MLP),
                            embed_out=embed_out, N_conv=N_CONV)
    elif NETWORK == 'U-Net':
        model = UNetTopo(nif=NIF, n_in=N_in, n_out=3, use_cbam=USE_CBAM, N_conv=N_CONV)
    else:
        raise ValueError("Invalid NETWORK value. Choose 'U-Net' or 'BE_UNet'.")

    model.load_state_dict(torch.load(BEST_PATH, map_location='cpu'))
    model.eval()
    return model, NETWORK, N_in


def save_smape_benchmark(list_benchmark, ds_iter, csv_path, RESULTS_ROOT, name_file,
                         device=None, eps=1e-6, reset=True):
    """
    Compute, for every model in `list_benchmark`, the total sMAPE of each sample
    of `ds_iter`, and store one row per (config, sample) in a CSV file.

    The sMAPE is computed exactly like at training time (`sMAPELoss` over the
    three stress components σx, σy, τxy), sample by sample. The resulting CSV is
    meant to be read back by `plot_smape_benchmark`.

    Parameters
    ----------
    list_benchmark : list[list] — configurations (see BENCHMARK_CONFIG_COLUMNS).
    ds_iter        : IterationDataset — samples to evaluate.
    csv_path       : path-like — destination CSV file.
    RESULTS_ROOT   : Path — root of the trained-model results tree.
    name_file      : str  — dataset tag used at training time.
    device         : torch.device | None — auto-selected (CUDA if available).
    eps            : float — sMAPE denominator epsilon.
    reset          : bool  — overwrite the CSV ('w') and write the header;
                             if False, append ('a') without a header.

    Returns
    -------
    None — writes `csv_path`.
    """
    import csv
    from torch.utils.data import DataLoader
    from train import sMAPELoss, _batch_to_tensors, _forward

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    criterion = sMAPELoss(eps=eps)
    loader    = DataLoader(ds_iter, batch_size=1, shuffle=False)
    columns   = BENCHMARK_CONFIG_COLUMNS + ['Sample index', 'sMAPE']

    with open(csv_path, 'w' if reset else 'a', newline='') as f:
        writer = csv.writer(f)
        if reset:
            writer.writerow(columns)

        for bench in list_benchmark:
            model, NETWORK, _ = _build_model_from_config(bench, RESULTS_ROOT, name_file)
            model.to(device)

            with torch.no_grad():
                for idx, batch in enumerate(loader):
                    tensors = _batch_to_tensors(batch, device, NETWORK)
                    pred, y = _forward(model, tensors, NETWORK)
                    smape   = criterion(pred, y).item()
                    writer.writerow(list(bench) + [idx, smape])

            print(f"[sMAPE] {bench} — done ({len(ds_iter)} samples)")


def _aggregate_smape(list_benchmark, csv_path):
    """
    Read the CSV written by `save_smape_benchmark` and return, per configuration
    (ordered like `list_benchmark`), the mean and std of the total sMAPE over the
    `ds_iter` samples.

    Returns
    -------
    tuple(np.ndarray, np.ndarray, int) — (mean_smape, std_smape, n_samples).
    """
    import pandas as pd

    df = pd.read_csv(csv_path)

    mean_smape, std_smape, n_samples = [], [], 0
    for bench in list_benchmark:
        mask = (df[BENCHMARK_CONFIG_COLUMNS] ==
                pd.Series(bench, index=BENCHMARK_CONFIG_COLUMNS)).all(axis=1)
        vals = df[mask]['sMAPE'].values
        mean_smape.append(vals.mean())
        std_smape.append(vals.std())
        n_samples = max(n_samples, len(vals))

    return np.array(mean_smape), np.array(std_smape), n_samples


def plot_smape_benchmark(list_benchmark, csv_path, TYPE_BENCHMARK='Architecture'):
    """
    Read the CSV written by `save_smape_benchmark` and draw a bar chart of the
    mean total sMAPE (± std) over the `ds_iter` samples, one bar per model.

    Visual style mirrors `plot_FEM_error_c`: a parameter table under the bars
    for the 'Architecture' benchmark, or two-line strategy labels for 'Hybrid'.

    Parameters
    ----------
    list_benchmark : list[list] — configurations, in the desired bar order.
    csv_path       : path-like — CSV produced by `save_smape_benchmark`.
    TYPE_BENCHMARK : str — 'Architecture' (default) or 'Hybrid'.

    Returns
    -------
    None — displays a matplotlib figure.
    """
    mean_smape, std_smape, n_samples = _aggregate_smape(list_benchmark, csv_path)

    n = len(list_benchmark)
    x = np.arange(n)
    width = 0.6

    FONT      = 18
    FONT_SIZE = 13  # font size of the parameter table cells

    # Build x-axis labels and optional parameter table (same logic as plot_FEM_error_c)
    if TYPE_BENCHMARK == 'Hybrid':
        labels = [f"{b[0]}\n{b[1]} start" for b in list_benchmark]
        table_data = None
    else:
        labels = [f"Config {i+1}" for i in range(n)]
        param_names = ['NIF', 'N_conv', 'CBAM', 'aug', 'p_aug', 'portion', 'bs']
        aug_idx = param_names.index('aug')

        def cell(b, k):
            # Hide p_aug value when aug is False
            if param_names[k] == 'p_aug' and not b[3 + aug_idx]:
                return ''
            return str(b[3 + k])

        table_data = [[cell(b, k) for b in list_benchmark] for k in range(len(param_names))]

    fig, ax = plt.subplots(figsize=(max(10, n * 1.5), 7 if table_data else 6))

    bars = ax.bar(x, mean_smape, width, yerr=std_smape,
                  capsize=4, color='tab:green', alpha=0.85)
    ax.set_ylabel('Mean total sMAPE', fontsize=13, color='tab:green')
    ax.tick_params(axis='y', labelcolor='tab:green')

    for bar, val in zip(bars, mean_smape):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, color='tab:green')

    ax.set_ylim(bottom=0)
    ax.set_xticks(x)
    rotation = 30 if TYPE_BENCHMARK == 'Hybrid' else 0
    ax.set_xticklabels(labels, fontsize=13, rotation=rotation,
                       ha='right' if rotation else 'center')

    if table_data is not None:
        table = ax.table(
            cellText=table_data,
            rowLabels=param_names,
            colLabels=labels,
            cellLoc='center',
            rowLoc='center',
            loc='bottom',
            bbox=[0, -0.6, 1, 0.55]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(FONT_SIZE)
        ax.set_xticklabels([])
        ax.set_xlabel('')

    plt.title(f'Model comparison — mean total sMAPE over {n_samples} stress predictions',
              fontsize=FONT)
    plt.tight_layout()
    plt.show()


def _model_display_name(b):
    """Human-readable model name for a benchmark row (b[1] = network id)."""
    return 'BE U-Net' if b[1] == 'BE_UNet' else 'U-Net'


def _model_param_count(b):
    """
    Count the trainable parameters of the model described by a benchmark row.
    Mirrors the model construction in TopOpt_benchmark_architecture.py.
    Returns a compact string (e.g. '1.24M') or '-' if the model cannot be built.
    """
    NETWORK, NIF, N_CONV, USE_CBAM = b[1], b[3], b[4], b[5]
    HIDDEN_LAYERS_MLP = [32, 64]
    EMBED_OUT = 128
    try:
        if NETWORK == 'BE_UNet':
            model = BE_UNetTopo(nif=NIF, n_in=1, n_out=3, use_cbam=USE_CBAM,
                                hidden_layers_MLP=HIDDEN_LAYERS_MLP,
                                embed_out=EMBED_OUT, N_conv=N_CONV)
        else:
            model = UNetTopo(nif=NIF, n_in=3, n_out=3,
                             use_cbam=USE_CBAM, N_conv=N_CONV)
        n_params = sum(p.numel() for p in model.parameters())
    except Exception:
        return '-'
    return f'{n_params / 1e6:.2f}M'


def plot_FEM_smape_c(list_benchmark, Tab_ratio_FEM, Tab_err_rel_c, smape_csv,
                     TYPE_BENCHMARK='Architecture'):
    """
    Fused comparison chart merging `plot_FEM_error_c` and `plot_smape_benchmark`:
    three grouped bars per configuration, each on its own y-axis.

      - blue   : ratio of FEM iterations  N_hybrid / N_FEM        (%)   — left axis
      - orange : relative compliance error                        (%)   — right axis
      - green  : mean total sMAPE over the ds_iter samples              — far-right axis

    Parameters
    ----------
    list_benchmark : list[list] — configurations, in the desired bar order.
    Tab_ratio_FEM  : np.ndarray (n_configs, SIZE_LOOP) — FEM-iteration ratios.
    Tab_err_rel_c  : np.ndarray (n_configs, SIZE_LOOP) — relative compliance errors.
    smape_csv      : path-like — CSV produced by `save_smape_benchmark`.
    TYPE_BENCHMARK : str — 'Architecture' (default) or 'Hybrid'.

    Returns
    -------
    None — displays a matplotlib figure.
    """
    n = len(list_benchmark)
    x = np.arange(n)
    width = 0.25

    FONT      = 18
    FONT_SIZE = 10  # font size of the parameter table cells

    # Aggregate the three metrics
    mean_FEM     = (Tab_ratio_FEM * 100).mean(axis=1)
    std_FEM      = (Tab_ratio_FEM * 100).std(axis=1)
    mean_err_pct = (Tab_err_rel_c * 100).mean(axis=1)
    std_err_pct  = (Tab_err_rel_c * 100).std(axis=1)
    mean_smape, std_smape, n_smp = _aggregate_smape(list_benchmark, smape_csv)
    mean_smape = mean_smape * 100   # sMAPE as a percentage
    std_smape  = std_smape * 100

    # Build x-axis labels and optional parameter table (same logic as plot_FEM_error_c)
    if TYPE_BENCHMARK == 'Hybrid':
        labels = [f"{b[0]}\n{b[1]} start" for b in list_benchmark]
        table_data = None
    else:
        labels = [f"Config {i+1}" for i in range(n)]
        param_names = ['NIF', 'N_conv', 'CBAM', 'aug', 'p_aug', 'portion', 'bs']
        aug_idx = param_names.index('aug')

        def cell(b, k):
            if param_names[k] == 'p_aug' and not b[3 + aug_idx]:
                return ''
            return str(b[3 + k])

        table_data = [[cell(b, k) for b in list_benchmark] for k in range(len(param_names))]

        # Model name (first row) and parameter count (last row)
        model_row  = [_model_display_name(b) for b in list_benchmark]
        nparam_row = [_model_param_count(b)  for b in list_benchmark]
        table_data = [model_row] + table_data + [nparam_row]
        row_labels = ['Model'] + param_names + ['# params']

    fig, ax1 = plt.subplots(figsize=(max(11, n * 1.6), 7 if table_data else 6))

    # Second and third y-axes; the third spine is pushed further right.
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))

    # FEM iteration ratio — left axis (blue)
    bars1 = ax1.bar(x - width, mean_FEM, width, yerr=std_FEM,
                    capsize=4, color='tab:blue', alpha=0.85)
    ax1.set_ylabel(r'Ratio of FEM iterations: $N_{Hybrid}/N_{FEM}$ (%)',
                   fontsize=13, color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Relative compliance error — right axis (orange)
    bars2 = ax2.bar(x, mean_err_pct, width, yerr=std_err_pct,
                    capsize=4, color='tab:orange', alpha=0.85)
    ax2.set_ylabel('Relative compliance error (%)', fontsize=13, color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    # Mean total sMAPE — far-right axis (green)
    bars3 = ax3.bar(x + width, mean_smape, width, yerr=std_smape,
                    capsize=4, color='tab:green', alpha=0.85)
    ax3.set_ylabel('Mean total sMAPE (%)', fontsize=13, color='tab:green')
    ax3.tick_params(axis='y', labelcolor='tab:green')

    # Bar value labels
    for bar, val in zip(bars1, mean_FEM):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f'{val:.1f}', ha='center', va='bottom', fontsize=8, color='tab:blue')
    for bar, val in zip(bars2, mean_err_pct):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f'{val:.2f}', ha='center', va='bottom', fontsize=8, color='tab:orange')
    for bar, val in zip(bars3, mean_smape):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f'{val:.1f}', ha='center', va='bottom', fontsize=8, color='tab:green')

    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)
    ax3.set_ylim(bottom=0)

    ax1.set_xticks(x)
    rotation = 30 if TYPE_BENCHMARK == 'Hybrid' else 0
    ax1.set_xticklabels(labels, fontsize=13, rotation=rotation,
                        ha='right' if rotation else 'center')

    ax1.legend([bars1, bars2, bars3],
               ['Ratio of FEM iterations (%)', 'Relative error (%)', 'Mean total sMAPE (%)'],
               fontsize=11, loc='upper left')

    if table_data is not None:
        table = ax1.table(
            cellText=table_data,
            rowLabels=row_labels,
            colLabels=labels,
            cellLoc='center',
            rowLoc='center',
            loc='bottom',
            bbox=[0, -0.6, 1, 0.55]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(FONT_SIZE)
        ax1.set_xticklabels([])
        ax1.set_xlabel('')

    plt.title(f'Model comparison — FEM ratio, compliance error & sMAPE '
              f'({len(Tab_ratio_FEM[0])} distributions, {n_smp} sMAPE samples)',
              fontsize=FONT)
    plt.tight_layout()
    plt.show()



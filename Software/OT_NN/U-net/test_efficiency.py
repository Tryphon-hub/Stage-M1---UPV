# test_efficiency.py
import torch
import scipy.io
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


from dataset import Dataset_TopOpt, IterationDataset, IterationSample, load_mat
from model   import UNetTopo
from train   import _batch_to_tensors

#%% ─── Configuration ────────────────────────────────────────────────────────────

# DATA_PATH   = Path(r"D:\Maxence\Heavy files\U-net\data\Test_3_gen.mat")
# BEST_PATH   = Path(r"D:\Maxence\Heavy files\U-net\results\unet_topo_best.pth")

DATA_PATH = Path(r"C:\Users\maxen\Documents\Stage\HeavyFiles\data\dataset_test.mat")
BEST_PATH = Path(r"C:\Users\maxen\Documents\Stage\Software\OT_NN\U-net\results\unet_topo_best.pth")

NIF      = 32
USE_CBAM = True
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% ─── Chargement du modèle ─────────────────────────────────────────────────────

model = UNetTopo(nif=NIF, n_in=3, n_out=3, use_cbam=USE_CBAM).to(device)
model.load_state_dict(torch.load(BEST_PATH, map_location=device))
model.eval()
print(f"Modèle chargé ({device})")

#%% ─── Chargement du dataset ────────────────────────────────────────────────────

data_raw    = load_mat(DATA_PATH)
ds_base = Dataset_TopOpt(data_raw)
ds_iter = IterationDataset(ds_base)



print(f"Distributions de forces : {len(ds_base)}")
print(f"Itérations totales      : {len(ds_iter)}")
print()


# ───────── First attempt to measure the difference between predicted and true stress fields ────────────────────────

#%% Retrieve inputs and outputs for a given sample

i = 0   # traction distribution index
j = 5   # iteration index

idx    = ds_iter.index.index((i, j))
sample = ds_iter[idx]

# Simulate a batch of size 1
batch = {k: v.unsqueeze(0) for k, v in sample.items()
         if hasattr(v, 'unsqueeze')}

with torch.no_grad():
    x, y_true = _batch_to_tensors(batch, device)
    y_pred    = model(x)              # [1, 3, 32, 32]

#%% Extract 2D maps

# Inputs
rho   = x[0, 0].cpu().numpy()        # density        [32, 32]
tx    = x[0, 1].cpu().numpy()        # traction tx    [32, 32]
ty    = x[0, 2].cpu().numpy()        # traction ty    [32, 32]


def extract_stress_maps(y):
    sx  = y[0, 0].cpu().numpy()   # σx  [32, 32]
    sy  = y[0, 1].cpu().numpy()   # σy  [32, 32]
    txy = y[0, 2].cpu().numpy()   # τxy [32, 32]
    return sx, sy, txy


# Ground truth outputs
sx_true, sy_true, txy_true = extract_stress_maps(y_true)

# Predicted outputs
sx_pred, sy_pred, txy_pred = extract_stress_maps(y_pred)

#%% Visualization

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


visualize_in_out(y_true, y_pred)

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

visualize_error(y_true, y_pred, 'MAE')
visualize_error(y_true, y_pred, 'MSE')
visualize_error(y_true, y_pred, 'SMAPE')

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

print("Mean Squared Error:", ErrorMetrics_1D(y_true, y_pred, 'MSE'))
print("Mean Absolute Error:", ErrorMetrics_1D(y_true, y_pred, 'MAE'))
print("Symmetric Mean Absolute Percentage Error:", ErrorMetrics_1D(y_true, y_pred, 'SMAPE'))
print("R² Score:", ErrorMetrics_1D(y_true, y_pred, 'R2'))


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


kernel_size = 3
pad = True

ErrorMetrics_Kernel(y_true, y_pred, kernel_size=kernel_size, pad=pad, strides=(1, 1), TYPE='MAE', plot=True)
ErrorMetrics_Kernel(y_true, y_pred, kernel_size=kernel_size, pad=pad, strides=(1, 1), TYPE='MSE', plot=True)
ErrorMetrics_Kernel(y_true, y_pred, kernel_size=kernel_size, pad=pad, strides=(1, 1), TYPE='SMAPE', plot=True)


# %% Error calculation with proximity in the latent space
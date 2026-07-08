#%%
"""Isolate the traction ROTATION transform, pure Python (no MATLAB).

The traction vector field (built in the same image frame as the density) must
transform under rotation_90 exactly like the density: spatial torch.rot90(k_image)
PLUS a rotation of the vectors themselves. We build both candidate 'correct'
targets (vector rotated by M and by M^T) and compare to what rotation_90 produces.

- match with M   -> augmentation correct (self-consistent)
- match with M^T -> only the vector-rotation direction is flipped
- match neither  -> the rotation transform is structurally broken (permutation)
"""
from pathlib import Path
import numpy as np
import torch
import dataset as ds

BASE = Path(__file__).parents[3]
data = ds.load_mat((BASE / 'HeavyFiles/data/dataset_test.mat').resolve())
di   = ds.IterationDataset(ds.Dataset_TopOpt(data))
s    = ds.IterationSample(di, 30)

tx0, ty0 = s.get_traction_distribution()          # image-frame vector field
M1 = np.array([[0, 1], [-1, 0]])                  # linear part of torch.rot90(1)

for k in (1, 2, 3):
    k_image = (-k) % 4
    # spatial rotation of each component (same as the density)
    txr = torch.rot90(torch.tensor(tx0), k_image, dims=(0, 1)).numpy()
    tyr = torch.rot90(torch.tensor(ty0), k_image, dims=(0, 1)).numpy()

    M = np.linalg.matrix_power(M1, k_image)       # vector rotation candidate
    def apply(Mx):
        return (Mx[0, 0]*txr + Mx[0, 1]*tyr, Mx[1, 0]*txr + Mx[1, 1]*tyr)
    tgt_M   = apply(M)
    tgt_MT  = apply(M.T)

    aug = ds.rotation_90(s.copy(), N_rot=k)
    txa, tya = aug.get_traction_distribution()

    def err(tgt):
        scale = max(np.abs(tx0).max(), np.abs(ty0).max()) + 1e-12
        return max(np.abs(tgt[0]-txa).max(), np.abs(tgt[1]-tya).max()) / scale

    print(f"k={k}: err vs M = {err(tgt_M):.2e}   err vs M^T = {err(tgt_MT):.2e}")

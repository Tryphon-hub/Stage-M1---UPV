#%% data_augmentation.py
#
# Augmentation wrapper around IterationDataset.
# Reuses the user's existing sample-level transforms (rotation_90, symmetry_x,
# symmetry_y) which operate directly on an IterationSample.

import numpy as np
import torch

from dataset import IterationSample


# ── Sample-level transforms (provided by the user) ─────────────────────────
# These functions are imported as-is from the user's module.
# Kept here only for reference; in practice import them from wherever
# they actually live (e.g. from topology_utils import rotation_90, ...).
#
# def permutation_tractions(sample, i, j): ...
# def symmetry_x(sample): ...
# def symmetry_y(sample): ...
# def rotation_90(sample, N_rot=1): ...


def dict_to_sample(sample_dict):
    """
    Build an IterationSample from a dict produced by IterationDataset.__getitem__.

    Uses __new__ instead of __init__ because the dict already holds individual
    tensors (Densities, Tractions, Stress, ...) rather than the raw arrays
    that IterationSample's normal constructor expects (which comes from an
    IterationDataset index lookup). __new__ creates a bare instance whose
    attributes we then fill in directly, exactly like GenTopology does.
    """
    s = IterationSample.__new__(IterationSample)
    s.Tractions         = sample_dict['Tractions']
    s.Densities         = sample_dict['Densities']
    s.Relative_Vol_Frac = sample_dict['Relative_Vol_Frac']
    s.FEM_Stress        = sample_dict['Stress']
    s.FEMc              = sample_dict['FEMc']
    s.c                 = sample_dict['c']
    s.NumIts            = sample_dict['NumIts']
    s.ItsFull           = sample_dict['ItsFull']
    s.TEnd              = sample_dict['TEnd']
    s.UNet_Stress       = None
    return s


def sample_to_dict(sample):
    """
    Convert an IterationSample back into the dict format expected by
    _batch_to_tensors downstream. Mirrors IterationSample's field names
    back onto the dict keys used by Dataset_TopOpt.__getitem__.
    """
    return {
        'Tractions'         : sample.Tractions,
        'Densities'         : sample.Densities,
        'Relative_Vol_Frac' : sample.Relative_Vol_Frac,
        'Stress'            : sample.FEM_Stress,
        'FEMc'              : sample.FEMc,
        'c'                 : sample.c,
        'NumIts'            : sample.NumIts,
        'ItsFull'           : sample.ItsFull,
        'TEnd'              : sample.TEnd,
    }


def random_augment(sample, rotation_90, symmetry_x, symmetry_y):
    """
    Apply a random transform to a single IterationSample:
    a rotation (1, 2, or 3 x 90 deg), a symmetry (horizontal or vertical),
    or both combined.

    The three transform functions are passed in as arguments rather than
    imported directly, so this module stays decoupled from wherever the
    user keeps them (topology_utils.py, a notebook cell, etc).
    """
    choice = np.random.choice(['rot', 'flip', 'both'])

    if choice in ('flip', 'both'):
        if np.random.rand() < 0.5:
            sample = symmetry_x(sample)
        else:
            sample = symmetry_y(sample)

    if choice in ('rot', 'both'):
        N_rot = np.random.randint(1, 4)   # 1, 2, or 3 x 90 deg
        sample = rotation_90(sample, N_rot=N_rot)

    return sample


class AugmentedIterationDataset(torch.utils.data.Dataset):
    """
    Wraps an IterationDataset (or any subset of it, e.g. from random_split)
    and applies a random augmentation to each sample independently, with
    probability p, at __getitem__ time.

    Only wrap the TRAIN split with this class. Validation must stay
    unaugmented so that metrics remain comparable across epochs.

    Usage:
        train_ds, val_ds = torch.utils.data.random_split(ds_iter, [n_train, n_val])
        train_ds = AugmentedIterationDataset(train_ds, p=0.2,
                                              rotation_90=rotation_90,
                                              symmetry_x=symmetry_x,
                                              symmetry_y=symmetry_y)
        # val_ds is left untouched
    """
    def __init__(self, base_dataset, p, rotation_90, symmetry_x, symmetry_y):
        self.base        = base_dataset
        self.p           = p
        self.rotation_90 = rotation_90
        self.symmetry_x  = symmetry_x
        self.symmetry_y  = symmetry_y

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample_dict = self.base[idx]

        if np.random.rand() < self.p:
            sample = dict_to_sample(sample_dict)
            sample = random_augment(sample, self.rotation_90,
                                    self.symmetry_x, self.symmetry_y)
            sample_dict = sample_to_dict(sample)

        return sample_dict

#%% Functions

def permutation_tractions(sample, i , j):
    t=sample.Tractions[0][:,i].clone()

    sample.Tractions[0][:,i] = sample.Tractions[0][:,j]
    sample.Tractions[0][:,j] = t



def symmetry_x(sample):
    new_sample=sample.copy()

    # Density symmetry 
    rho_2d = new_sample.Densities.squeeze().reshape(32, 32)
    rho_flipped = torch.flip(rho_2d, dims=(1,))
    new_sample.Densities = rho_flipped.reshape(1, -1)


    # Traction nodes - permutations
    Permutations = (
        (0,1),
        (7,2),
        (6,3),
        (5,4)
    )
    for (i,j) in Permutations:
        permutation_tractions(new_sample, i, j)

    # Traction nodes - sign change: tx' = -tx 
    new_sample.Tractions[0][0,:] = -new_sample.Tractions[0][0,:]


    # Stress - image symmetry
    sx_2d  = new_sample.FEM_Stress[:, 0].reshape(32, 32)
    sy_2d  = new_sample.FEM_Stress[:, 1].reshape(32, 32)
    txy_2d = new_sample.FEM_Stress[:, 3].reshape(32, 32)

    sx_flipped  = torch.flip(sx_2d,  dims=(1,))
    sy_flipped  = torch.flip(sy_2d,  dims=(1,))
    txy_flipped = torch.flip(txy_2d, dims=(1,))

    new_sample.FEM_Stress[:, 0] = sx_flipped.reshape(-1)
    new_sample.FEM_Stress[:, 1] = sy_flipped.reshape(-1)
    new_sample.FEM_Stress[:, 3] = txy_flipped.reshape(-1)


    # Stress - sign change: tau' = - tau'
    new_sample.FEM_Stress[:, 3] = - new_sample.FEM_Stress[:, 3]


    # UNet Stress - image symmetry
    if new_sample.UNet_Stress is not None:
        sx_2d  = new_sample.UNet_Stress[:, 0].reshape(32, 32)
        sy_2d  = new_sample.UNet_Stress[:, 1].reshape(32, 32)
        txy_2d = new_sample.UNet_Stress[:, 3].reshape(32, 32)

        sx_flipped  = torch.flip(sx_2d,  dims=(1,))
        sy_flipped  = torch.flip(sy_2d,  dims=(1,))
        txy_flipped = torch.flip(txy_2d, dims=(1,))

        new_sample.UNet_Stress[:, 0] = sx_flipped.reshape(-1)
        new_sample.UNet_Stress[:, 1] = sy_flipped.reshape(-1)
        new_sample.UNet_Stress[:, 3] = txy_flipped.reshape(-1)


        # UNet Stress - sign change: tau' = - tau'
        new_sample.FEM_Stress[:, 3] = - new_sample.FEM_Stress[:, 3]


    return new_sample




def symmetry_y(sample):
    new_sample=sample.copy()

    # Density symmetry 
    rho_2d = new_sample.Densities.squeeze().reshape(32, 32)
    rho_flipped = torch.flip(rho_2d, dims=(0,))
    new_sample.Densities = rho_flipped.reshape(1, -1)


    # Traction nodes - permutations
    Permutations = (
        (7,6),
        (0,5),
        (1,4),
        (2,3)
    )
    for (i,j) in Permutations:
        permutation_tractions(new_sample, i, j)

    # Traction nodes - sign change: ty' = -ty 
    new_sample.Tractions[0][1,:] = -new_sample.Tractions[0][1,:]


    # Stress - image symmetry
    sx_2d  = new_sample.FEM_Stress[:, 0].reshape(32, 32)
    sy_2d  = new_sample.FEM_Stress[:, 1].reshape(32, 32)
    txy_2d = new_sample.FEM_Stress[:, 3].reshape(32, 32)

    sx_flipped  = torch.flip(sx_2d,  dims=(0,))
    sy_flipped  = torch.flip(sy_2d,  dims=(0,))
    txy_flipped = torch.flip(txy_2d, dims=(0,))

    new_sample.FEM_Stress[:, 0] = sx_flipped.reshape(-1)
    new_sample.FEM_Stress[:, 1] = sy_flipped.reshape(-1)
    new_sample.FEM_Stress[:, 3] = txy_flipped.reshape(-1)


    # Stress - sign change: tau' = - tau'
    new_sample.FEM_Stress[:, 3] = - new_sample.FEM_Stress[:, 3]


    # UNet Stress - image symmetry
    if new_sample.UNet_Stress is not None:
        sx_2d  = new_sample.UNet_Stress[:, 0].reshape(32, 32)
        sy_2d  = new_sample.UNet_Stress[:, 1].reshape(32, 32)
        txy_2d = new_sample.UNet_Stress[:, 3].reshape(32, 32)

        sx_flipped  = torch.flip(sx_2d,  dims=(0,))
        sy_flipped  = torch.flip(sy_2d,  dims=(0,))
        txy_flipped = torch.flip(txy_2d, dims=(0,))

        new_sample.UNet_Stress[:, 0] = sx_flipped.reshape(-1)
        new_sample.UNet_Stress[:, 1] = sy_flipped.reshape(-1)
        new_sample.UNet_Stress[:, 3] = txy_flipped.reshape(-1)


        # UNet Stress - sign change: tau' = - tau'
        new_sample.FEM_Stress[:, 3] = - new_sample.FEM_Stress[:, 3]


    return new_sample



def rotation_90(sample, N_rot=1):
    new_sample = sample.copy()
    k       = N_rot % 4
    k_image = (-k) % 4   # image tourne dans le sens oppose des tractions

    # ── Density - image rotation ──────────────────────────────────────────
    rho_2d = new_sample.Densities.squeeze().reshape(32, 32)
    rho_rot = torch.rot90(rho_2d, k_image, dims=(0, 1))
    new_sample.Densities = rho_rot.reshape(1, -1)

    # ── Traction nodes - permutation (sens inchange, deja valide) ───────────
    Replacements = (
        (0, 2), (7, 1), (6, 0), (5, 7),
        (4, 6), (3, 5), (2, 4), (1, 3),
    )
    T_old = sample.Tractions[0].clone()
    for _ in range(k):
        T_new = T_old.clone()
        for (i, j) in Replacements:
            T_new[:, i] = T_old[:, j]
        T_old = T_new
    new_sample.Tractions[0] = T_old

    # ── Traction nodes - sign change (sens inchange, deja valide) ───────────
    for _ in range(k):
        tx_old = new_sample.Tractions[0][0, :].clone()
        ty_old = new_sample.Tractions[0][1, :].clone()
        new_sample.Tractions[0][0, :] = -ty_old
        new_sample.Tractions[0][1, :] =  tx_old

    # ── Stress - image rotation (meme sens que la densite) ─────────────────
    sx_2d  = new_sample.FEM_Stress[:, 0].reshape(32, 32)
    sy_2d  = new_sample.FEM_Stress[:, 1].reshape(32, 32)
    txy_2d = new_sample.FEM_Stress[:, 3].reshape(32, 32)

    sx_rot  = torch.rot90(sx_2d,  k_image, dims=(0, 1))
    sy_rot  = torch.rot90(sy_2d,  k_image, dims=(0, 1))
    txy_rot = torch.rot90(txy_2d, k_image, dims=(0, 1))

    if k % 2 == 1:
        new_sample.FEM_Stress[:, 0] = sy_rot.reshape(-1)
        new_sample.FEM_Stress[:, 1] = sx_rot.reshape(-1)
        new_sample.FEM_Stress[:, 3] = -txy_rot.reshape(-1)
    else:
        new_sample.FEM_Stress[:, 0] = sx_rot.reshape(-1)
        new_sample.FEM_Stress[:, 1] = sy_rot.reshape(-1)
        new_sample.FEM_Stress[:, 3] = txy_rot.reshape(-1)

    if new_sample.UNet_Stress is not None:
        sx_2d  = new_sample.UNet_Stress[:, 0].reshape(32, 32)
        sy_2d  = new_sample.UNet_Stress[:, 1].reshape(32, 32)
        txy_2d = new_sample.UNet_Stress[:, 3].reshape(32, 32)

        sx_rot  = torch.rot90(sx_2d,  k_image, dims=(0, 1))
        sy_rot  = torch.rot90(sy_2d,  k_image, dims=(0, 1))
        txy_rot = torch.rot90(txy_2d, k_image, dims=(0, 1))

        if k % 2 == 1:
            new_sample.UNet_Stress[:, 0] = sy_rot.reshape(-1)
            new_sample.UNet_Stress[:, 1] = sx_rot.reshape(-1)
            new_sample.UNet_Stress[:, 3] = -txy_rot.reshape(-1)
        else:
            new_sample.UNet_Stress[:, 0] = sx_rot.reshape(-1)
            new_sample.UNet_Stress[:, 1] = sy_rot.reshape(-1)
            new_sample.UNet_Stress[:, 3] = txy_rot.reshape(-1)

    return new_sample







#%% Test

if __name__ == '__main__':
    path = (Path.cwd().parents[2] / 'HeavyFiles/data/dataset_test.mat').resolve()

    data = load_mat(path)
    dataset = Dataset_TopOpt(data)

    data_iter = IterationDataset(dataset)

    sample = IterationSample(data_iter, 30)
    
    print('original sample')
    sample.plot()
    sample.plot_inputs()
    sample.plot_outputs('FEM')

    # new_sample=symmetry_y(sample)
    new_sample = rotation_90(sample, 1)

    print('new sample')   
    new_sample.plot()
    new_sample.plot_inputs()
    new_sample.plot_outputs('FEM')


    
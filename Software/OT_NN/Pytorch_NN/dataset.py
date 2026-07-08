#%%  Libraries
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
import scipy.io
import os
import matplotlib.pyplot as plt



# ─── Dataset indexing ────────────────────────────────────────────────────────
#
#  The dataset is indexed by (i, j) :
#    i = traction distribution index   [0 .. len(ds_base)-1]
#    j = iteration index within distribution [0 .. NumIts[i]-1]
#
#  ds_iter.index holds the flat list of all valid (i, j) pairs.
#
#  Examples :
#
#    # Access a sample by its global index
#    sample = ds_iter[42]
#
#    # Access distribution i=2, iteration j=10
#    idx = ds_iter.index.index((2, 10))
#    sample = ds_iter[idx]
#
#    # Loop over all iterations of distribution i=0
#    i = 0
#    n_its = int(ds_base.NumIts[i])
#    for j in range(n_its):
#        idx    = ds_iter.index.index((i, j))
#        sample = ds_iter[idx]
#
#    # Predict stress fields for a given sample
#    sample  = ds_iter[42]
#    batch   = {k: v.unsqueeze(0) for k, v in sample.items()
#               if hasattr(v, 'unsqueeze')}   # simulate a batch of size 1
#    with torch.no_grad():
#        x, y = _batch_to_tensors(batch, device)
#        pred = model(x)                      # [1, 3, 32, 32]
#    # pred[0, 0] → σx   pred[0, 1] → σy   pred[0, 2] → τxy



#%% Useful functions

def load_mat(filepath: str) -> dict:
    """
    Load a MATLAB .mat file into a dict, transparently handling both formats.

    Tries scipy.io.loadmat first (MATLAB v7 and earlier); on the v7.3/HDF5
    format it falls back to h5py and rebuilds the expected keys, undoing the
    axis transposition that HDF5 applies relative to MATLAB.

    Parameters
    ----------
    filepath : str — path to the .mat file.

    Returns
    -------
    dict — the dataset variables keyed by name.
    """
    import scipy.io
    import h5py

    try:
        return scipy.io.loadmat(filepath, squeeze_me=True, struct_as_record=False)
    except NotImplementedError:
        with h5py.File(filepath, 'r') as f:
            def read(key):
                return np.array(f[key]).squeeze()

            # HDF5 transposes all arrays vs MATLAB
            data = {
                'MeshData'          : None,
                'Tractions'         : read('Tractions'),          # (N, 8, 2) → transpose needed
                'Relative_Vol_Frac' : read('Relative_Vol_Frac'),
                'Rel_Density'       : read('Rel_Density'),
                'NumIts'            : read('NumIts').astype(int),
                'ItsFull'           : read('ItsFull').astype(int),
                'TEnd'              : read('TEnd'),
                'FEMc'              : read('FEMc'),
                'Stress'            : read('Stress'),
                'Densities'         : read('Densities'),
                'c'                 : read('c'),
            }
        return data



def inspect_mat(filepath: str) -> None:
    """
    Print a summary of the raw content of a .mat file.
    For each key, displays type, shape and dtype if it is an ndarray.

    Parameters:
        filepath (str): Path to the .mat file.

    Returns:
        None
    """
    mat = load_mat(filepath)
    print(f"\n📂 {filepath}")
    for k, v in mat.items():
        if isinstance(v, np.ndarray):
            print(f"  [{k}]  shape={v.shape}  dtype={v.dtype}")
        else:
            print(f"  [{k}]  type={type(v).__name__}")
    print()


#%% Dataset class for U-net 

class Dataset_TopOpt(Dataset):
    """
    PyTorch Dataset containing N topology optimizations loaded from a .mat file.

    Each optimization corresponds to a traction distribution (boundary conditions)
    and contains the full iteration history of the SIMP algorithm.

    Indexing : dataset[i, j]
        i : traction distribution index [0 .. N-1]
        j : iteration index             [0 .. NumIts[i]-1] or [-NumIts[i] .. -1]

    Attributes:
        mesh (mat_struct)           : finite element mesh data.
        Tractions (ndarray)         : nodal force distributions, shape (2, 8, N).
        Relative_Vol_Frac (ndarray) : target volume fraction per case, shape (N,).
        Rel_Density (ndarray)       : converged densities (last iteration), shape (NumEls, N).
        Stress (ndarray)            : stress history, object array (N,) of (NumEls, 6, n_iter).
        Densities (ndarray)         : density history, object array (N,) of (NumEls, n_iter).
        c (ndarray)                 : compliance per iteration, object array (N,).
        NumIts (ndarray)            : number of iterations per case, shape (N,).
        ItsFull (ndarray)           : number of full iterations per case, shape (N,).
        FEMc (ndarray)              : FEM compliance per iteration, object array (N,).
        TEnd (float)                : total computation time.
        Ener (ndarray)              : pixel-wise energy (Stress .* Strain) of the
                                      FIRST image of each traction distribution,
                                      object array (N,) of (NumEls, 6).
                                      Per-distribution only — independent of iteration j.
    """

    def __init__(self, dataset):
        """
        Initialize the dataset from a dictionary loaded via load_mat().

        Parameters:
            dataset (dict): Dictionary returned by load_mat().
        """
        self.dataset = dataset
        self.mesh                = dataset['MeshData']
        self.Tractions           = dataset['Tractions']
        self.Relative_Vol_Frac   = dataset['Relative_Vol_Frac']
        self.Rel_Density         = dataset['Rel_Density']
        self.Stress              = dataset['Stress']
        self.Densities           = dataset['Densities']
        self.c                   = dataset['c']
        self.NumIts              = dataset['NumIts']
        self.ItsFull             = dataset['ItsFull']
        self.FEMc                = dataset['FEMc']
        self.TEnd                = dataset['TEnd']
        self.Ener                = dataset['Ener']

    def __len__(self):
        """
        Return the number of traction distributions in the dataset.

        Returns:
            int: Number of cases N.
        """
        return len(np.atleast_1d(self.NumIts))

    def __getitem__(self, idx):
        """
        Return a dictionary of tensors for the (i, j) pair.

        Parameters:
            idx (tuple): Pair (i, j) where i is the distribution index
                         and j the iteration index (negative indices supported).

        Returns:
            dict: Dictionary containing Tractions, Densities, Stress,
                  Relative_Vol_Frac, c, FEMc, NumIts, ItsFull, TEnd.
        """
        i, j = idx
        d = self.Densities
        density_ij = d[:, j] if d.ndim == 2 else d[i][:, j]

        # Ener is per-distribution (first image only) — same for every iteration j.
        ener_i = self.Ener if self.Ener.ndim == 2 else self.Ener[i]

        numits = np.atleast_1d(self.NumIts)
        vf     = np.atleast_1d(self.Relative_Vol_Frac)
        return {
            'Tractions'         : torch.from_numpy(self.Tractions[:, :, i]).float().unsqueeze(0),
            'Densities'         : torch.from_numpy(density_ij).float().unsqueeze(0),
            'Relative_Vol_Frac' : torch.tensor(float(vf[i])).float(),
            'Stress'            : torch.from_numpy(self.Stress[i][:, :, j]).float(),
            'Ener'              : torch.from_numpy(ener_i).float(),
            'FEMc'              : torch.tensor(self.FEMc[i][j]).float(),
            'c'                 : torch.tensor(self.c[i][j]).float(),
            'NumIts'            : torch.tensor(float(numits[i])).float(),
            'ItsFull'           : torch.tensor(self.ItsFull[i]).float(),
            'TEnd'              : torch.tensor(float(self.TEnd)).float(),
        }

    def __repr__(self):
        """Return a short summary: number of distributions and their NumIts."""
        numits = np.atleast_1d(self.NumIts).tolist()
        return (f"Dataset_TopOpt\n"
                f"  Num distributions : {len(numits)}\n"
                f"  NumIts : {numits}")

    def get_density(self, i, j):
        """
        Return the density vector for case i at iteration j.
        Handles both cases: N=1 (Densities is 2D) and N>1 (Densities is object array).

        Parameters:
            i (int): Traction distribution index.
            j (int): Iteration index.

        Returns:
            ndarray: Density vector (NumEls,).
        """
        d = self.Densities
        return d[:, j] if d.ndim == 2 else d[i][:, j]

    def n_iter(self, i):
        """
        Number of iterations actually stored for case i, read from the
        Densities array itself rather than NumIts.

        NumIts records the full SIMP iteration count, but datasets generated
        with save_last_only=true store only [first, last] (2 columns). Relying
        on NumIts would walk past the stored columns. Densities.shape[1] is the
        ground truth for how many iterations are available.

        Parameters:
            i (int): Traction distribution index.

        Returns:
            int: Number of stored iterations for case i.
        """
        d = self.Densities
        return d.shape[1] if d.ndim == 2 else d[i].shape[1]

    def get_energy(self, i):
        """
        Return the pixel-wise energy (NumEls, 6) of the first image for case i.
        Handles both cases: N=1 (Ener is 2D) and N>1 (Ener is object array).

        Parameters:
            i (int): Traction distribution index.

        Returns:
            ndarray: Energy field (NumEls, 6).
        """
        e = self.Ener
        return e if e.ndim == 2 else e[i]

    def get_series(self, i: int) -> 'Dataset_TopOpt':
        """
        Return a Dataset_TopOpt restricted to distribution i.

        Parameters:
            i (int): Traction distribution index.

        Returns:
            Dataset_TopOpt: Dataset containing only case i.
        """
        numits = np.atleast_1d(self.NumIts)
        vf     = np.atleast_1d(self.Relative_Vol_Frac)
        sub = {
            'MeshData'          : self.mesh,
            'Tractions'         : self.Tractions[:, :, i:i+1],
            'Relative_Vol_Frac' : vf[i:i+1],
            'Rel_Density'       : self.Rel_Density[:, i:i+1],
            'Stress'            : self.Stress[i:i+1],
            'Densities'         : self.Densities[i],       # (NumEls, n_iter) — no N dimension
            'c'                 : self.c[i:i+1],
            'NumIts'            : numits[i:i+1],
            'ItsFull'           : self.ItsFull[i:i+1],
            'FEMc'              : self.FEMc[i:i+1],
            'TEnd'              : self.TEnd,
            'Ener'              : self.Ener[i:i+1],
        }
        return Dataset_TopOpt(sub)

    def filtre_dataset(self, rho_min: float = 0.15, rho_max: float = 0.85) -> 'Dataset_TopOpt':
        """
        Return a filtered Dataset_TopOpt: keeps only distributions
        whose target volume fraction AND all iterations are within [rho_min, rho_max].

        Parameters:
            rho_min (float): Minimum volume fraction. Default: 0.15.
            rho_max (float): Maximum volume fraction. Default: 0.85.

        Returns:
            Dataset_TopOpt: Filtered dataset.
        """
        numits = np.atleast_1d(self.NumIts)
        vf     = np.atleast_1d(self.Relative_Vol_Frac)

        valid = [
            i for i in range(len(self))
            if rho_min <= float(vf[i]) <= rho_max
            and all(
                rho_min <= self.get_density(i, j).mean() <= rho_max
                for j in range(self.n_iter(i))
            )
        ]

        sub = {
            'MeshData'          : self.mesh,
            'Tractions'         : self.Tractions[:, :, valid],
            'Relative_Vol_Frac' : vf[valid],
            'Rel_Density'       : self.Rel_Density[:, valid],
            'Stress'            : self.Stress[valid],
            'Densities'         : self.Densities[valid],
            'c'                 : self.c[valid],
            'NumIts'            : numits[valid],
            'ItsFull'           : self.ItsFull[valid],
            'FEMc'              : self.FEMc[valid],
            'TEnd'              : self.TEnd,
            'Ener'              : self.Ener[valid],
        }
        return Dataset_TopOpt(sub)

    def plot_densities(self, iteration=-1):
        """
        Plot the density distribution of each case in the dataset.
        By default shows the last iteration of each case.

        Parameters:
            iteration (int): iteration index to display for each case (default: -1, last).
        """
        N = len(self)

        n_cols = int(np.ceil(np.sqrt(N)))
        n_rows = int(np.ceil(N / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5*n_cols, 2.5*n_rows))
        axes = np.array(axes).reshape(-1)

        for i in range(N):
            ax = axes[i]

            # resolve iteration index per case (handles negative indices)
            j = iteration if iteration >= 0 else self.n_iter(i) + iteration
            topo = self.get_density(i, j)

            img_size = int(np.sqrt(len(topo)))
            img = topo.reshape(img_size, img_size)

            ax.imshow(img, cmap='gray_r', origin='lower', vmin=0, vmax=1)
            ax.set_aspect('equal')
            ax.axis('off')
            ax.set_title(f'#{i}', fontsize=12)

        # hide unused subplots
        for i in range(N, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

    def plot_all_densities(self):
        """
        Plot every density distribution (all iterations of all samples) in the dataset.
        Each subplot is titled 'sample {i}\niteration {j}'.
        """
        N = len(self)

        # Build list of (i, j) pairs for every iteration of every sample
        pairs = [(i, j) for i in range(N) for j in range(self.n_iter(i))]
        n_plots = len(pairs)

        n_cols = int(np.ceil(np.sqrt(n_plots)))
        n_rows = int(np.ceil(n_plots / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2*n_cols, 2.4*n_rows))
        axes = np.array(axes).reshape(-1)

        for plot_idx, (i, j) in enumerate(pairs):
            ax = axes[plot_idx]
            topo = self.get_density(i, j)

            img_size = int(np.sqrt(len(topo)))
            img = topo.reshape(img_size, img_size)

            ax.imshow(img, cmap='gray_r', origin='lower', vmin=0, vmax=1)
            ax.set_aspect('equal')
            ax.axis('off')
            ax.set_title(f'sample {i}\niteration {j}', fontsize=8)

        # Hide unused subplots
        for plot_idx in range(n_plots, len(axes)):
            axes[plot_idx].axis('off')

        plt.tight_layout()
        plt.show()

    def normalize_dataset(self):
        """
        Return a new Dataset_TopOpt with Tractions, Stress, c, FEMc, and Ener
        rescaled, exploiting the linearity of elasticity:
        - Tractions and Stress scale linearly:   σ(λf) = λσ(f)
        - Compliance scales quadratically:        c(λf) = λ² c(f)
        - Strain energy (Ener = stress * strain)
            scales quadratically as well, since strain itself scales linearly
            with the load:                          Ener(λf) = λ² Ener(f)

        Each distribution is normalized independently by its own maximum
        absolute traction value (tx or ty).

        Returns:
            Dataset_TopOpt: new dataset with normalized Tractions, Stress, c,
                            FEMc, and Ener.
        """
        N = len(self)
        Tractions_norm = self.Tractions.copy().astype(np.float64)

        Stress_norm = (np.empty(N, dtype=object) 
                    if self.Stress.dtype == object else self.Stress.copy())
        c_norm    = (np.empty(N, dtype=object) 
                    if self.c.dtype == object else self.c.copy())
        FEMc_norm = (np.empty(N, dtype=object) 
                    if self.FEMc.dtype == object else self.FEMc.copy())
        Ener_norm = (np.empty(N, dtype=object) 
                    if self.Ener.dtype == object else self.Ener.copy())

        for i in range(N):
            s = np.abs(self.Tractions[:, :, i]).max()
            if s == 0:
                s = 1.0

            Tractions_norm[:, :, i] = self.Tractions[:, :, i] / s
            Stress_norm[i] = self.Stress[i] / s
            c_norm[i]    = self.c[i]    / (s ** 2)
            FEMc_norm[i] = self.FEMc[i] / (s ** 2)
            Ener_norm[i] = self.Ener[i] / (s ** 2)

        sub = {
            'MeshData'          : self.mesh,
            'Tractions'         : Tractions_norm,
            'Relative_Vol_Frac' : self.Relative_Vol_Frac,
            'Rel_Density'       : self.Rel_Density,
            'Stress'            : Stress_norm,
            'Ener'              : Ener_norm,
            'Densities'         : self.Densities,
            'c'                 : c_norm,
            'NumIts'            : self.NumIts,
            'ItsFull'           : self.ItsFull,
            'FEMc'              : FEMc_norm,
            'TEnd'              : self.TEnd,
        }
        return Dataset_TopOpt(sub)

#%% Dataset for all iterations

def _to_object_array(arr):
    """Force any array into an object array (N,) where each element is a sub-array."""
    if arr.dtype == object:
        return arr.flatten()
    else:
        # 3D array (NumEls, 6, n_iter) or 2D (NumEls, n_iter) — wrap in object array
        wrapped = np.empty(1, dtype=object)
        wrapped[0] = arr
        return wrapped
class IterationDataset(Dataset):
    """
    PyTorch Dataset that exposes each (i, j) pair as an independent sample,
    allowing the DataLoader to iterate over all iterations of all distributions.

    Only samples whose mean density lies within [0.15, 0.85] are kept.

    Attributes:
        dataset (Dataset_TopOpt)    : base dataset.
        index (list of tuple)       : flat list of valid (i, j) pairs.
        last_iteration_index (list) : global indices of the last iteration of each distribution.
    """

    def __init__(self, dataset: Dataset_TopOpt):
        """
        Build the flat list of valid (i, j) pairs.

        Parameters:
            dataset (Dataset_TopOpt): Base dataset containing all distributions.
        """
        self.dataset = dataset
        vf     = np.atleast_1d(dataset.Relative_Vol_Frac)

        self.index = [
            (i, j)
            for i in range(len(dataset))
            for j in range(dataset.n_iter(i))
            if 0.15 <= self.get_density(dataset, i, j).mean() <= 0.85
            and 0.15 <= float(vf[i]) <= 0.85
        ]

        self.last_iteration_index = [
            self.index.index((i, dataset.n_iter(i) - 1))
            for i in range(len(dataset))
            if (i, dataset.n_iter(i) - 1) in self.index
        ]

    def get_density(self, dataset, i, j):
        """
        Return the density vector for case i at iteration j.
        Handles both cases: N=1 (Densities is 2D) and N>1 (Densities is object array).

        Parameters:
            dataset (Dataset_TopOpt): Source dataset.
            i (int): Distribution index.
            j (int): Iteration index.

        Returns:
            ndarray: Density vector (NumEls,).
        """
        d = dataset.Densities
        if d.ndim == 2:      # N=1 : (NumEls, n_iter)
            return d[:, j]
        else:                # N>1 : object array (N,) → each element is (NumEls, n_iter)
            return d[i][:, j]

    def __len__(self):
        """
        Return the total number of valid (i, j) samples.

        Returns:
            int: Number of samples.
        """
        return len(self.index)

    def __getitem__(self, idx):
        """
        Return the tensor dictionary for the sample at global index idx.

        Parameters:
            idx (int): Global index in self.index.

        Returns:
            dict: Tensor dictionary (see Dataset_TopOpt.__getitem__).
        """
        return self.dataset[self.index[idx]]

    def __repr__(self):
        """Return a short summary: number of samples and the (i, j) index list."""
        return (f"IterationDataset\n"
                f"  Sample       : {len(self.index)}\n"
                f"  Index (i,j)  : {self.index}")

    def get_traction_distribution(self, idx):
        """
        Return the traction images tx, ty for the sample at global index idx.

        Parameters:
            idx (int): Global index in self.index.

        Returns:
            ndarray: Traction images, shape (2, img_size, img_size).
        """
        sample = IterationSample(self, idx)
        return sample.get_traction_distribution()

    def __add__(self, IterData: 'IterationDataset') -> 'IterationDataset':
        """
        Merge two IterationDatasets with correct indexation.
        The indices (i, j) of IterData are shifted by the number of distributions
        in the current dataset.

        Parameters:
            IterData (IterationDataset): Dataset to merge with.

        Returns:
            IterationDataset: New merged dataset.
        """
        ds1 = self.dataset
        ds2 = IterData.dataset

        n_shift = len(ds1)  # offset to apply to i indices of IterData

        # Merge base datasets
        sub = {
            'MeshData'          : ds1.mesh,
            'Tractions'         : np.concatenate([ds1.Tractions, ds2.Tractions], axis=-1),
            'Relative_Vol_Frac' : np.concatenate([np.atleast_1d(ds1.Relative_Vol_Frac), np.atleast_1d(ds2.Relative_Vol_Frac)]),
            'Rel_Density'       : np.concatenate([ds1.Rel_Density, ds2.Rel_Density], axis=-1),
            'NumIts'            : np.concatenate([np.atleast_1d(ds1.NumIts), np.atleast_1d(ds2.NumIts)]),
            'ItsFull'           : np.concatenate([np.atleast_1d(ds1.ItsFull), np.atleast_1d(ds2.ItsFull)]),
            'TEnd'              : ds1.TEnd,
            'Stress'    : np.concatenate([_to_object_array(ds1.Stress),    _to_object_array(ds2.Stress)]),
            'Densities' : np.concatenate([_to_object_array(ds1.Densities), _to_object_array(ds2.Densities)]),
            'c'         : np.concatenate([_to_object_array(ds1.c),         _to_object_array(ds2.c)]),
            'FEMc'      : np.concatenate([_to_object_array(ds1.FEMc),      _to_object_array(ds2.FEMc)]),
            'Ener'      : np.concatenate([_to_object_array(ds1.Ener),      _to_object_array(ds2.Ener)]),
        }

        merged_base = Dataset_TopOpt(sub)
        merged_iter = IterationDataset.__new__(IterationDataset)
        merged_iter.dataset = merged_base
        merged_iter.extra_samples = []

        # Merge indices with shift
        merged_iter.index = self.index + [(i + n_shift, j) for i, j in IterData.index]

        # Merge last_iteration_index with shift
        merged_iter.last_iteration_index = (
            self.last_iteration_index +
            [idx + len(self.index) for idx in IterData.last_iteration_index]
        )

        return merged_iter

#%% IterationSample class for visualization and attribute access of a single (i, j) sample
#In case we want to access one specific iteration sample directly

class IterationSample:
    """
    Represents a single (i, j) sample with attribute access and visualization methods.
    Unlike the dictionary returned by IterationDataset.__getitem__,
    this class provides named field access and stores the U-Net prediction.

    Attributes:
        Tractions (Tensor)         : nodal tractions, shape (1, 2, 8).
        Densities (Tensor)         : densities at iteration j, shape (1, NumEls).
        Relative_Vol_Frac (Tensor) : target volume fraction, scalar.
        FEM_Stress (Tensor)        : FEM stress fields, shape (NumEls, 6).
        UNet_Stress (Tensor|None)  : U-Net predicted stress, None before prediction.
        c (Tensor)                 : compliance at iteration j, scalar.
        FEMc (Tensor)              : FEM compliance, scalar.
        NumIts (Tensor)            : number of iterations for this case, scalar.
        ItsFull (Tensor)           : number of full iterations, scalar.
        TEnd (Tensor)              : total computation time, scalar.
    """

    def __init__(self, dataset: IterationDataset, idx: int):
        """
        Initialize the sample from an IterationDataset and a global index.

        Parameters:
            dataset (IterationDataset): Source dataset.
            idx (int): Global index in dataset.index.
        """
        sample = dataset[idx]
        self.Tractions         = sample['Tractions']
        self.Densities         = sample['Densities']
        self.Relative_Vol_Frac = sample['Relative_Vol_Frac']
        self.FEM_Stress        = sample['Stress']
        self.Ener              = sample['Ener']
        self.FEMc              = sample['FEMc']
        self.c                 = sample['c']
        self.NumIts            = sample['NumIts']
        self.ItsFull           = sample['ItsFull']
        self.TEnd              = sample['TEnd']
        self.UNet_Stress       = None  # Filled after prediction

    def __repr__(self):
        """Return a multi-line summary of the sample's tensor shapes and scalars."""
        return (f"IterationSample\n"
                f"  Tractions         : {tuple(self.Tractions.shape)}\n"
                f"  Densities         : {tuple(self.Densities.shape)}\n"
                f"  FEM_Stress        : {tuple(self.FEM_Stress.shape)}\n"
                f"  Ener              : {tuple(self.Ener.shape)}\n"
                f"  UNet_Stress       : {tuple(self.UNet_Stress.shape) if self.UNet_Stress is not None else 'Not computed'}\n"
                f"  Relative_Vol_Frac : {self.Relative_Vol_Frac.item():.3f}\n"
                f"  c                 : {self.c.item():.6f}\n"
                f"  NumIts            : {int(self.NumIts.item())}\n"
                f"  ItsFull           : {int(self.ItsFull.item())}\n"
                f"  TEnd              : {self.TEnd.item():.4f}")

    def get_traction_distribution(self, width=1):
        """
        Build the 2D traction images tx and ty (global components) along the 4
        edges of the square domain, by linear interpolation of the boundary
        nodes. The 8 nodes are distributed 2 per edge.

        The stored tractions are per-edge (Tn, Tt) = (normal, tangential), so
        they are converted to global (tx, ty) and each node placed at its true
        physical boundary position (see `tractions_to_global` /
        `node_positions_pixels`). This is BOTH the U-Net input builder and the
        display source, and must stay consistent with `train._tractions_to_maps`.

        Each edge is drawn as a band of `width` pixels, replicated inward toward
        the center. Contributions accumulate (+=), so corner pixels sum both edges.

        Parameters:
            width (int): Border thickness in pixels (>= 1). Default: 1.

        Returns:
            ndarray: Stacked traction images, shape (2, img_size, img_size).
                     Channel 0: tx (global), Channel 1: ty (global).
        """
        width     = max(1, int(round(width)))   # border thickness is in pixels
        img_size  = int(np.sqrt(self.Densities.shape[1]))
        tx        = np.zeros((img_size, img_size))
        ty        = np.zeros((img_size, img_size))

        # per-edge (Tn, Tt) -> global (tx, ty); nodes at their physical positions
        T_global = tractions_to_global(self.Tractions.squeeze().numpy())  # (2, 8)
        Points   = node_positions_pixels(img_size, inclusive=False)       # (8, 2)

        center = (img_size - 1) / 2.0
        for k in range(0, 8, 2):
            p1 = Points[k]
            p2 = Points[k+1]
            xs = np.round(np.linspace(p1[0], p2[0], img_size)).astype(int)
            ys = np.round(np.linspace(p1[1], p2[1], img_size)).astype(int)
            prof_x = np.linspace(T_global[0, k], T_global[0, k+1], img_size)
            prof_y = np.linspace(T_global[1, k], T_global[1, k+1], img_size)

            # Inward offset (perpendicular to the edge, toward the center)
            mid    = (p1 + p2) / 2.0
            inward = np.sign(np.round(center - mid)).astype(int)  # (dx, dy)

            for d in range(width):
                xo = np.clip(xs + inward[0] * d, 0, img_size - 1)
                yo = np.clip(ys + inward[1] * d, 0, img_size - 1)
                tx[yo, xo] += prof_x
                ty[yo, xo] += prof_y

        return np.stack([tx, ty], axis=0)  # (2, img_size, img_size)

    def plot(self,scale_force = 10) -> None:
        """
        Display the optimized topology with boundary force distributions.
        Red: nodal forces. Blue: normal force distributions per edge.

        Returns:
            None
        """
        
        cadre       = int(scale_force)

        topo     = self.Densities.squeeze().numpy()
        img_size = int(np.sqrt(len(topo)))
        img      = topo.reshape(img_size, img_size)

        fig, ax = plt.subplots()
        ax.imshow(img, cmap='gray_r', origin='lower',
                  extent=[0, img_size, 0, img_size], vmin=0, vmax=1)
        ax.set_xlim(-cadre, img_size + cadre)
        ax.set_ylim(-cadre, img_size + cadre)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Relative Density (OT Result)', fontsize=16)

        cb = fig.colorbar(ax.images[0], ax=ax, ticks=np.arange(0, 1.2, 0.2))
        cb.mappable.set_clim(0, 1)

        # Physical node positions and global (tx, ty) vectors (the stored rows
        # are normal/tangential, not global x/y).
        Points  = node_positions_pixels(img_size, inclusive=True)          # (8, 2)
        T_scale = tractions_to_global(self.Tractions.squeeze().numpy())    # (2, 8)
        T_scale = (T_scale * scale_force).T                                # (8, 2)

        q = None
        for k in range(8):
            sx, sy = Points[k] 
            tx, ty = T_scale[k]
            q = ax.quiver(sx, sy, tx, ty, angles='xy', scale_units='xy', scale=1,
                          color='r', linewidth=1, headwidth=2)

        b = None
        for k in range(0, 8, 2):
            edge   = Points[k+1] - Points[k]
            normal = np.array([-edge[1], edge[0]])
            normal = normal / np.linalg.norm(normal)

            border_1 = T_scale[k]   @ normal
            border_2 = T_scale[k+1] @ normal

            p1 = Points[k]   
            p2 = Points[k+1] 

            b = ax.quiver(p1[0], p1[1], border_1*normal[0], border_1*normal[1],
                          angles='xy', scale_units='xy', scale=1,
                          color='b', linewidth=1, headwidth=2)
            ax.quiver(p2[0], p2[1], border_2*normal[0], border_2*normal[1],
                      angles='xy', scale_units='xy', scale=1,
                      color='b', linewidth=1, headwidth=2)
            ax.plot([p1[0] + border_1*normal[0], p2[0] + border_2*normal[0]],
                    [p1[1] + border_1*normal[1], p2[1] + border_2*normal[1]],
                    'b-', linewidth=1)

        ax.legend([b, q], ['Normal force distributions', 'Side nodal forces'],
                  fontsize=14, handlelength=0.5, handleheight=0.01)
        plt.tight_layout()
        plt.show()

    def plot_inputs(self, TITLE=None, width=1, SAVE_DIR=None, SEPARATED_IMG=False) -> None:
        """
        Display the 3 U-Net inputs: densities, tx, ty.
        gray_r colormap [0,1] for densities, symmetric RdBu for tractions.

        Parameters:
            TITLE (str|None)     : Figure title. Default: 'Inputs'.
            width (int)          : Border thickness (in pixels) used to display the
                                traction distributions. Each of the 4 edges is drawn
                                as a band of `width` pixels, forces summing at the
                                corners. Default: 1.
            SAVE_DIR (str|None)  : Directory to save the figure. Default: None.
            SEPARATED_IMG (bool) : If True, save the 3 images separately without
                                title, colorbar or axis labels. Default: False.

        Returns:
            None
        """
        topo     = self.Densities.squeeze().numpy()
        img_size = int(np.sqrt(len(topo)))
        img      = topo.reshape(img_size, img_size)

        tx_ty   = self.get_traction_distribution(width)
        vmax_tx = np.abs(tx_ty[0]).max()
        vmax_ty = np.abs(tx_ty[1]).max()

        img_data_list = [img,      tx_ty[0], tx_ty[1]]
        title_list    = ['Densities', 'tx',  'ty'    ]
        cmap_list     = ['gray_r', 'RdBu',   'RdBu'  ]
        vmin_list     = [0,        -vmax_tx, -vmax_ty ]
        vmax_list     = [1,         vmax_tx,  vmax_ty ]

        # ── Mode separated images ─────────────────────────────────────────────────
        if SEPARATED_IMG:
            for img_data, name, cmap, vmin, vmax in zip(
                img_data_list, title_list, cmap_list, vmin_list, vmax_list
            ):
                fig, ax = plt.subplots(1, 1, figsize=(4, 4))
                ax.imshow(img_data, cmap=cmap, origin='lower',
                        vmin=vmin, vmax=vmax)

                ax.axis('off')  # supprime ticks, labels ET spines (bordures)
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

                if SAVE_DIR is not None:
                    plt.savefig(f"{SAVE_DIR}/{name}.png", dpi=150)
                plt.show()
            return

        # ── Mode combined figure ──────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        for ax, img_data, title, cmap, vmin, vmax in zip(
            axes, img_data_list, title_list, cmap_list, vmin_list, vmax_list
        ):
            im = ax.imshow(img_data, cmap=cmap, origin='lower',
                        extent=[0, img_size, 0, img_size],
                        vmin=vmin, vmax=vmax)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(title, fontsize=14)

            # Black border around each image (keep spines, hide ticks)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor('black')
                spine.set_linewidth(1.5)

        plt.suptitle(TITLE if TITLE is not None else 'Inputs', fontsize=16)
        plt.tight_layout()

        if SAVE_DIR is not None:
            plt.savefig(f"{SAVE_DIR}/inputs.png", dpi=150, bbox_inches='tight')

        plt.show()

    def plot_inputs_3d(self, width=1, gap=0.7, angle=30, depth=0.6) -> None:
        """
        Display the 3 U-Net inputs (densities, tx, ty) in an oblique projection:
        the x-axis (columns) is drawn at an iso angle, the y-axis (rows) stays
        vertical. The three images are placed side by side and overlap, so each
        one is partially hidden by its left neighbour.

        Order (left → right, front → back): densities (main, on top), then tx,
        then ty. gray_r [0,1] for densities, symmetric RdBu for tractions.

        Parameters:
            width (int): Traction border thickness in pixels (see plot_inputs).
            gap (float): Horizontal spacing between images, as a fraction of the
                         image size. Smaller = more overlap. Default: 0.7.
            angle (float): Iso angle of the x-axis in degrees. Default: 30.
            depth (float): Foreshortening of the x-axis (oblique depth scale).
                           Default: 0.6.

        Returns:
            None
        """
        from matplotlib.colors import Normalize

        topo     = self.Densities.squeeze().numpy()
        img_size = int(np.sqrt(len(topo)))
        img      = topo.reshape(img_size, img_size)

        tx_ty   = self.get_traction_distribution(width)
        vmax_tx = np.abs(tx_ty[0]).max()
        vmax_ty = np.abs(tx_ty[1]).max()

        layers = [tx_ty[1], tx_ty[0], img]
        cmaps  = ['RdBu', 'RdBu', 'gray_r']
        norms  = [Normalize(-vmax_ty, vmax_ty),
                  Normalize(-vmax_tx, vmax_tx),
                  Normalize(0, 1)]

        # Sheared corner grid: x (columns) goes up-right at `angle`, y vertical
        N      = img_size
        theta  = np.deg2rad(angle)
        xi, yi = np.meshgrid(np.arange(N + 1), np.arange(N + 1))
        sx     = xi * depth * np.cos(theta)
        sy     = yi + xi * depth * np.sin(theta)

        offset = gap * N   # horizontal shift between successive images

        _, ax = plt.subplots(figsize=(12, 5))
        for k in range(len(layers)):
            # ty (k=0) leftmost/behind; densities (k=2) rightmost and on top
            ax.pcolormesh(sx + k * offset, sy, layers[k],
                          cmap=cmaps[k], norm=norms[k],
                          shading='flat', zorder=k)

        ax.set_aspect('equal')
        ax.axis('off')

        plt.tight_layout()
        plt.show()

    def plot_outputs(self, TYPE, SAVE_DIR=None, SEPARATED_IMG=False) -> None:
        """
        Display the 3 stress components: σ_xx, σ_yy, τ_xy.

        Parameters:
            TYPE (str)           : Stress source. 'FEM' for reference stress fields,
                                'UNet' for predicted stress (requires prior prediction).
            SAVE_DIR (str|None)  : Directory to save the figure. Default: None.
            SEPARATED_IMG (bool) : If True, save the 3 images separately without
                                title, colorbar or axis labels. Default: False.

        Returns:
            None
        """
        if TYPE == 'FEM':
            stress = self.FEM_Stress.numpy()
        elif TYPE == 'UNet':
            assert self.UNet_Stress is not None, "UNet stress not computed yet. Run prediction first."
            stress = self.UNet_Stress.numpy()

        img_size = int(np.sqrt(stress.shape[0]))
        sigma_x  = stress[:, 0].reshape(img_size, img_size)
        sigma_y  = stress[:, 1].reshape(img_size, img_size)
        tau_xy   = stress[:, 3].reshape(img_size, img_size)

        vmax = max(np.abs(sigma_x).max(), np.abs(sigma_y).max(), np.abs(tau_xy).max())

        img_data_list = [sigma_x,  sigma_y,  tau_xy ]
        title_list    = ['sigma_x', 'sigma_y', 'tau_xy']

        # ── Mode separated images ─────────────────────────────────────────────────
        if SEPARATED_IMG:
            for img_data, name in zip(img_data_list, title_list):
                fig, ax = plt.subplots(1, 1, figsize=(4, 4))
                ax.imshow(img_data, cmap='RdBu', origin='lower',
                        vmin=-vmax, vmax=vmax)
                ax.axis('off')
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                if SAVE_DIR is not None:
                    plt.savefig(f"{SAVE_DIR}/{TYPE}_{name}.png", dpi=150)
                plt.show()
            return

        # ── Mode combined figure ──────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        for ax, img_data, title in zip(
            axes, img_data_list, ['σ_xx', 'σ_yy', 'τ_xy']
        ):
            im = ax.imshow(img_data, cmap='RdBu', origin='lower',
                        extent=[0, img_size, 0, img_size],
                        vmin=-vmax, vmax=vmax)
            ax.set_title(title, fontsize=14)

            # Black border around each image
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor('black')
                spine.set_linewidth(1.5)

        fig.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04)
        plt.suptitle(f'{TYPE} outputs', fontsize=16)

        if SAVE_DIR is not None:
            plt.savefig(f"{SAVE_DIR}/{TYPE}_outputs.png", dpi=150, bbox_inches='tight')

        plt.show()

        
    def plot_outputs_3d(self, TYPE, gap=0.7, angle=30, depth=0.6, width=2) -> None:
        """
        Display the 3 stress components (σ_xx, σ_yy, τ_xy) in an oblique
        projection: the x-axis (columns) is drawn at an iso angle, the y-axis
        (rows) stays vertical. The three images are placed side by side and
        overlap, so each one is partially hidden by its neighbour. A black
        outline is drawn around each image.

        Order (right → left): σ_xx, σ_yy, τ_xy. σ_xx is rightmost and on top.
        Shared symmetric RdBu scale (white at 0) for the three components.

        Parameters:
            TYPE (str): Stress source. 'FEM' for reference stress fields,
                        'UNet' for predicted stress (requires prior prediction).
            gap (float): Horizontal spacing between images, as a fraction of the
                         image size. Smaller = more overlap. Default: 0.7.
            angle (float): Iso angle of the x-axis in degrees. Default: 30.
            depth (float): Foreshortening of the x-axis (oblique depth scale).
                           Default: 0.6.
            width (float): Line width of the black outline around each image.
                           Default: 2.

        Returns:
            None
        """
        from matplotlib.colors import Normalize
        from matplotlib.patches import Polygon

        if TYPE == 'FEM':
            stress = self.FEM_Stress.numpy()   # (NumEls, 6)
        elif TYPE == 'UNet':
            assert self.UNet_Stress is not None, "UNet stress not computed yet. Run prediction first."
            stress = self.UNet_Stress.numpy()  # (NumEls, 6)

        img_size = int(np.sqrt(stress.shape[0]))
        sigma_x  = stress[:, 0].reshape(img_size, img_size)
        sigma_y  = stress[:, 1].reshape(img_size, img_size)
        tau_xy   = stress[:, 3].reshape(img_size, img_size)

        # Shared symmetric scale around 0 → white at 0 for all 3 components
        vmax = max(np.abs(sigma_x).max(), np.abs(sigma_y).max(), np.abs(tau_xy).max())
        norm = Normalize(-vmax, vmax)

        # Left → right: τ_xy, σ_yy, σ_xx  (i.e. right → left: σ_xx, σ_yy, τ_xy)
        layers = [tau_xy, sigma_y, sigma_x]

        # Sheared corner grid: x (columns) goes up-right at `angle`, y vertical
        N      = img_size
        theta  = np.deg2rad(angle)
        xi, yi = np.meshgrid(np.arange(N + 1), np.arange(N + 1))
        sx     = xi * depth * np.cos(theta)
        sy     = yi + xi * depth * np.sin(theta)

        offset = gap * N   # horizontal shift between successive images

        _, ax = plt.subplots(figsize=(12, 5))
        im = None
        for k in range(len(layers)):
            x0 = k * offset
            # τ_xy (k=0) leftmost/behind; σ_xx (k=2) rightmost and on top
            im = ax.pcolormesh(sx + x0, sy, layers[k],
                               cmap='RdBu', norm=norm,
                               shading='flat', zorder=k)

            # Black outline around the image (parallelogram corners)
            corners = np.array([
                [x0,                          0           ],  # bottom-left
                [x0 + N * depth * np.cos(theta), N * depth * np.sin(theta)],  # bottom-right
                [x0 + N * depth * np.cos(theta), N + N * depth * np.sin(theta)],  # top-right
                [x0,                          N           ],  # top-left
            ])
            ax.add_patch(Polygon(corners, closed=True, fill=False,
                                 edgecolor='black', lw=width, zorder=k + 0.5))

        ax.set_aspect('equal')
        ax.axis('off')

        fig = ax.figure
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()

    def copy(self):
        """Creates an independant copy of a sample """
        new = IterationSample.__new__(IterationSample)
        new.Tractions         = self.Tractions.clone()
        new.Densities         = self.Densities.clone()
        new.Relative_Vol_Frac = self.Relative_Vol_Frac.clone()
        new.FEM_Stress        = self.FEM_Stress.clone()
        new.Ener              = self.Ener.clone()
        new.FEMc              = self.FEMc.clone()
        new.c                 = self.c.clone()
        new.NumIts            = self.NumIts.clone()
        new.ItsFull            = self.ItsFull.clone()
        new.TEnd              = self.TEnd.clone()
        new.UNet_Stress       = self.UNet_Stress.clone() if self.UNet_Stress is not None else None
        return new


#%% Dataset for accelerated process
class AcceleratedDataset(Dataset):
    """
    Lightweight dataset keeping only the converged (last-iteration) density of
    each case plus its tractions and first-image energy.

    It is used to accelerate a new optimization: given an unoptimized sample,
    `closest_point` finds the case whose energy field is most similar and reuses
    its converged geometry as a warm start. `augment` builds the full D4
    (8-fold) symmetric expansion of the dataset.
    """

    def __init__(self, dataset):
        """
        Parameters
        ----------
        dataset : Dataset_TopOpt — source dataset; only the last density of each
            case is kept, together with all tractions and energies.
        """
        self.Ener = dataset.Ener
        self.Tractions = dataset.Tractions
        # Keep only the converged (last) density of each case → (N, NumEls)
        self.Densities = np.array([dataset.Densities[i][:, -1] for i in range(len(dataset))])  # (N, 1024)
        self.size = len(dataset)

    def __len__(self):
        """
        Return the number of traction distributions in the dataset.

        Returns:
            int: Number of cases N.
        """
        return self.size

    def __repr__(self):
        """Return a short summary of the dataset size and array shapes."""
        ener_shape = self.Ener[0].shape if len(self) else None
        return (f"AcceleratedDataset\n"
                f"  Num distributions : {len(self)}\n"
                f"  Tractions         : {tuple(np.shape(self.Tractions))}\n"
                f"  Densities         : {tuple(np.shape(self.Densities))}\n"
                f"  Ener (per case)   : {tuple(ener_shape) if ener_shape is not None else 'empty'}")

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            idx = idx[0]
        ener = torch.from_numpy(self.Ener[idx]).float() if self.Ener is not None else None
        return {
            'Tractions'         : torch.from_numpy(self.Tractions[:, :, idx]).float().unsqueeze(0),
            'MeshData'          : None,
            'Densities'         : torch.from_numpy(self.Densities[idx]).float().unsqueeze(0),
            'Relative_Vol_Frac' : torch.tensor(float(self.Densities[idx].mean())).float(),
            'Stress'            : ener,
            'Ener'              : ener,
            'FEMc'              : torch.tensor(0.0).float(),
            'c'                 : torch.tensor(0.0).float(),
            'NumIts'            : torch.tensor(1.0).float(),
            'ItsFull'           : torch.tensor(1.0).float(),
            'TEnd'              : torch.tensor(0.0).float(),
        }

    def closest_point(self, sample:IterationSample):
        '''
        Find the dataset case whose first-image energy field is closest (in
        squared-error sense) to that of `sample`, and return a copy of `sample`
        whose density is replaced by that case's converged geometry (warm start).

        Parameters
        ----------
        sample : IterationSample — query sample (uses its Ener field).

        Returns
        -------
        tuple(int, IterationSample)
            index_min_ener : index of the closest dataset case.
            new_sample     : copy of `sample` with the matched converged density.
        '''

        Ener_dataset = self.Ener[0] # 1024 x 6
        Ener_sample  = sample.Ener
  
        min_distance_ener = ((Ener_dataset - Ener_sample.numpy()) ** 2).sum()
        index_min_ener    = 0

        for i in range(1,len(self)):
            Ener_dataset = self.Ener[i] # 1024 x 6

            # computes distance between energies           
            distance_ener = ((Ener_dataset - Ener_sample.numpy()) ** 2).sum()

            if distance_ener<min_distance_ener:
                min_distance_ener = distance_ener
                index_min_ener    = i


        # creates a new sample with closest optimised geometry
        new_sample = sample.copy()

        # Density actualisation
        new_sample.Densities[0] = torch.from_numpy(
            self.Densities[index_min_ener]
        ).float()

        return index_min_ener, min_distance_ener, new_sample

    def _sample_from_index(self, i):
        """
        Build a minimal IterationSample for case i so the sample-level D4
        transforms (symmetry_x, symmetry_y, rotation_90) can be applied.

        Only the fields those transforms actually read are filled with real
        data (Tractions, Densities, Ener). The remaining fields get harmless
        dummy tensors so that IterationSample.copy() works and FEM_Stress can
        be reshaped without crashing (its transformed value is discarded).

        Parameters:
            i (int): Case index.

        Returns:
            IterationSample: sample carrying the case-i tractions, density and energy.
        """
        s = IterationSample.__new__(IterationSample)
        s.Tractions         = torch.from_numpy(self.Tractions[:, :, i]).float().unsqueeze(0)  # (1, 2, 8)
        s.Densities         = torch.from_numpy(self.Densities[i]).float().reshape(1, -1)       # (1, NumEls)
        s.Ener              = torch.from_numpy(self.Ener[i]).float()                            # (NumEls, 6)
        s.FEM_Stress        = torch.zeros_like(s.Ener)   # dummy — transformed but unused
        s.Relative_Vol_Frac = torch.tensor(0.0)
        s.FEMc              = torch.tensor(0.0)
        s.c                 = torch.tensor(0.0)
        s.NumIts            = torch.tensor(0.0)
        s.ItsFull           = torch.tensor(0.0)
        s.TEnd              = torch.tensor(0.0)
        s.UNet_Stress       = None
        return s

    def augment(self):
        """
        Build the full dihedral-group (D4) augmentation of the dataset.

        Every case is replicated under all 8 square symmetries — the 4 rotations
        (0/90/180/270 deg) each optionally preceded by a horizontal mirror.
        Together {r^k} and {s_x . r^k} for k = 0..3 enumerate the 8 elements of D4.

        The sample-level transforms rotate/mirror Tractions, Densities and Ener
        CONSISTENTLY (a fixed load-independent map, FEM-verified on bending/shear
        across the full D4), so each augmented case is a true physical image.

        Returns:
            AcceleratedDataset: new dataset of size 8*N, with two extra
                provenance attributes:
                  source_index (ndarray, 8N) : original case index of each entry.
                  transforms   (list of (flip, k)) : the (mirror, n_rot90) applied.
        """
        Ener_aug, Tract_aug, Dens_aug = [], [], []
        source_index, transforms = [], []

        for i in range(len(self)):
            base = self._sample_from_index(i)
            for flip in (False, True):
                for k in (0, 1, 2, 3):
                    s = symmetry_x(base) if flip else base
                    s = rotation_90(s, N_rot=k)   # k=0 identity .. k=3 rot270

                    Ener_aug.append(s.Ener.numpy())                  # (NumEls, 6)
                    Tract_aug.append(s.Tractions.squeeze(0).numpy()) # (2, 8)
                    Dens_aug.append(s.Densities.squeeze().numpy())   # (NumEls,)
                    source_index.append(i)
                    transforms.append((flip, k))

        # Ener kept as a (8N,) object array of (NumEls, 6) blocks, matching
        # the indexing used by closest_point (self.Ener[i] -> 2D array).
        Ener_obj = np.empty(len(Ener_aug), dtype=object)
        for idx, e in enumerate(Ener_aug):
            Ener_obj[idx] = e

        aug = AcceleratedDataset.__new__(AcceleratedDataset)
        aug.Ener         = Ener_obj
        aug.Tractions    = np.stack(Tract_aug, axis=-1)   # (2, 8, 8N)
        aug.Densities    = np.array(Dens_aug)             # (8N, NumEls)
        aug.size         = len(Ener_aug)
        aug.source_index = np.array(source_index)
        aug.transforms   = transforms
        return aug



#%% Data loader

def get_dataloader(dataset: IterationDataset, batch_size: int = 32,
                   val_split: float = 0.15, shuffle: bool = True,
                   num_workers: int = 0) -> tuple:
    """
    Create train and validation DataLoaders from an IterationDataset.

    Parameters:
        dataset (IterationDataset) : source dataset.
        batch_size (int)           : mini-batch size. Default: 32.
        val_split (float)          : fraction of the dataset used for validation. Default: 0.15.
        shuffle (bool)             : shuffle training data. Default: True.
        num_workers (int)          : number of worker processes. Default: 0.

    Returns:
        tuple: (train_loader, val_loader)
    """
    n_val   = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val

    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle,   num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,     num_workers=num_workers)

    return train_loader, val_loader


def get_traction_distribution(Tractions, img_size=32):
    """
    Build 2D traction images tx and ty (global components) by linear
    interpolation of the boundary nodes along the 4 edges of the square domain.

    The input rows are per-edge (Tn, Tt) = (normal, tangential); they are
    converted to global (tx, ty) and each node placed at its physical position
    (see `tractions_to_global` / `node_positions_pixels`). Kept consistent with
    the IterationSample method and `train._tractions_to_maps`.

    Parameters:
        Tractions (ndarray) : nodal (Tn, Tt), shape (2, 8).
        img_size (int)      : square image size. Default: 32.

    Returns:
        tuple: (tx, ty) two ndarrays of shape (img_size, img_size), global.
    """
    tx = np.zeros((img_size, img_size))
    ty = np.zeros((img_size, img_size))

    T_global = tractions_to_global(np.asarray(Tractions))          # (2, 8)
    Points   = node_positions_pixels(img_size, inclusive=False)    # (8, 2)

    for k in range(0, 8, 2):
        p1 = Points[k]
        p2 = Points[k+1]
        xs = np.round(np.linspace(p1[0], p2[0], img_size)).astype(int)
        ys = np.round(np.linspace(p1[1], p2[1], img_size)).astype(int)
        tx[ys, xs] += np.linspace(T_global[0, k], T_global[0, k+1], img_size)
        ty[ys, xs] += np.linspace(T_global[1, k], T_global[1, k+1], img_size)

    return tx, ty



#%% List to IterationSample
def list_to_IterationDataset(list_samples: list[IterationSample]) -> IterationDataset:
    """
    Convert a list of IterationSample into an IterationDataset.
    Samples are assumed to be successive iterations of one topology optimization.

    Parameters:
        list_samples (list[IterationSample]): List of successive iteration samples.

    Returns:
        IterationDataset: Dataset containing all samples as iteration (0, j).
    """
    n_iter = len(list_samples)

    # Build the (NumEls, 6, n_iter) array with an explicit numeric dtype
    stress_stack = np.stack(
        [s.FEM_Stress.numpy().astype(np.float32) for s in list_samples], axis=-1
    )  # (NumEls, 6, n_iter), dtype=float32

    sub = {
        'MeshData'          : None,
        'Tractions'         : list_samples[0].Tractions.squeeze().numpy()[:, :, np.newaxis],  # (2, 8, 1)
        'Relative_Vol_Frac' : np.array([list_samples[0].Relative_Vol_Frac.item()]),
        'Rel_Density'       : list_samples[-1].Densities.squeeze().numpy()[:, np.newaxis],    # (NumEls, 1)
        'Stress'            : np.empty(1, dtype=object),  # object array of length 1, filled below
        'Ener'              : np.empty(1, dtype=object),  # object array of length 1, filled below
        'Densities'         : np.stack(
                                [s.Densities.squeeze().numpy() for s in list_samples], axis=-1 # (NumEls, n_iter)
                            ),
        'c'                 : np.array([[s.c.item() for s in list_samples]]),                  # (1, n_iter)
        'NumIts'            : np.array([n_iter]),
        'ItsFull'           : np.array([list_samples[0].ItsFull.item()]),
        'FEMc'              : np.array([[s.FEMc.item() for s in list_samples]]),               # (1, n_iter)
        'TEnd'              : list_samples[0].TEnd.item(),
    }
    sub['Stress'][0] = stress_stack  # explicitly assign the float32 array into the object slot
    sub['Ener'][0]   = list_samples[0].Ener.numpy().astype(np.float32)  # same trick for Ener

    ds_base = Dataset_TopOpt(sub)
    return IterationDataset(ds_base)

#%% Data augmentation 



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
    s.Ener              = sample_dict['Ener']
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
        'Ener'              : sample.Ener,
        'FEMc'              : sample.FEMc,
        'c'                 : sample.c,
        'NumIts'            : sample.NumIts,
        'ItsFull'           : sample.ItsFull,
        'TEnd'              : sample.TEnd,
    }


def random_augment(sample, rotation_90, symmetry_x, symmetry_y):
    """
    Apply a random element of the square's symmetry group (dihedral D4) to a
    sample: an optional horizontal mirror followed by a random 0/90/180/270 deg
    rotation. The 8 elements {r^k} and {s_x . r^k}, k=0..3, enumerate all of D4.

    All transforms rotate/mirror density, tractions AND stress CONSISTENTLY (the
    transform is a fixed load-independent linear map, FEM-verified on bending and
    shear across the full D4). The augmented (rho, T, stress) is the true physical
    image of the sample under the symmetry — a valid training example that teaches
    the network the rotation/reflection equivariance of elasticity.

    The transform functions are passed in as arguments rather than imported
    directly, so this module stays decoupled from wherever they live.
    """
    if np.random.rand() < 0.5:
        sample = symmetry_x(sample)
    sample = rotation_90(sample, N_rot=np.random.randint(0, 4))
    return sample

class AugmentedIterationDataset(torch.utils.data.Dataset):
    """
    Wrapper that applies random augmentation to a base dataset.

    With probability `p` (and only when `enabled`), each fetched sample is
    transformed by a random rotation/symmetry before being returned, so the
    network sees fresh orientations every epoch without growing the stored data.
    """
    def __init__(self, base_dataset, p, rotation_90, symmetry_x, symmetry_y, enabled=True):
        """
        Parameters
        ----------
        base_dataset : Dataset — underlying dataset returning sample dicts.
        p            : float — probability of augmenting a given sample.
        rotation_90, symmetry_x, symmetry_y : callables — the transform
            functions, injected to avoid a hard import dependency.
        enabled      : bool — master switch (disable for validation/eval).
        """
        self.base        = base_dataset
        self.p           = p
        self.enabled      = enabled
        self.rotation_90 = rotation_90
        self.symmetry_x  = symmetry_x
        self.symmetry_y  = symmetry_y

    def __len__(self):
        """Return the number of samples (same as the base dataset)."""
        return len(self.base)

    def __getitem__(self, idx):
        """Fetch sample `idx`, randomly augmenting it with probability `p`."""
        sample_dict = self.base[idx]

        if self.enabled and np.random.rand() < self.p:
            sample = dict_to_sample(sample_dict)
            sample = random_augment(sample, self.rotation_90,
                                    self.symmetry_x, self.symmetry_y)
            sample_dict = sample_to_dict(sample)

        return sample_dict


def permutation_tractions(sample, i , j):
    """
    Swap traction nodes i and j in place (both tx and ty components).

    Helper used by the symmetry transforms to remap the 8 boundary nodes.

    Parameters
    ----------
    sample : IterationSample — modified in place.
    i, j   : int — node indices to swap.
    """
    t=sample.Tractions[0][:,i].clone()

    sample.Tractions[0][:,i] = sample.Tractions[0][:,j]
    sample.Tractions[0][:,j] = t


# ─── Traction transforms in the global frame ─────────────────────────────────
#
# The 8x2 Tractions store, per boundary node, (Tn, Tt) = (normal, tangential)
# components in a per-EDGE local frame — NOT global (tx, ty). This is dictated by
# the MATLAB solver (OT_Functions/VectorF_Line.m, OT_Software/GenerateTractions.m).
# So a D4 symmetry cannot be applied by naive xy sign flips: we convert each node
# to its global vector, apply the true geometric map (relocate nodes + rotate the
# vector), then convert back to (Tn, Tt) in the node's NEW edge frame.
#
# Node index -> edge (0-based): 0,1 bottom | 2,3 right | 4,5 top | 6,7 left.
# Positions in centred coords, outward normal and tangent per node.
#
# FRAME NOTE: these are first given in the solver's MESH frame (x right, y up,
# from SolveFE.m). But the density/stress IMAGE handled in Python is the
# TRANSPOSE of that frame — Python reshapes the element vector row-major whereas
# MATLAB/the mesh order is column-major. So a torch.rot90/flip on the density
# image is a transposed operation relative to the mesh. To keep the traction
# transform (and the tx/ty maps) consistent with the density/stress images, we
# express the node geometry in that same IMAGE frame by swapping x<->y.
_NODE_POS_MESH = np.array([[-1, -1], [ 1, -1], [ 1,  1], [ 1, -1],
                           [ 1,  1], [-1,  1], [-1, -1], [-1,  1]], dtype=float)
_NODE_NRM_MESH = np.array([[ 0, -1], [ 0, -1], [ 1,  0], [ 1,  0],
                           [ 0,  1], [ 0,  1], [-1,  0], [-1,  0]], dtype=float)
_NODE_TAN_MESH = np.array([[ 1,  0], [ 1,  0], [ 0,  1], [ 0,  1],
                           [-1,  0], [-1,  0], [ 0, -1], [ 0, -1]], dtype=float)

# mesh frame -> image frame (swap x<->y, i.e. transpose the plane)
_NODE_POS = _NODE_POS_MESH[:, ::-1].copy()
_NODE_NRM = _NODE_NRM_MESH[:, ::-1].copy()
_NODE_TAN = _NODE_TAN_MESH[:, ::-1].copy()

_R_FLIP_X = np.array([[-1., 0.], [0., 1.]])   # horizontal mirror (x -> -x)
_R_FLIP_Y = np.array([[1., 0.], [0., -1.]])   # vertical   mirror (y -> -y)


def _R_rot(n_rot):
    """2x2 physical rotation matching the density's torch.rot90(k_image=-n_rot):
    a CCW rotation by 90*n_rot degrees."""
    a = np.deg2rad(90 * n_rot)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def _node_permutation(R):
    """Bijection old node index -> new node index induced by the map R on the
    square (matches both the relocated corner AND the new outward normal, which
    disambiguates the two nodes sharing each corner)."""
    pi = np.empty(8, dtype=int)
    for k in range(8):
        pos, nrm = R @ _NODE_POS[k], R @ _NODE_NRM[k]
        for m in range(8):
            if np.allclose(_NODE_POS[m], pos) and np.allclose(_NODE_NRM[m], nrm):
                pi[k] = m
                break
        else:
            raise ValueError(f"no matching node for index {k} under R={R.tolist()}")
    return pi


def transform_tractions(T_tensor, R):
    """
    Transform the (2, 8) traction array under a D4 element given by the 2x2
    orthogonal matrix R, working in the global frame.

    Parameters
    ----------
    T_tensor : torch.Tensor — shape (2, 8), rows are (Tn, Tt) per node.
    R        : np.ndarray   — 2x2 rotation/reflection matrix.

    Returns
    -------
    torch.Tensor — transformed (2, 8) traction array, same dtype as input.
    """
    T  = T_tensor.detach().cpu().numpy().astype(float)          # (2, 8): [Tn; Tt]
    G  = _NODE_NRM.T * T[0] + _NODE_TAN.T * T[1]                 # (2, 8) global [tx; ty]
    pi = _node_permutation(R)

    G_new = np.zeros_like(G)
    for k in range(8):
        G_new[:, pi[k]] = R @ G[:, k]                           # relocate + rotate vector

    Tn = np.sum(G_new.T * _NODE_NRM, axis=1)                    # back to local (Tn, Tt)
    Tt = np.sum(G_new.T * _NODE_TAN, axis=1)
    return torch.from_numpy(np.stack([Tn, Tt], axis=0)).to(T_tensor.dtype)


def tractions_to_global(T):
    """
    Convert a (2, 8) traction array from local per-edge (Tn, Tt) = (normal,
    tangential) components to global (tx, ty). Needed for any *visualisation*,
    since the stored rows are NOT global x/y (see transform_tractions).

    Parameters
    ----------
    T : array-like — shape (2, 8), rows (Tn, Tt) per node.

    Returns
    -------
    np.ndarray — shape (2, 8), rows (tx, ty) global.
    """
    T = np.asarray(T, dtype=float)
    return _NODE_NRM.T * T[0] + _NODE_TAN.T * T[1]              # (2, 8) [tx; ty]


def node_positions_pixels(img_size, inclusive=True):
    """
    Physical boundary-node positions (x, y) in pixel coordinates, shape (8, 2),
    following the solver convention (node 0,1 bottom | 2,3 right | 4,5 top |
    6,7 left). `inclusive=True` spans [0, img_size] (for quiver overlays on an
    imshow extent); `False` spans [0, img_size-1] (raw pixel indices).
    """
    span = img_size if inclusive else img_size - 1
    return (_NODE_POS + 1) / 2 * span                          # (8, 2), (x, y)


def symmetry_x(sample):
    """
    Apply a horizontal mirror (reflection about the vertical axis) to a sample,
    transforming every field consistently: density, the 8 traction nodes
    (permutation + tx sign flip), the stress fields (mirror + τxy sign flip),
    the energy (mirror, no sign change since energy is quadratic) and, if
    present, the U-Net stress.

    Parameters
    ----------
    sample : IterationSample — source sample (left unchanged; a copy is returned).

    Returns
    -------
    IterationSample — mirrored copy.
    """
    new_sample=sample.copy()

    img_size = int(np.sqrt(sample.Densities.shape[1]))

    # Density symmetry
    rho_2d = new_sample.Densities.squeeze().reshape(img_size, img_size)
    rho_flipped = torch.flip(rho_2d, dims=(1,))
    new_sample.Densities = rho_flipped.reshape(1, -1)


    # Traction nodes - full global-frame transform (relocation + tx sign flip)
    new_sample.Tractions[0] = transform_tractions(sample.Tractions[0], _R_FLIP_X)


    # Stress - image symmetry
    sx_2d  = new_sample.FEM_Stress[:, 0].reshape(img_size, img_size)
    sy_2d  = new_sample.FEM_Stress[:, 1].reshape(img_size, img_size)
    txy_2d = new_sample.FEM_Stress[:, 3].reshape(img_size, img_size)

    sx_flipped  = torch.flip(sx_2d,  dims=(1,))
    sy_flipped  = torch.flip(sy_2d,  dims=(1,))
    txy_flipped = torch.flip(txy_2d, dims=(1,))

    new_sample.FEM_Stress[:, 0] = sx_flipped.reshape(-1)
    new_sample.FEM_Stress[:, 1] = sy_flipped.reshape(-1)
    new_sample.FEM_Stress[:, 3] = txy_flipped.reshape(-1)


    # Stress - sign change: tau' = - tau'
    new_sample.FEM_Stress[:, 3] = - new_sample.FEM_Stress[:, 3]


    # Energy - image symmetry (no sign change: energy is quadratic in the fields,
    # so the shear product E_xy = sigma_xy * eps_xy is invariant under reflection)
    if getattr(new_sample, 'Ener', None) is not None:
        for comp in (0, 1, 3):
            e_2d = new_sample.Ener[:, comp].reshape(img_size, img_size)
            new_sample.Ener[:, comp] = torch.flip(e_2d, dims=(1,)).reshape(-1)


    # UNet Stress - image symmetry
    if new_sample.UNet_Stress is not None:
        sx_2d  = new_sample.UNet_Stress[:, 0].reshape(img_size, img_size)
        sy_2d  = new_sample.UNet_Stress[:, 1].reshape(img_size, img_size)
        txy_2d = new_sample.UNet_Stress[:, 3].reshape(img_size, img_size)

        sx_flipped  = torch.flip(sx_2d,  dims=(1,))
        sy_flipped  = torch.flip(sy_2d,  dims=(1,))
        txy_flipped = torch.flip(txy_2d, dims=(1,))

        new_sample.UNet_Stress[:, 0] = sx_flipped.reshape(-1)
        new_sample.UNet_Stress[:, 1] = sy_flipped.reshape(-1)
        new_sample.UNet_Stress[:, 3] = txy_flipped.reshape(-1)


        # UNet Stress - sign change: tau' = - tau'
        new_sample.UNet_Stress[:, 3] = - new_sample.UNet_Stress[:, 3]


    return new_sample


def symmetry_y(sample):
    """
    Apply a vertical mirror (reflection about the horizontal axis) to a sample,
    transforming every field consistently: density, the 8 traction nodes
    (permutation + ty sign flip), the stress fields (mirror + τxy sign flip),
    the energy (mirror, no sign change) and, if present, the U-Net stress.

    Parameters
    ----------
    sample : IterationSample — source sample (left unchanged; a copy is returned).

    Returns
    -------
    IterationSample — mirrored copy.
    """
    new_sample=sample.copy()

    img_size = int(np.sqrt(sample.Densities.shape[1]))

    # Density symmetry
    rho_2d = new_sample.Densities.squeeze().reshape(img_size, img_size)
    rho_flipped = torch.flip(rho_2d, dims=(0,))
    new_sample.Densities = rho_flipped.reshape(1, -1)


    # Traction nodes - full global-frame transform (relocation + ty sign flip)
    new_sample.Tractions[0] = transform_tractions(sample.Tractions[0], _R_FLIP_Y)


    # Stress - image symmetry
    sx_2d  = new_sample.FEM_Stress[:, 0].reshape(img_size, img_size)
    sy_2d  = new_sample.FEM_Stress[:, 1].reshape(img_size, img_size)
    txy_2d = new_sample.FEM_Stress[:, 3].reshape(img_size, img_size)

    sx_flipped  = torch.flip(sx_2d,  dims=(0,))
    sy_flipped  = torch.flip(sy_2d,  dims=(0,))
    txy_flipped = torch.flip(txy_2d, dims=(0,))

    new_sample.FEM_Stress[:, 0] = sx_flipped.reshape(-1)
    new_sample.FEM_Stress[:, 1] = sy_flipped.reshape(-1)
    new_sample.FEM_Stress[:, 3] = txy_flipped.reshape(-1)


    # Stress - sign change: tau' = - tau'
    new_sample.FEM_Stress[:, 3] = - new_sample.FEM_Stress[:, 3]


    # Energy - image symmetry (no sign change: energy is quadratic in the fields,
    # so the shear product E_xy = sigma_xy * eps_xy is invariant under reflection)
    if getattr(new_sample, 'Ener', None) is not None:
        for comp in (0, 1, 3):
            e_2d = new_sample.Ener[:, comp].reshape(img_size, img_size)
            new_sample.Ener[:, comp] = torch.flip(e_2d, dims=(0,)).reshape(-1)


    # UNet Stress - image symmetry
    if new_sample.UNet_Stress is not None:
        sx_2d  = new_sample.UNet_Stress[:, 0].reshape(img_size, img_size)
        sy_2d  = new_sample.UNet_Stress[:, 1].reshape(img_size, img_size)
        txy_2d = new_sample.UNet_Stress[:, 3].reshape(img_size, img_size)

        sx_flipped  = torch.flip(sx_2d,  dims=(0,))
        sy_flipped  = torch.flip(sy_2d,  dims=(0,))
        txy_flipped = torch.flip(txy_2d, dims=(0,))

        new_sample.UNet_Stress[:, 0] = sx_flipped.reshape(-1)
        new_sample.UNet_Stress[:, 1] = sy_flipped.reshape(-1)
        new_sample.UNet_Stress[:, 3] = txy_flipped.reshape(-1)


        # UNet Stress - sign change: tau' = - tau'
        new_sample.UNet_Stress[:, 3] = - new_sample.UNet_Stress[:, 3]


    return new_sample


def rotation_90(sample, N_rot=1):
    """
    Rotate a sample by N_rot * 90 degrees, transforming every field
    consistently. The image fields (density, stress, energy) rotate in the
    opposite direction to the traction nodes; on odd rotations the σx/σy (and
    energy xx/yy) channels swap and τxy changes sign, while the traction nodes
    are permuted and their vectors rotated.

    The density/stress image rotation and the traction node rotation use opposite
    matrix signs because the element image is the transpose of the mesh frame (see
    _NODE_POS); this makes all fields represent the SAME physical rotation. Verified
    FEM-consistent for all N_rot on bending and shear (load-independent transform).

    Parameters
    ----------
    sample : IterationSample — source sample (left unchanged; a copy is returned).
    N_rot  : int — number of 90-degree steps (taken modulo 4).

    Returns
    -------
    IterationSample — rotated copy.
    """
    new_sample = sample.copy()

    img_size = int(np.sqrt(sample.Densities.shape[1]))

    k       = N_rot % 4
    k_image = (-k) % 4   # the image rotates in the opposite direction to the tractions

    # ── Density - image rotation ──────────────────────────────────────────
    rho_2d = new_sample.Densities.squeeze().reshape(img_size, img_size)
    rho_rot = torch.rot90(rho_2d, k_image, dims=(0, 1))
    new_sample.Densities = rho_rot.reshape(1, -1)

    # ── Traction nodes - full global-frame transform (relocation + vector rot) ──
    # _R_rot(-k): tractions rotate with the OPPOSITE matrix sign to the density's
    # torch.rot90 because the element image is the transpose of the mesh frame, so
    # both represent the same physical rotation (FEM-verified on bending/shear).
    new_sample.Tractions[0] = transform_tractions(sample.Tractions[0], _R_rot(-k))

    # ── Stress - image rotation (same direction as the density) ────────────
    sx_2d  = new_sample.FEM_Stress[:, 0].reshape(img_size, img_size)
    sy_2d  = new_sample.FEM_Stress[:, 1].reshape(img_size, img_size)
    txy_2d = new_sample.FEM_Stress[:, 3].reshape(img_size, img_size)

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

    # ── Energy - image rotation (xx<->yy swap on odd k, NO shear sign change) ──
    if getattr(new_sample, 'Ener', None) is not None:
        ex_2d  = new_sample.Ener[:, 0].reshape(img_size, img_size)
        ey_2d  = new_sample.Ener[:, 1].reshape(img_size, img_size)
        exy_2d = new_sample.Ener[:, 3].reshape(img_size, img_size)

        ex_rot  = torch.rot90(ex_2d,  k_image, dims=(0, 1))
        ey_rot  = torch.rot90(ey_2d,  k_image, dims=(0, 1))
        exy_rot = torch.rot90(exy_2d, k_image, dims=(0, 1))

        if k % 2 == 1:
            new_sample.Ener[:, 0] = ey_rot.reshape(-1)
            new_sample.Ener[:, 1] = ex_rot.reshape(-1)
            new_sample.Ener[:, 3] = exy_rot.reshape(-1)
        else:
            new_sample.Ener[:, 0] = ex_rot.reshape(-1)
            new_sample.Ener[:, 1] = ey_rot.reshape(-1)
            new_sample.Ener[:, 3] = exy_rot.reshape(-1)

    if new_sample.UNet_Stress is not None:
        sx_2d  = new_sample.UNet_Stress[:, 0].reshape(img_size, img_size)
        sy_2d  = new_sample.UNet_Stress[:, 1].reshape(img_size, img_size)
        txy_2d = new_sample.UNet_Stress[:, 3].reshape(img_size, img_size)

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
    BASE = Path(__file__).parents[3]

    # Reference Dataset
    path = (BASE / 'HeavyFiles/data/dataset_128.mat').resolve()
    data = load_mat(path)
    ds_base = Dataset_TopOpt(data)
    data_iter = IterationDataset(ds_base)
    sample = IterationSample(data_iter, 40)
    
    sample.plot(scale_force=100)
    sample.plot_inputs(width=5)
    sample.plot_outputs('FEM')

    


    # ID = 0
    # sample = IterationSample(IterationDataset(dataset.get_series(ID)), -1)
    # sample.plot_inputs(width=5)
    # sample.plot_outputs('FEM')
    # sample.plot_inputs_3d(width=5, gap=0.2, angle=45)
    # sample.plot_outputs_3d('FEM', width=1, gap=0.2, angle=45)
 

    # ── Extraire toutes les tractions du dataset ──────────────────────────────────
    # ds_base.Tractions shape : dépend du dataset, typiquement (N_distrib, 2, 8)
    # On récupère les angles de chaque composante nodale (tx, ty) pour chaque noeud

    # T = data_acc_aug.Tractions   # (N_distrib, 2, 8) ou (2, 8, N_distrib) selon le mat

    # # Vérifier la shape
    # print("Tractions shape :", T.shape)

    # # Adapter selon la shape réelle
    # # Si T.shape = (2, 8, N) :
    # tx = T[0, :, :]   
    # ty = T[1, :, :]   

    # # ── Angles des vecteurs de traction ──────────────────────────────────────────
    # # arctan2(ty, tx) donne l'angle de chaque vecteur nodal en radians
    # angles = np.arctan2(ty, tx)   # (N, 8) — un angle par noeud par distribution

    # # Amplitude de chaque vecteur
    # amplitudes = np.sqrt(tx**2 + ty**2)   # (N, 8)

    # # ── Figure 1 : distribution des angles ───────────────────────────────────────
    # fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # axes[0].hist(angles.flatten(), bins=72, color='steelblue', edgecolor='none')
    # axes[0].set_xlabel("Angle (rad)", fontsize=13)
    # axes[0].set_ylabel("Count", fontsize=13)
    # axes[0].set_title("Distribution of traction angles\n(all nodes, all distributions)", fontsize=13)
    # axes[0].axhline(len(angles.flatten()) / 72, color='coral', linestyle='--',
    #                 label='Uniform reference')
    # axes[0].legend()
    # axes[0].set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    # axes[0].set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])

    # # ── Figure 2 : distribution des amplitudes ───────────────────────────────────
    # axes[1].hist(amplitudes.flatten(), bins=50, color='teal', edgecolor='none')
    # axes[1].set_xlabel("Amplitude", fontsize=13)
    # axes[1].set_ylabel("Count", fontsize=13)
    # axes[1].set_title("Distribution of traction amplitudes\n(all nodes, all distributions)", fontsize=13)

    # plt.suptitle(f"Traction diversity — {len(ds_base)} distributions, {T.shape[1]} nodes each",
    #             fontsize=14)
    # plt.tight_layout()
    # plt.show()

    # # ── Figure 3 : rose des vents (polar histogram) ───────────────────────────────
    # fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 6))
    # counts, bin_edges = np.histogram(angles.flatten(), bins=36)
    # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # bar_width   = 2 * np.pi / 36

    # bars = ax.bar(bin_centers, counts, width=bar_width,
    #             bottom=0, color='steelblue', alpha=0.8, edgecolor='none')
    # ax.set_title("Polar distribution of traction angles", fontsize=13, pad=20)
    # plt.tight_layout()
    # plt.show()

    # # ── Statistique résumée ───────────────────────────────────────────────────────
    # print(f"\nNombre de distributions : {len(ds_base)}")
    # print(f"Nombre total de vecteurs : {angles.size}")
    # print(f"Angle moyen   : {np.degrees(angles.mean()):.1f} deg")
    # print(f"Ecart-type    : {np.degrees(angles.std()):.1f} deg  (uniforme attendu : {np.degrees(np.pi/np.sqrt(3)):.1f} deg)")
    # print(f"Amplitude min : {amplitudes.min():.4f}")
    # print(f"Amplitude max : {amplitudes.max():.4f}")
    # print(f"Amplitude moy : {amplitudes.mean():.4f}")


#  plot_inputs_3d(self, TITLE=None, width=1, gap=0.7,
                    #    angle=30, depth=0.6)

    # acc_data = AcceleratedDataset(dataset)

    # # Test dataset
    # path_test = (BASE / 'HeavyFiles/data/dataset_macro_cantilever.mat').resolve()
    # data_test = load_mat(path_test)
    # dataset_test = Dataset_TopOpt(data_test)
    # data_iter_test = IterationDataset(dataset_test)


    # ID = 20
    # # unoptimised sample
    # sample = IterationSample(IterationDataset(dataset_test.get_series(ID)), 0)
    # idx_old, acc_starting_point = acc_data.closest_point(sample)

    # # compare with old sample
    # old_sample = IterationSample(IterationDataset(dataset.get_series(idx_old)), -1)

    # # plot empty -> old -> new samples
    # print(f'Empty sample \n {sample.Tractions}')
    # sample.plot()
    # print(f'Old sample (wrong tractions) \n {old_sample.Tractions}')
    # old_sample.plot()
    # print(f'New sample (actualised tractions) \n {acc_starting_point.Tractions}')
    # acc_starting_point.plot()

    # sample.plot()
    # sample.plot_inputs()
    # sample.plot_outputs('FEM')


#%%

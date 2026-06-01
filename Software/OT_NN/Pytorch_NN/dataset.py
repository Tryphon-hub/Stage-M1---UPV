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
    Load a MATLAB .mat file into a Python dictionary.
    Compatible with MATLAB < 7.3 (scipy) and >= 7.3 (HDF5) formats.
    squeeze_me=True removes spurious dimensions added by MATLAB.

    Parameters:
        filepath (str): Path to the .mat file.

    Returns:
        dict: Dictionary {key: value} of the .mat file.
    """
    import scipy.io
    return scipy.io.loadmat(filepath, squeeze_me=True, struct_as_record=False)


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

        numits = np.atleast_1d(self.NumIts)
        vf     = np.atleast_1d(self.Relative_Vol_Frac)
        return {
            'Tractions'         : torch.from_numpy(self.Tractions[:, :, i]).float().unsqueeze(0),
            'Densities'         : torch.from_numpy(density_ij).float().unsqueeze(0),
            'Relative_Vol_Frac' : torch.tensor(float(vf[i])).float(),
            'Stress'            : torch.from_numpy(self.Stress[i][:, :, j]).float(),
            'FEMc'              : torch.tensor(self.FEMc[i][j]).float(),
            'c'                 : torch.tensor(self.c[i][j]).float(),
            'NumIts'            : torch.tensor(float(numits[i])).float(),
            'ItsFull'           : torch.tensor(self.ItsFull[i]).float(),
            'TEnd'              : torch.tensor(float(self.TEnd)).float(),
        }

    def __repr__(self):
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
                for j in range(int(numits[i]))
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
        }
        return Dataset_TopOpt(sub)


#%% Dataset for all iterations

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
        numits = np.atleast_1d(dataset.NumIts)
        vf     = np.atleast_1d(dataset.Relative_Vol_Frac)

        self.index = [
            (i, j)
            for i in range(len(dataset))
            for j in range(int(numits[i]))
            if 0.15 <= self.get_density(dataset, i, j).mean() <= 0.85
            and 0.15 <= float(vf[i]) <= 0.85
        ]

        self.last_iteration_index = [
            self.index.index((i, int(numits[i]) - 1))
            for i in range(len(dataset))
            if (i, int(numits[i]) - 1) in self.index
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


#%% In case we want to access one specific iteration sample directly

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
        self.FEMc              = sample['FEMc']
        self.c                 = sample['c']
        self.NumIts            = sample['NumIts']
        self.ItsFull           = sample['ItsFull']
        self.TEnd              = sample['TEnd']
        self.UNet_Stress       = None  # Filled after prediction

    def __repr__(self):
        return (f"IterationSample\n"
                f"  Tractions         : {tuple(self.Tractions.shape)}\n"
                f"  Densities         : {tuple(self.Densities.shape)}\n"
                f"  FEM_Stress        : {tuple(self.FEM_Stress.shape)}\n"
                f"  UNet_Stress       : {tuple(self.UNet_Stress.shape) if self.UNet_Stress is not None else 'Not computed'}\n"
                f"  Relative_Vol_Frac : {self.Relative_Vol_Frac.item():.3f}\n"
                f"  c                 : {self.c.item():.6f}\n"
                f"  NumIts            : {int(self.NumIts.item())}\n"
                f"  ItsFull           : {int(self.ItsFull.item())}\n"
                f"  TEnd              : {self.TEnd.item():.4f}")

    def get_traction_distribution(self):
        """
        Build 2D traction images tx and ty by linear interpolation
        of nodal forces along the 4 edges of the square domain.
        The 8 nodal points are distributed 2 per edge (corner + mid-edge).
        Corner pixels accumulate contributions from both adjacent edges.

        Returns:
            ndarray: Stacked traction images, shape (2, img_size, img_size).
                     Channel 0: tx, Channel 1: ty.
        """
        img_size  = int(np.sqrt(self.Densities.shape[1]))
        tx        = np.zeros((img_size, img_size))
        ty        = np.zeros((img_size, img_size))
        tractions = self.Tractions.squeeze().numpy()  # (2, 8)

        Points = np.array([
            [0,          img_size-1],
            [img_size-1, img_size-1],
            [img_size-1, img_size-1],
            [img_size-1, 0         ],
            [img_size-1, 0         ],
            [0,          0         ],
            [0,          0         ],
            [0,          img_size-1],
        ], dtype=float)

        for k in range(0, 8, 2):
            p1 = Points[k]
            p2 = Points[k+1]
            xs = np.round(np.linspace(p1[0], p2[0], img_size)).astype(int)
            ys = np.round(np.linspace(p1[1], p2[1], img_size)).astype(int)
            tx[ys, xs] += np.linspace(tractions[0, k], tractions[0, k+1], img_size)
            ty[ys, xs] += np.linspace(tractions[1, k], tractions[1, k+1], img_size)

        return np.stack([tx, ty], axis=0)  # (2, img_size, img_size)

    def plot(self) -> None:
        """
        Display the optimized topology with boundary force distributions.
        Red: nodal forces. Blue: normal force distributions per edge.

        Returns:
            None
        """
        scale_force = 10
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

        T_scale = self.Tractions.squeeze().numpy() * scale_force  # (2, 8)
        T_scale = T_scale.T                                        # (8, 2)

        q = None
        for k in range(8):
            sx, sy = Points[k] + 0.5
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

            p1 = Points[k]   + 0.5
            p2 = Points[k+1] + 0.5

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

    def plot_inputs(self, TITLE=None) -> None:
        """
        Display the 3 U-Net inputs: densities, tx, ty.
        gray_r colormap [0,1] for densities, symmetric RdBu for tractions.

        Parameters:
            TITLE (str|None): Figure title. Default: 'Inputs'.

        Returns:
            None
        """
        topo     = self.Densities.squeeze().numpy()
        img_size = int(np.sqrt(len(topo)))
        img      = topo.reshape(img_size, img_size)

        tx_ty   = self.get_traction_distribution()
        vmax_tx = np.abs(tx_ty[0]).max()
        vmax_ty = np.abs(tx_ty[1]).max()

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        for ax, img_data, title, cmap, vmin, vmax in zip(
            axes,
            [img, tx_ty[0], tx_ty[1]],
            ['Densities', 'tx', 'ty'],
            ['gray_r', 'RdBu', 'RdBu'],
            [0, -vmax_tx, -vmax_ty],
            [1, vmax_tx, vmax_ty]
        ):
            im = ax.imshow(img_data, cmap=cmap, origin='lower',
                           extent=[0, img_size, 0, img_size],
                           vmin=vmin, vmax=vmax)
            fig.colorbar(im, ax=ax)
            ax.set_title(title, fontsize=14)
            ax.axis('off')

        plt.suptitle(TITLE if TITLE is not None else 'Inputs', fontsize=16)
        plt.tight_layout()
        plt.show()

    def plot_outputs(self, TYPE) -> None:
        """
        Display the 3 stress components: σ_xx, σ_yy, τ_xy.

        Parameters:
            TYPE (str): Stress source. 'FEM' for reference stress fields,
                        'UNet' for predicted stress (requires prior prediction).

        Returns:
            None
        """
        if TYPE == 'FEM':
            stress = self.FEM_Stress.numpy()   # (NumEls, 6)
        elif TYPE == 'UNet':
            assert self.UNet_Stress is not None, "UNet stress not computed yet. Run prediction first."
            stress = self.UNet_Stress.numpy()  # (NumEls, 6)

        img_size = int(np.sqrt(stress.shape[0]))
        sigma_x  = stress[:, 0].reshape(img_size, img_size)
        sigma_y  = stress[:, 1].reshape(img_size, img_size)
        tau_xy   = stress[:, 3].reshape(img_size, img_size)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        for ax, img_data, title in zip(
            axes,
            [sigma_x, sigma_y, tau_xy],
            ['σ_xx', 'σ_yy', 'τ_xy']
        ):
            im = ax.imshow(img_data, cmap='RdBu', origin='lower',
                           extent=[0, img_size, 0, img_size])
            fig.colorbar(im, ax=ax)
            ax.set_title(title, fontsize=14)
            ax.axis('off')

        plt.suptitle(f'{TYPE} outputs', fontsize=16)
        plt.tight_layout()
        plt.show()


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
    Build 2D traction images tx and ty by linear interpolation
    of nodal forces along the 4 edges of the square domain.

    Parameters:
        Tractions (ndarray) : nodal forces, shape (2, 8).
        img_size (int)      : square image size. Default: 32.

    Returns:
        tuple: (tx, ty) two ndarrays of shape (img_size, img_size).
    """
    tx = np.zeros((img_size, img_size))
    ty = np.zeros((img_size, img_size))

    Points = np.array([
        [0,          img_size-1],
        [img_size-1, img_size-1],
        [img_size-1, img_size-1],
        [img_size-1, 0         ],
        [img_size-1, 0         ],
        [0,          0         ],
        [0,          0         ],
        [0,          img_size-1],
    ], dtype=float)

    for k in range(0, 8, 2):
        p1 = Points[k]
        p2 = Points[k+1]
        xs = np.round(np.linspace(p1[0], p2[0], img_size)).astype(int)
        ys = np.round(np.linspace(p1[1], p2[1], img_size)).astype(int)
        tx[ys, xs] += np.linspace(Tractions[0, k], Tractions[0, k+1], img_size)
        ty[ys, xs] += np.linspace(Tractions[1, k], Tractions[1, k+1], img_size)

    return tx, ty


#%% Test

if __name__ == '__main__':
    os.chdir(r'C:\Users\maxen\Documents\Stage')
    print("Current working directory:", Path.cwd())
    path = (Path.cwd() / 'HeavyFiles/data/dataset_test.mat').resolve()
    data = load_mat(path)
    dataset = Dataset_TopOpt(data)

    data_iter = IterationDataset(dataset)

    data0    = dataset.get_series(0)
    ds0_iter = IterationDataset(data0)

    sample = IterationSample(ds0_iter, idx=10)
    print(ds0_iter)

    # sample.plot()
    # sample.plot_inputs()
    # sample.plot_outputs('FEM')
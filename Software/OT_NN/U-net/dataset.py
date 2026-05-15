
#%%  Libraries
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
import scipy.io
import os

#%% Useful functions

def load_mat(filepath: str) -> dict:
    import scipy.io
    return scipy.io.loadmat(filepath, squeeze_me=True, struct_as_record=False)


def inspect_mat(filepath: str) -> None:
    """Affiche un résumé du contenu brut du fichier .mat."""
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
    def __init__(self, dataset):
        self.dataset = dataset
        self.mesh=dataset['MeshData']
        self.Tractions=dataset['Tractions']
        self.Relative_Vol_Frac=dataset['Relative_Vol_Frac']
        self.Rel_Density=dataset['Rel_Density']
        self.Stress=dataset['Stress']
        self.Densities=dataset['Densities']
        self.c=dataset['c']
        self.NumIts=dataset['NumIts']
        self.ItsFull=dataset['ItsFull']
        self.FEMc=dataset['FEMc']
        self.TEnd=dataset['TEnd']

    def __len__(self):
        return len(self.c) # number of traction distributions in the dataset
        
    def __getitem__(self, idx):
        i, j = idx
        return {
            'Tractions'         : torch.from_numpy(self.Tractions[:, :, i]).float().unsqueeze(0),
            'Densities'         : torch.from_numpy(self.Densities[i][:, j]).float().unsqueeze(0),
            'Relative_Vol_Frac' : torch.tensor(self.Relative_Vol_Frac[i]).float(),
            'Stress'            : torch.from_numpy(self.Stress[i][:, :, j]).float(),
            'FEMc'              : torch.tensor(self.FEMc[i][j]).float(),
            'c'                 : torch.tensor(self.c[i][j]).float(),
            'NumIts'            : torch.tensor(self.NumIts[i]).float(),
            'ItsFull'           : torch.tensor(self.ItsFull[i]).float(),
            'TEnd'              : torch.tensor(self.TEnd).float(),
        }
    

    def __repr__(self):
        return (f"Dataset_TopOpt\n"
            f"  NumIts : {[self.NumIts[i] for i in range(len(self.c))]}")

#%% Dataset for all iterations

class IterationDataset(Dataset):
    def __init__(self, dataset: Dataset_TopOpt):
        self.dataset = dataset
        self.index = [
            (i, j)
            for i in range(len(dataset))
            for j in range(int(dataset.NumIts[i]))
        ]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        return self.dataset[self.index[idx]]

    def __repr__(self):
        return (f"IterationDataset\n"
                f"  Sample       : {len(self.index)}\n"
                f"  Index (i,j)  : {self.index}")

    def get_traction_distribution(self, idx):
        sample = IterationSample(self, idx)
        return sample.get_traction_distribution()


#%% In case we want to access one specific iteration sample directly

class IterationSample:
    def __init__(self, dataset: IterationDataset, idx: int):
        sample = dataset[idx]
        self.Tractions         = sample['Tractions']
        self.Densities         = sample['Densities']
        self.Relative_Vol_Frac = sample['Relative_Vol_Frac']
        self.Stress            = sample['Stress']
        self.FEMc              = sample['FEMc']
        self.c                 = sample['c']
        self.NumIts            = sample['NumIts']
        self.ItsFull           = sample['ItsFull']
        self.TEnd              = sample['TEnd']

    def __repr__(self):
        return (f"IterationSample\n"
                f"  Tractions         : {tuple(self.Tractions.shape)}\n"
                f"  Densities         : {tuple(self.Densities.shape)}\n"
                f"  Stress            : {tuple(self.Stress.shape)}\n"
                f"  Relative_Vol_Frac : {self.Relative_Vol_Frac.item():.3f}\n"
                f"  c                 : {self.c.item():.6f}\n"
                f"  NumIts            : {int(self.NumIts.item())}\n"
                f"  ItsFull           : {int(self.ItsFull.item())}\n"
                f"  TEnd              : {self.TEnd.item():.4f}")

    def get_traction_distribution(self):
         # Linear interpolation of the nodal boundary forces
        img_size=int(np.sqrt(self.Densities.shape[1]))

        tx=np.zeros((img_size, img_size))
        ty=np.zeros((img_size, img_size))

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
        import matplotlib.pyplot as plt

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

    def plot_inputs(self) -> None:
        import matplotlib.pyplot as plt

        topo     = self.Densities.squeeze().numpy()
        img_size = int(np.sqrt(len(topo)))
        img      = topo.reshape(img_size, img_size)

        tx_ty = self.get_traction_distribution()

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        for ax, img_data, title, cmap in zip(
            axes,
            [img, tx_ty[0], tx_ty[1]],
            ['Densities', 'tx', 'ty'],
            ['gray_r', 'RdBu', 'RdBu']
        ):
            im = ax.imshow(img_data, cmap=cmap, origin='lower',
                        extent=[0, img_size, 0, img_size])
            fig.colorbar(im, ax=ax)
            ax.set_title(title, fontsize=14)
            ax.axis('off')

        plt.suptitle('U-Net inputs', fontsize=16)
        plt.tight_layout()
        plt.show()

    def plot_outputs(self) -> None:
        import matplotlib.pyplot as plt

        stress   = self.Stress.numpy()  # (1024, 6)
        img_size = int(np.sqrt(stress.shape[0]))

        sigma_x = stress[:, 0].reshape(img_size, img_size)
        sigma_y = stress[:, 1].reshape(img_size, img_size)
        tau_xy  = stress[:, 3].reshape(img_size, img_size)

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

        plt.suptitle('U-Net outputs', fontsize=16)
        plt.tight_layout()
        plt.show()
#%% data loader

def get_dataloader(dataset: IterationDataset, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
    
    n_val   = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val

    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle,   num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,     num_workers=num_workers)
    
    return train_loader, val_loader




#%%Test

if __name__ == '__main__':
    os.chdir(r'C:\Users\maxen\Documents\Stage')
    print("Current working directory:", Path.cwd())
    path = (Path.cwd() / 'Heavy files/data/dataset_test.mat').resolve()
    data = load_mat(path)
    dataset = Dataset_TopOpt(data)

    data_iter = IterationDataset(dataset)

    sample = IterationSample(data_iter, idx=30)
    sample.plot()
    sample.plot_inputs()
    sample.plot_outputs()

#%%
# plot.py
import numpy as np
import matplotlib.pyplot as plt
from dataset import Dataset_TopOpt

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
import scipy.io
import os


def plot_Traction(dataset: Dataset_TopOpt, i: int, j: int) -> None:

    scale_force = 10
    cadre = int(scale_force)

    sample = dataset[i, j]

    # ── Topologie ────────────────────────────────────────────────────
    topo = sample['Densities'].squeeze().numpy()
    img_size = int(np.sqrt(len(topo)))
    img = topo.reshape(img_size, img_size)

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

    # ── Points d'application des forces ──────────────────────────────
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

    T_scale = sample['Tractions'].squeeze().numpy() * scale_force  # (2, 8)
    T_scale = T_scale.T  # (8, 2)

    # Forces nodales (rouge)
    q = None
    for k in range(8):
        sx, sy = Points[k] + 0.5
        tx, ty = T_scale[k]
        q = ax.quiver(sx, sy, tx, ty, angles='xy', scale_units='xy', scale=1,
                      color='r', linewidth=1, headwidth=2)

    # ── Distributions de forces normales (bleu) ───────────────────────
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

    legend = ax.legend([b, q], ['Normal force distributions', 'Side nodal forces'], fontsize=14,
                   handlelength=0.5, handleheight=0.01)
    

    plt.tight_layout()
    plt.show()

#%% Call the function for a sample (i, j)
os.chdir(r'C:\Users\maxen\Documents\Stage')
path = (Path.cwd() / 'Software/OT_Software/data/Test_3_gen.mat').resolve()
data = scipy.io.loadmat(path)
dataset = Dataset_TopOpt(data)
plot_Traction(dataset, i=0, j=-1)


#%% 
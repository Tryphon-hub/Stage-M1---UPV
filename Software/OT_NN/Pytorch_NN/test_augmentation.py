#%% test_augmentation.py
# Test de Data_Augmentation sur un batch de taille 5 puise dans dataset_test.mat
# Verification via la classe IterationSample (plot_inputs / plot_outputs).
import sys
from pathlib import Path
from topology_utils import *

import torch

sys.path.append(str(Path.cwd()))   

from dataset import Dataset_TopOpt, IterationDataset, IterationSample, load_mat
from data_augmentation import Data_Augmentation


def batch_to_samples(batch):
    """
    Transforme un batch (dict) en liste d'IterationSample.
    Construit chaque sample via __new__ pour remplir les champs directement,
    sans repasser par un IterationDataset.
    """
    B = batch['Densities'].shape[0]
    samples = []
    for b in range(B):
        s = IterationSample.__new__(IterationSample)
        s.Tractions         = batch['Tractions'][b]          # [1, 2, 8]
        s.Densities         = batch['Densities'][b]          # [1, 1024]
        s.Relative_Vol_Frac = batch['Relative_Vol_Frac'][b]
        s.FEM_Stress        = batch['Stress'][b]             # [1024, 6]
        s.FEMc              = batch['FEMc'][b]
        s.c                 = batch['c'][b]
        s.NumIts            = batch['NumIts'][b]
        s.ItsFull           = batch['ItsFull'][b]
        s.TEnd              = batch['TEnd'][b]
        s.UNet_Stress       = None
        samples.append(s)
    return samples


#%% ── 1. Charger le dataset ─────────────────────────────────────────────────────
path = (Path.cwd().parents[2] / 'HeavyFiles/data/dataset_test.mat').resolve()
data = load_mat(path)
ds_base = Dataset_TopOpt(data)
ds_iter = IterationDataset(ds_base)

#%% ── 2. Construire un batch de taille 5 ────────────────────────────────────────
loader = torch.utils.data.DataLoader(ds_iter, batch_size=5, shuffle=True)
batch  = next(iter(loader))

#%% ── 3. Samples AVANT augmentation ─────────────────────────────────────────────
orig_batch    = {k: v.clone() for k, v in batch.items() if torch.is_tensor(v)}
samples_orig  = batch_to_samples(orig_batch)

#%% ── 4. Augmentation (p=1.0 pour forcer sur les 5 samples) ─────────────────────
batch         = Data_Augmentation(batch, p=1.0)
samples_aug   = batch_to_samples(batch)

#%% ── 5. Comparaison visuelle sample par sample ─────────────────────────────────
# Pour chaque sample : inputs (densite + tractions) puis outputs (contraintes)
for b in range(5):
    print(f"\n========== SAMPLE {b} ==========")

    print("--- AVANT augmentation ---")
    samples_orig[b].plot_inputs()
    samples_orig[b].plot_outputs('FEM')

    print("--- APRES augmentation ---")
    samples_aug[b].plot_inputs()
    samples_aug[b].plot_outputs('FEM')

print("\nTest termine.")

#%% 

i=1
s=samples_orig[i]
s_aug=samples_aug[i]
compare_NN_FEM(s,s_aug)

s_aug.FEM_Stress = torch.tensor(predict_stress_FEM(eng, s_aug)).float()
s_aug.plot_outputs('FEM')

s_aug.plot()
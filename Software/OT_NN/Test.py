#%% Librairies

import scipy.io
import os
from pathlib import Path

#%% Working directory
os.chdir(r'C:\Users\maxen\Documents\Stage')
print("Current working directory:", Path.cwd())
path = (Path.cwd() / 'Software/OT_Software/data/Test_3_gen.mat').resolve()


#%% Loading data
data = scipy.io.loadmat(path)
print(data.keys())  # voir les variables disponibles

#%% Variables

mesh=data['MeshData']
Tractions=data['Tractions']
Relative_Vol_Frac=data['Relative_Vol_Frac']
Rel_Density=data['Rel_Density']
Stress=data['Stress']
Densities=data['Densities']
c=data['c']
NumIts=data['NumIts']
ItsFull=data['ItsFull']
FEMc=data['FEMc']
TEnd=data['TEnd']
# %%

#%% Import libraries
import sys
import matlab.engine
from pathlib import Path
import re
import csv
import pandas as pd

import torch
import numpy as np
from model         import *
from dataset       import *
from topology_utils import *


name_file = 'dataset_1k'
name_data = 'dataset_test'

# NETWORK   = 'BE_UNet'
NETWORK   = 'U-Net'


NIF = 32
N_CONV=2
HIDDEN_LAYERS_MLP=[32,64]
EMBED_OUT   = 128     # dimension de l'embedding

USE_CBAM = False
USE_AUGMENTATION = False  # Data augmentation

AUGMENTATION_P   = 0.2
PORTION_DATA     = 1

BATCH_SIZE = 16



if NETWORK=='BE_UNet':
    N_in=1
else:
    N_in=3




BASE = Path(__file__).parents[3]


GMSH_EXE = BASE.parent / 'gmsh' / 'gmsh.exe'


DATA_PATH       = BASE / 'HeavyFiles' / 'data' / (name_data + '.mat')

aug_tag = f'{int(100*AUGMENTATION_P)}%' if USE_AUGMENTATION else 'False'

if NETWORK == 'U-Net':
    tag = f'{name_file}_NIF={NIF}_{N_CONV}_conv_CBAM={USE_CBAM}_aug={aug_tag}_portion={int(PORTION_DATA*100)}%_batch={BATCH_SIZE}'
else:
    mlp_tag = '-'.join(str(h) for h in HIDDEN_LAYERS_MLP)
    tag = f'{name_file}_NIF={NIF}_{N_CONV}_conv_{mlp_tag}_CBAM={USE_CBAM}_aug={aug_tag}_portion={int(PORTION_DATA*100)}%_batch={BATCH_SIZE}'

RESULTS_DIR       = BASE / 'Software' / 'OT_NN' / 'Pytorch_NN' / 'results' / NETWORK / tag
ILLUSTRATIONS_DIR = BASE / 'Software' / 'OT_NN' / 'Pytorch_NN' / 'illustrations' / NETWORK / tag
CHECKPOINT_PATH   = RESULTS_DIR / ('unet_' + name_file + '_checkpoint.pth')
BEST_PATH         = RESULTS_DIR / ('unet_' + name_file + '_best.pth')
TB_LOG_DIR        = RESULTS_DIR / ('runs_' + name_file) / ('unet_' + name_file)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


sys.path.append(str(BASE / 'Software' / 'OT_NN' / 'Pytorch_NN'))
sys.path.append(str(BASE / 'Software' / 'OT_Functions'))
sys.path.append(str(BASE / 'Software' / 'OT_Software'))



#%% Constants
IMG_SIZE = 32
PENAL    = 3
RMIN     = 1.5
NGPpL    = 2
NGPpS    = 9
E        = 1000
NU       = 0.3


#%% Load model

if NETWORK=='BE_UNet':
    model = BE_UNetTopo(
        nif           = NIF,
        n_in          = N_in,          # ρ seul — tractions via BoundaryEmbedding
        n_out         = 3,
        use_cbam      = USE_CBAM,
        hidden_layers_MLP = HIDDEN_LAYERS_MLP,
        embed_out     = EMBED_OUT,
        N_conv=N_CONV,
    )
    
elif NETWORK=='U-Net':
    model = UNetTopo(
        nif=32, 
        n_in=N_in, 
        n_out=3, 
        use_cbam=USE_CBAM,
        N_conv=N_CONV,
        )

else:
    raise ValueError("Invalid NETWORK value. Choose 'U-net' or 'BE_Unet'.")

state_dict = torch.load(
    BEST_PATH,
    map_location='cpu'
)

model.load_state_dict(state_dict)
model.eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"Number of {NETWORK} parameters : {n_params:_}".replace('_',' '))


#%% Load dataset
data    = load_mat(DATA_PATH)
ds_base = Dataset_TopOpt(data)
ds_filtre = ds_base.filtre_dataset(rho_min=0.15, rho_max=0.85)
IterData_FEM = IterationDataset(ds_filtre)

List_List_iterations=[]



#%% Start MATLAB engine
eng = matlab.engine.start_matlab()
eng.addpath(str(BASE / 'Software' / 'OT_Functions'))
eng.addpath(str(BASE / 'Software' / 'OT_Software'))

#%% Regenerate and load the mesh

geo_file  = BASE / 'Software' / 'OT_Software' / 'Square.geo'
mesh_file = BASE / 'Software' / 'OT_Software' / 'Square.msh'

eng.workspace['GmshExe']     = str(GMSH_EXE)
eng.workspace['GeoFileName'] = str(geo_file)
eng.workspace['Mesh_File']   = str(mesh_file)

print(f"GmshExe exists: {GMSH_EXE.exists()}")
print(f"GeoFileName exists: {geo_file.exists()}")

eng.eval(rf"""
if isfile(Mesh_File)
    delete(Mesh_File)
end
CallString = ['"' GmshExe '" "' GeoFileName '" -setnumber numLayers {IMG_SIZE} -o "' Mesh_File '" -'];
disp(CallString)
status = system(CallString);
disp(['system() status: ' num2str(status)])
[MeshData] = ReadGMSH(Mesh_File);
""", nargout=0)

print(f"NumEls after regen: {eng.eval('length(MeshData.Surf.Elements)')}")

eng.eval("D = DHooks2D(1000, 0.3, 'Plane Stress');", nargout=0)


#%% One sample

ID_distrib = 67

idx_FEM_sol = IterData_FEM.last_iteration_index[ID_distrib]
FEM_sample  = IterationSample(IterData_FEM, idx_FEM_sol)

sample_start = IterationSample(IterationDataset(ds_filtre.get_series(ID_distrib)), 0)

ds_iter=IterationDataset(ds_filtre.get_series(ID_distrib))

list_images=[
    ['Only UNet', 'UNet', 3, False],
    ['Only UNet', 'UNet', 3, True],
    ['Only UNet', 'FEM',  3, True],
    ['10 Unet - 1 FEM', 'UNet', 3, True],
    ['10 Unet - 1 FEM', 'FEM',  3, True],
    ['Decreasing compliance', 'UNet', 3, True],
    ['Decreasing compliance', 'FEM',  3, True],
]

COMPUTE_FEM_COMPLIANCE = True

if COMPUTE_FEM_COMPLIANCE:
    ENG = eng
else:
    ENG = None

for Strategy, Type_First, window_Unet, end_FEM in list_images:
    print(f"Running strategy: {Strategy}, Type_First: {Type_First}, window_Unet: {window_Unet}, end_FEM: {end_FEM}")


    List_iterations, List_count_FEM = run_topology_optimization(
            sample_start,
            eng, 
            model, 
            N_in = N_in,
            N_max_iterations = 100, 
            RULE = Strategy,
            TYPE_FIRST = Type_First,
            threshold = 0.0,
            N_end_FEM_iterations = 0,
            window_Unet = window_Unet,
            window_FEM = 1, 
            tol_c=1e-3, 
            tol_rho=0.1,
            end_FEM=end_FEM,
            )


    save_path = Path(r'C:\Users\maxen\Pictures\Screenshots\Hybrid strategy\ID=67')
    save_path.mkdir(parents=True, exist_ok=True)

    img_convergence_path = save_path / f'convergence_{NETWORK}_ID={ID_distrib}_{Strategy}_TypeFirst={Type_First}_windowUnet={window_Unet}_endFEM={end_FEM}.png'

    FEM_c, UNet_c = visualize_convergence(
        List_iterations, 
        ds_iter, 
        List_count_FEM,
        NETWORK=NETWORK,
        PLOT=True,
        SCALE='linear',
        eng=ENG,   # re-evaluate each iterate with FEM: removes U-Net pseudo-compliance spikes
        SAVE_PATH= img_convergence_path,
        SHOW_FEM_C=True,
        )


    img_comparison_path = save_path / f'comparison_{NETWORK}_ID={ID_distrib}_{Strategy}_TypeFirst={Type_First}_windowUnet={window_Unet}_endFEM={end_FEM}.png'

    compare_NN_FEM(List_iterations[-1], FEM_sample, SAVE_PATH=img_comparison_path)

    density_evolution(List_iterations,List_count_FEM, 1)

    c_FEM  = FEM_sample.c.item()

    c_Unet = compliance_FEM(eng, List_iterations[-1])

    err_rel_c  = (c_Unet - c_FEM) / c_FEM


#%%
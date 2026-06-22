#%% Import libraries
import sys
import matlab.engine
from pathlib import Path
import re

user      = 'laptop'
# user      = 'server'

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

PORTION_DATA = 1


if NETWORK=='BE_UNet':
    N_in=1
else:
    N_in=3


BASE = Path(__file__).parents[3]

# GMSH_EXE = Path(r'C:\Program Files\gmsh\gmsh.exe')
GMSH_EXE = Path(r'D:\Maxence\gmsh\gmsh.exe')


DATA_PATH       = BASE / 'HeavyFiles' / 'data' / (name_file + '.mat')
if NETWORK == 'U-Net':
    RESULTS_DIR     = BASE / 'Software' / 'OT_NN' / 'Pytorch_NN' / 'results' / NETWORK / f'{name_file}_{N_CONV}_conv_CBAM={USE_CBAM}_aug={USE_AUGMENTATION}_portion={PORTION_DATA*100:.0f}%'
    ILLUSTRATIONS_DIR = BASE / 'Software' / 'OT_NN' / 'Pytorch_NN' / 'illustrations' / NETWORK / f'{name_file}_{N_CONV}_conv_CBAM={USE_CBAM}_aug={USE_AUGMENTATION}_portion={PORTION_DATA*100:.0f}%'
    CHECKPOINT_PATH = RESULTS_DIR / ('unet_' + name_file + '_checkpoint.pth')
    BEST_PATH       = RESULTS_DIR / ('unet_' + name_file + '_best.pth')
    TB_LOG_DIR      = RESULTS_DIR / ('runs_' + name_file) / ('unet_' + name_file)



else:
    RESULTS_DIR     = BASE / 'Software' / 'OT_NN' / 'Pytorch_NN' / 'results' / NETWORK / f'{name_file}_{N_CONV}_conv_{HIDDEN_LAYERS_MLP}_CBAM={USE_CBAM}_aug={USE_AUGMENTATION}_portion={PORTION_DATA*100:.0f}%'
    ILLUSTRATIONS_DIR = BASE / 'Software' / 'OT_NN' / 'Pytorch_NN' / 'illustrations' / NETWORK / f'{name_file}_{N_CONV}_conv_{HIDDEN_LAYERS_MLP}_CBAM={USE_CBAM}_aug={USE_AUGMENTATION}_portion={PORTION_DATA*100:.0f}%'
    CHECKPOINT_PATH = RESULTS_DIR / ('unet_' + name_file + '_checkpoint.pth')
    BEST_PATH       = RESULTS_DIR / ('unet_' + name_file + '_best.pth')
    TB_LOG_DIR      = RESULTS_DIR / ('runs_' + name_file) / ('unet_' + name_file)





sys.path.append(str(BASE / 'Software' / 'OT_NN' / 'Pytorch_NN'))
sys.path.append(str(BASE / 'Software' / 'OT_Functions'))
sys.path.append(str(BASE / 'Software' / 'OT_Software'))

import torch
import numpy as np
from model         import *
from dataset       import *
from topology_utils import *

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

# n_total = sum(p.numel() for p in model.parameters())
# n_embed = sum(p.numel() for p in model.boundary_embedding.parameters())

# print(f"Total           : {n_total:_}".replace('_', ' '))
# print(f"dont embedding  : {n_embed:_}".replace('_', ' '))
# print(f"reste U-Net     : {n_total - n_embed:_}".replace('_', ' '))


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
#%% Plot

# FEM_c, UNet_c = statistical_convergence(
#     List_List_iterations, 
#     IterData_FEM, 
#     NETWORK=NETWORK, 
#     PLOT=True, 
#     TYPE='std'
#     )






#%% One sample

# ID_distrib=3
# idx_FEM_sol = IterData_FEM.last_iteration_index[ID_distrib]
# FEM_sample  = IterationSample(IterData_FEM, idx_FEM_sol)

# ds_iter=IterationDataset(ds_filtre.get_series(ID_distrib))

# List_iterations, List_count_FEM = run_topology_optimization(
#         ds_filtre, 
#         ID_distrib, 
#         eng, model, 
#         N_in = N_in,
#         N_max_iterations = 100, 
#         # RULE = 'Only UNet',
#         # RULE = '10 Unet - 1 FEM',
#         RULE = 'Decreasing compliance', 
#         # RULE = 'Only FEM',
#         # TYPE_FIRST = 'UNet',
#         TYPE_FIRST = 'FEM',
#         threshold = 0.0,
#         N_end_FEM_iterations = 0,
#         window_Unet = 5,
#         window_FEM = 1, 
#         tol_c=1e-3, 
#         tol_rho=0.1,
#         end_FEM=False
#         )

# FEM_c, UNet_c = visualize_convergence(
#     List_iterations, 
#     ds_iter, 
#     List_count_FEM,
#     NETWORK=NETWORK,
#     PLOT=True,
#     SCALE='linear'
#     )


# compare_NN_FEM(List_iterations[-1], FEM_sample)

# density_evolution(List_iterations,List_count_FEM, 1)

#%% Main loop : evaluate method

SIZE_LOOP = 10


list_benchmark = [
    ['Only UNet', 'UNet'],
    ['Only UNet', 'FEM'],
    ['10 Unet - 1 FEM', 'UNet'],
    ['10 Unet - 1 FEM', 'FEM'],
    ['10 Unet - 3 FEM', 'UNet'],
    ['10 Unet - 3 FEM', 'FEM'],
    ['Decreasing compliance', 'UNet'],
    ['Decreasing compliance', 'FEM'],
]


Tab_number_FEM = []
Tab_err_rel_c = []


# window showing the avances of the process
total = len(list_benchmark) * SIZE_LOOP
win = ProgressWindow(total)
thread = threading.Thread(target=run_window, args=(win,), daemon=True)
thread.start()

#main loop
for i in range(len(list_benchmark)):
    List_number_FEM = []
    List_err_rel_c = []

    # loop for each distribution ID
    for ID in range(SIZE_LOOP):
        List_iterations, List_count_FEM = run_topology_optimization(
            ds_filtre, 
            ID, 
            eng, model, 
            N_in = N_in,
            N_max_iterations = 100, 
            RULE = list_benchmark[i][0],
            TYPE_FIRST = list_benchmark[i][1],
            threshold = 0.0,
            N_end_FEM_iterations = 0,
            window_Unet = 5,
            window_FEM = 1, 
            tol_c=1e-3, 
            tol_rho=0.1,
            end_FEM=True
            )
        
        idx_FEM_sol = IterData_FEM.last_iteration_index[ID]
        FEM_sample  = IterationSample(IterData_FEM, idx_FEM_sol)

        c_FEM   = FEM_sample.c.item()
        c_Unet  = List_iterations[-1].c.item()
        
        List_err_rel_c.append(abs(c_FEM - c_Unet)/c_FEM)
        List_number_FEM.append(len(List_count_FEM))

        # window actualisation
        win.increment()

    Tab_number_FEM.append(List_number_FEM)
    Tab_err_rel_c.append(List_err_rel_c)

Tab_number_FEM = np.array(Tab_number_FEM)
Tab_err_rel_c  = np.array(Tab_err_rel_c)

win.close()

plot_FEM_error_c(list_benchmark, Tab_number_FEM, Tab_err_rel_c)


#%%
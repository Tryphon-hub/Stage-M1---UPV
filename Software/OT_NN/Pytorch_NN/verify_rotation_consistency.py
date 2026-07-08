import sys
from pathlib import Path
import numpy as np, torch, matlab, matlab.engine
BASE=Path(__file__).parents[3]
for p in ['Software/OT_NN/Pytorch_NN','Software/OT_Functions','Software/OT_Software']: sys.path.append(str(BASE/p))
import dataset as ds
from dataset import load_mat,Dataset_TopOpt,IterationDataset,IterationSample,rotation_90,symmetry_x,symmetry_y
from topology_utils import predict_stress_FEM
IMG,E,NU=32,1000,0.3
GMSH=BASE.parent/'gmsh'/'gmsh.exe'; GEO=BASE/'Software/OT_Software/Square.geo'; MSH=BASE/'Software/OT_Software/Square.msh'
eng=matlab.engine.start_matlab(); eng.addpath(str(BASE/'Software/OT_Functions')); eng.addpath(str(BASE/'Software/OT_Software'))
eng.workspace['GmshExe']=str(GMSH); eng.workspace['GeoFileName']=str(GEO); eng.workspace['Mesh_File']=str(MSH)
eng.eval(rf"""if isfile(Mesh_File); delete(Mesh_File); end
system(['"' GmshExe '" "' GeoFileName '" -setnumber numLayers {IMG} -o "' Mesh_File '" -']); [MeshData]=ReadGMSH(Mesh_File);""",nargout=0)
eng.eval(f"D=DHooks2D({E},{NU},'Plane Stress');",nargout=0)
data=load_mat((BASE/'HeavyFiles/data/dataset_test.mat').resolve()); s0=IterationSample(IterationDataset(Dataset_TopOpt(data)),30)
def mk(T):
    s=s0.copy(); s.Densities=torch.full((1,IMG*IMG),0.5); s.Tractions=torch.from_numpy(T).float().unsqueeze(0); predict_stress_FEM(eng,s); return s
BEND=np.zeros((2,8)); BEND[0,[2,7]]=1; BEND[0,[3,6]]=-1
SHEAR=np.zeros((2,8)); SHEAR[1,[0,1,4,5]]=-1; SHEAR[1,[2,3,6,7]]=1
def chk(s,fn,nm):
    aug=fn(s.copy()); claimed=aug.FEM_Stress.numpy(); a2=aug.copy(); predict_stress_FEM(eng,a2); fem=a2.FEM_Stress.numpy()
    c=[0,1,3]; e=np.abs(claimed[:,c]-fem[:,c]).max()/(np.abs(fem[:,c]).max()+1e-9)
    print(f"  {nm:22s}: err={e:.2e}"+("  OK" if e<1e-2 else "  BAD"))
for load,ln in [(BEND,"bending"),(SHEAR,"shear")]:
    s=mk(load)
    for k in (1,2,3): chk(s,lambda x,k=k:rotation_90(x,N_rot=k),f"{ln} rot{k*90}")
    chk(s,symmetry_x,f"{ln} flip_x"); chk(s,symmetry_y,f"{ln} flip_y")

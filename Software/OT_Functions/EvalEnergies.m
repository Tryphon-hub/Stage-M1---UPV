% function Energy_map = EvalEnergies(MeshData,Tractions,Relative_Vol_Frac,ProbInfo,IniDentsity)

addpath C:\Users\maxen\Documents\Stage\Software\OT_Functions
addpath C:\Users\maxen\Documents\Stage\Software\OT_Software
addpath C:\Users\maxen\Documents\Stage\Software\OT_NN\U-net

Relative_Vol_Frac = 0.5*ones(1, NumSamples);


load dataset_test, and compute energy strain for first sample in each of the 100 traction distributions. Use this formula : 


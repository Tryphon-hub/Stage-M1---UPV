% addpath 'D:\Maxence\Stage-M1---UPV\Software\OT_Functions'
% addpath 'D:\Maxence\Stage-M1---UPV\Software\OT_Software'

% DatasetFile = 'D:\Maxence\Stage-M1---UPV\HeavyFiles\data\dataset_1k.mat';

addpath C:\Users\maxen\Documents\Stage\Software\OT_Functions
addpath C:\Users\maxen\Documents\Stage\Software\OT_Software

DatasetFile = 'C:\Users\maxen\Documents\Stage\HeavyFiles\data\dataset_macro_cantilever.mat';

fprintf('Chargement du dataset...\n');
data = load(DatasetFile, 'Rel_Density', 'Tractions', 'MeshData');

MeshData      = data.MeshData;
Rel_Density_1 = data.Rel_Density(:, 1);  % densité de la première image
Tractions_all = data.Tractions;           % (2, 8, NumSamples)

NumSamples = size(Tractions_all, 3);
NumEls     = length(MeshData.Surf.Elements);

%% Paramètres identiques à GenerateSamples.m
E      = 1000;
nu     = 0.3;
Penal  = 3;
NGPpS  = 9;
NGPpL  = 2;
SquareEls = true;
ElemTO    = true;

D = DHooks2D(E, nu, 'Plane Stress');

Top = MeshData.Surf.Topology;
XY  = MeshData.XYZ;

Ener = cell(NumSamples, 1);

fprintf('Calcul de Ener pour %d distributions de traction...\n', NumSamples);
for iSample = 1:NumSamples
    Tvert = Tractions_all(:, :, iSample);

    Sol = SolveFE(MeshData, Rel_Density_1.^Penal, NGPpS, NGPpL, D, Tvert, SquareEls, ElemTO);

    [Stress_i, ~, Strain_i] = EvalStress(Top, XY, Rel_Density_1.^1, D, Sol, 2, SquareEls, ElemTO, 'Plane Stress', E, nu);

    Ener{iSample} = Stress_i .* Strain_i;  % (NumEls, 6)

    fprintf('  Sample %d/%d done\n', iSample, NumSamples);
end

save(DatasetFile, 'Ener', '-append');
fprintf('Ener sauvegardé dans %s\n', DatasetFile);



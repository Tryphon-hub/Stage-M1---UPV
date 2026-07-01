% addpath 'D:\Maxence\Stage-M1---UPV\Software\OT_Functions'
% addpath 'D:\Maxence\Stage-M1---UPV\Software\OT_Software'

addpath C:\Users\maxen\Documents\Stage\Software\OT_Functions
addpath C:\Users\maxen\Documents\Stage\Software\OT_Software

DatasetFile = 'C:\Users\maxen\Documents\Stage\HeavyFiles\data\dataset_macro_cantilever.mat';

fprintf('Chargement du dataset...\n');
data = load(DatasetFile);

NumSamples = size(data.Tractions, 3);
NumEls     = length(data.MeshData.Surf.Elements);

%% Keep only the first 2 iterations for each force distribution, forced to 0.5
for iSample = 1:NumSamples
    nIter = data.NumIts(iSample);
    nKeep = min(2, nIter);

    data.Densities{iSample} = 0.5 * ones(NumEls, nKeep);
    data.Stress{iSample}    = data.Stress{iSample}(:, :, 1:nKeep);
    data.c{iSample}         = data.c{iSample}(1:nKeep);
    data.FEMc{iSample}      = data.FEMc{iSample}(1:nKeep);

    data.NumIts(iSample)  = nKeep;
    data.ItsFull(iSample) = nKeep;
end

data.Rel_Density = 0.5 * ones(NumEls, NumSamples);

NewDatasetFile = strrep(DatasetFile, 'dataset_macro_cantilever.mat', 'dataset_macro_cantilever_2iter_05.mat');
save(NewDatasetFile, '-struct', 'data');
fprintf('Dataset (2 premières itérations, densité 0.5) sauvegardé dans %s\n', NewDatasetFile);
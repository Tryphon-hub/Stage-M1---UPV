addpath 'D:\Maxence\Stage-M1---UPV\Software\OT_Functions'
addpath 'D:\Maxence\Stage-M1---UPV\Software\OT_Software'

Folder='D:\Maxence\Stage-M1---UPV\HeavyFiles\data';

% Folder='C:\Users\maxen\Documents\Stage\HeavyFiles\data';

% Ensure the folder exists before proceeding
if ~isfolder(Folder)
    error('The specified folder does not exist: %s', Folder);
end


FileName='dataset_energy';
TractionFile='tractions_energy';
GenerateNewTractions=true;
Net=[];
NumSamples=10;


GenerateSamples(Folder,FileName,TractionFile,GenerateNewTractions,Net,NumSamples);

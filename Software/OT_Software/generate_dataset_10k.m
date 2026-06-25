% addpath 'D:\Maxence\Stage-M1---UPV\Software\OT_Functions'
% addpath 'D:\Maxence\Stage-M1---UPV\Software\OT_Software'

% Folder='D:\Maxence\Stage-M1---UPV\HeavyFiles\data';

addpath 'C:\Users\maxen\Documents\Stage\Software\OT_Functions'
addpath 'C:\Users\maxen\Documents\Stage\Software\OT_Software'

Folder='C:\Users\maxen\Documents\Stage\HeavyFiles\data';

% Ensure the folder exists before proceeding
if ~isfolder(Folder)
    error('The specified folder does not exist: %s', Folder);
end


FileName='dataset_10k';
TractionFile='tractions_10k';
GenerateNewTractions=true;
Net=[];
NumSamples=10;
save_last_only = true;


GenerateSamples(Folder,FileName,TractionFile,GenerateNewTractions,Net,NumSamples, save_last_only);

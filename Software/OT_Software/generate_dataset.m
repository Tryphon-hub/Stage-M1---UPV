%% Path
addpath 'D:\Maxence\Stage-M1---UPV\Software\OT_Functions'
addpath 'D:\Maxence\Stage-M1---UPV\Software\OT_Software'
Folder='D:\Maxence\Stage-M1---UPV\HeavyFiles\data';

% addpath 'C:\Users\maxen\Documents\Stage\Software\OT_Functions'
% addpath 'C:\Users\maxen\Documents\Stage\Software\OT_Software'
% Folder = 'C:\Users\maxen\Documents\Stage\HeavyFiles\data';

% Ensure the folder exists before proceeding
if ~isfolder(Folder)
    error('The specified folder does not exist: %s', Folder);
end

%% Input parameters
FileName             = 'dataset_10k';
TractionFileName     = 'tractions_10k';
GenerateNewTractions = true;
Net                  = [];
NumSamples           = 10000;
save_last_only       = true;


%% Data generation
GenerateSamples(Folder, FileName, TractionFileName, GenerateNewTractions, Net, NumSamples, save_last_only);
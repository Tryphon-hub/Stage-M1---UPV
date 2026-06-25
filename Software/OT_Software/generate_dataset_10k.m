%% Path
% addpath 'D:\Maxence\Stage-M1---UPV\Software\OT_Functions'
% addpath 'D:\Maxence\Stage-M1---UPV\Software\OT_Software'
% Folder='D:\Maxence\Stage-M1---UPV\HeavyFiles\data';
addpath 'C:\Users\maxen\Documents\Stage\Software\OT_Functions'
addpath 'C:\Users\maxen\Documents\Stage\Software\OT_Software'
Folder = 'C:\Users\maxen\Documents\Stage\HeavyFiles\data';

% Ensure the folder exists before proceeding
if ~isfolder(Folder)
    error('The specified folder does not exist: %s', Folder);
end

%% Input parameters
FileName             = 'dataset_10k';
TractionFileName     = 'tractions_10k';
GenerateNewTractions = false;
Net                  = [];
NumSamples           = 10;
save_last_only       = false;

%% Extract tractions from dataset and save them as TractionFile
macro       = load(fullfile(Folder, FileName), 'Tractions');
EqTractions = macro.Tractions;                        % (2, 8, N)
save(fullfile(Folder, TractionFileName), 'EqTractions');

%% Check consistency
New_FileName = [FileName, '_05'];
A = load(fullfile(Folder, 'dataset_10k.mat'),    'Tractions');
B = load(fullfile(Folder, 'dataset_10k_05.mat'), 'Tractions');
isequal(A.Tractions, B.Tractions)
max(abs(A.Tractions(:) - B.Tractions(:)))

%% Data generation
GenerateSamples(Folder, New_FileName, TractionFileName, GenerateNewTractions, Net, NumSamples, save_last_only);
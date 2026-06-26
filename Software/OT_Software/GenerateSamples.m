function GenerateSamples(Folder,FileName,TractionFile,GenerateNewTractions,Net,NumSamples,save_last_only)
arguments
    Folder
    FileName
    TractionFile
    GenerateNewTractions
    Net
    NumSamples
    save_last_only = false
end
clc
addpath 'D:\Maxence\Stage-M1---UPV\Software\OT_Functions'
addpath 'D:\Maxence\Stage-M1---UPV\Software\OT_Software'

addpath C:\Users\maxen\Documents\Stage\Software\OT_Functions
addpath C:\Users\maxen\Documents\Stage\Software\OT_Software

%% Generate Mesh
GeoFileName = 'D:\Maxence\Stage-M1---UPV\Software\OT_Software\Square.geo';
Mesh_File   = 'D:\Maxence\Stage-M1---UPV\Software\OT_Software\Square.msh';

% GeoFileName = 'C:\Users\maxen\Documents\Stage\Software\OT_Software\Square.geo';
% Mesh_File   = 'C:\Users\maxen\Documents\Stage\Software\OT_Software\Square.msh';

if isfile(Mesh_File)
    delete(Mesh_File)
end
CallString = ['"D:\Maxence\gmsh\gmsh.exe" "' GeoFileName '" -setnumber numLayers 32 -o "' Mesh_File '" -'];

% CallString = ['"C:\Users\maxen\Documents\gmsh\gmsh.exe" "' GeoFileName '" -setnumber numLayers 32 -o "' Mesh_File '" -'];


system(CallString);
[MeshData] = ReadGMSH(Mesh_File);

%% Create output folder
if ~exist(Folder,'dir')
    mkdir(Folder)
end

%% Cell information
if NumSamples == -1
    AutoEquilibrate = 0;
    [Tractions,Center,Scale,List,Relative_Vol_Frac] = LoadMacroProblemData('Cantilever_MacroTO_Info.mat',AutoEquilibrate);
    NumSamples = size(Tractions,3);
else
    Tractions = GenerateTractions(NumSamples,[Folder '\' TractionFile],GenerateNewTractions);
    Center = [0;0]*ones(1,NumSamples);
    Scale = 1*ones(1,NumSamples);
    List = 1:NumSamples;

    % In case user wants to generate random relative volumic fractions, uncomment next lines
    % rho_min=0.15;
    % rho_max=0.85;
    % Relative_Vol_Frac = rho_min+(rho_max-rho_min)*rand(1,NumSamples);

    Relative_Vol_Frac = 0.5*ones(1, NumSamples);
end

%% TO conditions
ProbInfo.NGPpL = 2;
ProbInfo.NGPpS = 9;
ProbInfo.E = 1000;
ProbInfo.nu = 0.3;
ProbInfo.Penal = 3;
ProbInfo.rmin = 1.5;
ProbInfo.NumIt = 1000;
ProbInfo.ElemTO = true;
ProbInfo.SquareEls = true;
ProbInfo.Folder = Folder;
ProbInfo.Net = Net;

NumEls = length(MeshData.Surf.Elements);
Rel_Density = zeros(NumEls,NumSamples);
Stress = cell(NumSamples,1);
Strain = cell(NumSamples,1);
Densities = cell(NumSamples,1);
Ener = cell(NumSamples,1);
c = cell(NumSamples,1);
FEMc = cell(NumSamples,1);
NumIts = zeros(NumSamples,1);
ItsFull = zeros(NumSamples,1);

D = DHooks2D(ProbInfo.E, ProbInfo.nu, 'Plane Stress');

%% Initialisation sauvegarde incrémentale
m = matfile([Folder '\' FileName], 'Writable', true);
m.MeshData          = MeshData;
m.Tractions         = Tractions;
m.Relative_Vol_Frac = Relative_Vol_Frac;
m.Rel_Density       = zeros(NumEls, NumSamples);
m.Stress            = cell(NumSamples, 1);
m.Ener              = cell(NumSamples, 1);
m.Densities         = cell(NumSamples, 1);
m.NumIts            = zeros(NumSamples, 1);
m.ItsFull           = zeros(NumSamples, 1);
m.c                 = cell(NumSamples, 1);
m.FEMc              = cell(NumSamples, 1);
m.TEnd              = 0;

%% Generate topology
TStart = tic;
% figure
MainFig = gcf;
hold on

pbar = waitbar(0, 'Iniciando Optimización Topológica...');

for iSample = List
    waitbar(iSample/NumSamples, pbar, sprintf('Progreso: %d/%d', iSample, NumSamples));

    % Modified by Maxence Barberet-Pinto: a constant density field is generated for each traction distribution
    IniDentsity = Relative_Vol_Frac(iSample) * ones(size(Rel_Density(:,iSample)));

    [Rel_Density(:,iSample),Stress{iSample},Strain{iSample},Densities{iSample},NumIts(iSample),ItsFull(iSample),c{iSample},FEMc{iSample}] = ...
        GenTopology(MeshData,Tractions(:,:,iSample),Relative_Vol_Frac(iSample),ProbInfo,IniDentsity);

    Sol_ini = SolveFE(MeshData, IniDentsity.^ProbInfo.Penal, ProbInfo.NGPpS, ProbInfo.NGPpL, D, Tractions(:,:,iSample), ProbInfo.SquareEls, ProbInfo.ElemTO);
    [Stress_ini, ~, Strain_ini] = EvalStress(MeshData.Surf.Topology, MeshData.XYZ, IniDentsity, D, Sol_ini, 2, ProbInfo.SquareEls, ProbInfo.ElemTO, 'Plane Stress', ProbInfo.E, ProbInfo.nu);
    Ener{iSample} = Stress_ini .* Strain_ini;

    if save_last_only
        Stress{iSample}    = Stress{iSample}(:, :, [1, end]);
        Strain{iSample}    = Strain{iSample}(:, :, [1, end]);
        Densities{iSample} = Densities{iSample}(:, [1, end]);
    end

    %% Sauvegarde incrémentale
    m.Rel_Density(:, iSample)  = Rel_Density(:, iSample);
    m.Stress(iSample, 1)       = {Stress{iSample}};
    m.Ener(iSample, 1)         = {Ener{iSample}};
    m.Densities(iSample, 1)    = {Densities{iSample}};
    m.NumIts(iSample, 1)       = NumIts(iSample);
    m.ItsFull(iSample, 1)      = ItsFull(iSample);
    m.c(iSample, 1)            = {c{iSample}};
    m.FEMc(iSample, 1)         = {FEMc{iSample}};

    Top = MeshData.Surf.Topology;
    XY = MeshData.XYZ;

    if ProbInfo.ElemTO
        Value = repmat(Rel_Density(:,iSample)',size(Top,1),1);
    else
        Value = reshape(Rel_Density(Top,iSample),size(Top,1),[]);
    end
    Xval = reshape(XY(1,Top),size(Top,1),[])*Scale(iSample) + Center(1,iSample);
    Yval = reshape(XY(2,Top),size(Top,1),[])*Scale(iSample) + Center(2,iSample);

    % figure(MainFig)
    patch(Xval,Yval,-Value,'EdgeColor','none');
    axis equal
    caxis([-1,0])
    colormap gray
    drawnow
end

close(pbar);
TEnd = toc(TStart);
m.TEnd = TEnd;

saveas(gcf,[Folder '\' FileName])

end
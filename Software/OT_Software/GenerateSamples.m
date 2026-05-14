function GenerateSamples(Folder,FileName,TractionFile,GenerateNewTractions,Net,NumSamples)
clc
addpath 'D:\Maxence\Stage-M1---UPV\Software\OT_Functions'
addpath 'D:\Maxence\Stage-M1---UPV\Software\OT_Software'

%% Generate Mesh
GeoFileName = 'D:\Maxence\Stage-M1---UPV\Software\OT_Software\Square.geo';
Mesh_File   = 'D:\Maxence\Stage-M1---UPV\Software\OT_Software\Square.msh';

if isfile(Mesh_File)
    delete(Mesh_File)
end
CallString = ['"D:\Maxence\gmsh\gmsh.exe" "' GeoFileName '" -setnumber numLayers 32 -o "' Mesh_File '" -'];
system(CallString);
[MeshData] = ReadGMSH(Mesh_File);

%% Create output folder
if ~exist(Folder,'dir')
    mkdir(Folder)
end
%% Cell information
% Tractions = [  tn_s1_n1   tn_s1_n2 ...
%                tt_s1_n1   tt_s1_n2 ...
% Tractions = [  0   0   3   3  0  0  1  1;...
%               -1  -1   0   0  1  1  0  0];

%TractionFile = 'TraccionesPueba.mat';

if ~exist(Folder,'dir')
    mkdir(Folder)
end

if NumSamples == -1
    AutoEquilibrate = 0;
    [Tractions,Center,Scale,List,Relative_Vol_Frac] = LoadMacroProblemData('Beam_MacroTO_Info.mat',AutoEquilibrate);
    % 'Beam_MacroTO_Info.mat'
    % 'Cantilever_MacroTO_Info'
    NumSamples = size(Tractions,3);
else
    Tractions = GenerateTractions(NumSamples,[Folder '\' TractionFile],GenerateNewTractions);
    Center = [0;0]*ones(1,NumSamples);
    Scale = 1*ones(1,NumSamples);
    List = 1:NumSamples;
    Relative_Vol_Frac = 0.5*ones(1,NumSamples);
end

%% TO conditions
% General Info
ProbInfo.NGPpL = 2; % 2; Number integration points 1D
ProbInfo.NGPpS = 9; % Number integration points 2D. En elementos 3x3
ProbInfo.E = 1000; % Young's modulus
ProbInfo.nu = 0.3; % Poisson's ratio
%Relative_Vol_Frac = 0.5; % Quiero que este lleno el 50%
ProbInfo.Penal = 3; % recomendado en la bibliografía
ProbInfo.rmin = 1.5; % recomendado en la bibliografía. Igual a 1.5 lados de elemento.
ProbInfo.NumIt = 1000; % Como máximo
ProbInfo.ElemTO = true;
ProbInfo.SquareEls = true;
ProbInfo.Folder = Folder;
ProbInfo.Net = Net;

NumEls = length(MeshData.Surf.Elements);
Rel_Density = zeros(NumEls,NumSamples);
Stress = cell(NumSamples,1);
Densities = cell(NumSamples,1);
c = cell(NumSamples,1);
NumIts = zeros(NumSamples,1);
ItsFull = zeros(NumSamples,1);

%% Barra de progreso
q = parallel.pool.DataQueue;
pbar = waitbar(0, 'Iniciando Optimización Topológica...');
% Configurar la función que reacciona a los datos de la cola
afterEach(q, @(data) updateBar());

% Contador para el progreso
iterCount = 0;
function updateBar()
    iterCount = iterCount + 1;
    waitbar(iterCount / NumSamples, pbar, sprintf('Progreso: %d/%d', iterCount, NumSamples));
end

%% Generate topology
TStart = tic;
figure
MainFig = gcf;
hold on
for iSample = List
    if iSample == 1
        IniDentsity = Relative_Vol_Frac(iSample) * ones(size(Rel_Density(:,iSample)));
    else
        IniDentsity = Rel_Density(:,iSample-1);
    end
    [Rel_Density(:,iSample),Stress{iSample},Densities{iSample},NumIts(iSample),ItsFull(iSample),c{iSample},FEMc{iSample}] = GenTopology(MeshData,Tractions(:,:,iSample),Relative_Vol_Frac(iSample),ProbInfo,IniDentsity);
    % Plot final topology
     Top = MeshData.Surf.Topology; % Que nodos forma cada elemento
     XY = MeshData.XYZ; % coordenadas de los nodos

     if ProbInfo.ElemTO
         Value = repmat(Rel_Density(:,iSample)',size(Top,1),1); % repetir matrices
     else
         Value = reshape(Rel_Density(Top,iSample),size(Top,1),[]); % repetir matrices
     end
     Xval = reshape(XY(1,Top),size(Top,1),[])*Scale(iSample) + Center(1,iSample);
     Yval = reshape(XY(2,Top),size(Top,1),[])*Scale(iSample) + Center(2,iSample);
     %Value = repmat((dc-adc)',size(Top,1),1);

     % figure
     % figure(MainFig)
     % patch(Xval,Yval,-Value,'EdgeColor','none');
     % axis equal
     % caxis([-1,0]) %límites de colores
     % colormap gray
     
    % Al terminar, enviamos un mensaje a la cola para avisar
    send(q, iSample);
end
if ~exist(Folder,'dir')
    mkdir(Folder)
end
saveas(gcf,[Folder '\' FileName(1:end-4)])

close(pbar);
TEnd = toc(TStart);

if ~exist(Folder,'dir')
    mkdir(Folder)
end

save([Folder '\' FileName],'Rel_Density','Tractions','MeshData','Relative_Vol_Frac','Stress','Densities','TEnd','NumIts','ItsFull','c','FEMc')
%save([Folder '\' FileName],'Rel_Density','Tractions','MeshData','Relative_Vol_Frac')

end

% ================= Funciones auxiliares  ==================================
function updateProgress(~)
    % Esta función se llama cada vez que llega un mensaje a la cola
    persistent pbar % Usamos una variable persistente o una global
    if isempty(pbar)
        pbar = waitbar(0, 'Procesando celdas de optimización...');
    end
    
    % Necesitas conocer el total de iteraciones para calcular el porcentaje
    % Aquí un ejemplo simplificado; lo ideal es usar una variable externa
end
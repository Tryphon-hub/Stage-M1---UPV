function [Rel_Density,Stress,Densities,it,CounterFull,c,FEMc] = GenTopology(MeshData,Tractions,Relative_Vol_Frac,ProbInfo,IniDentsity)
% close all
tic
%profile on
% addpath '.\Funciones Optimizacion'
% addpath '.\Mallas'
% addpath '.\Otras funciones'

%% General Info
NGPpL = ProbInfo.NGPpL;  % 2; Number integration points 1D
NGPpS = ProbInfo.NGPpS; %9; % Number integration points 2D. En elementos 3x3
E = ProbInfo.E; %1000; % Young's modulus
nu = ProbInfo.nu; %0.0; % Poisson's ratio
%Relative_Vol_Frac = 0.4; % Quiero que este lleno el 50%
Penal = ProbInfo.Penal; %3; % recomendado en la bibliografía
Rmin = ProbInfo.rmin; %1.5; % recomendado en la bibliografía. Igual a 1.5 lados de elemento.
NumIt = ProbInfo.NumIt; %1000; % Como máximo
ElemTO = ProbInfo.ElemTO;
SquareEls = ProbInfo.SquareEls;

%% Material properties
D = DHooks2D(E,nu,'Plane Stress');

% Tracciones en cada lado
%Tractions = zeros(2,8); % Normal and tangential traction at vertex of the side
%Tractions = [  0   0   3   3  0  0  1  1;...
%              -1  -1   0   0  1  1  0  0];
% Tractions = [  0   0   1   1  0  0  1  1;...
%                0   0   0   0  0  0  0  0];
% %% Generate Mesh
% Mesh_File = 'Square.msh';
% [MeshData] = ReadGMSH(Mesh_File);
%Trucate single precission mesh
%MeshData.XYZ = round(MeshData.XYZ,10); % El problema que genera la malla es de simple precision

NumEls = length(MeshData.Surf.Elements);

Top = MeshData.Surf.Topology; % Que nodos forma cada elemento
XY = MeshData.XYZ; % coordenadas de los nodos
c = zeros(NumIt,1); % var que guarda la f objectivo
RelVariation = zeros(NumIt,1); % variation of the compliance


if ElemTO
    %Rel_Density(1:length(MeshData.Surf.Elements),1) = Relative_Vol_Frac;
    Rel_Density(1:length(MeshData.Surf.Elements),1) = IniDentsity;
else
    %Rel_Density(1:length(unique(MeshData.Surf.Topology)),1) = Relative_Vol_Frac;
    Rel_Density(1:length(unique(MeshData.Surf.Topology)),1) = IniDentsity;
end

%% Postprocesado
IntegrationType = 1;
dir = 1; % Escoger dirección para cálculo de tensión
%%
% figure(1)
% figure(2)
Rel_Density_old = zeros(size(Rel_Density,1),2);
LOld = zeros(NumEls,2);
UOld = zeros(NumEls,2);
AA = tic;

% Load Neural Network
if ~isempty(ProbInfo.Net)
    ImageSize = 32;
    Net = load([ProbInfo.Folder '/' ProbInfo.Net]);

    NormalTractions = Tractions(1,:);
    TangentialTractions = Tractions(2,:);

    TnSide = zeros(4,ImageSize);
    TnSide(1,:) = linspace(NormalTractions(1),NormalTractions(2),ImageSize);
    TnSide(2,:) = linspace(NormalTractions(3),NormalTractions(4),ImageSize);
    TnSide(3,:) = linspace(NormalTractions(5),NormalTractions(6),ImageSize);
    TnSide(4,:) = linspace(NormalTractions(7),NormalTractions(8),ImageSize);

    TtSide = zeros(4,ImageSize);
    TtSide(1,:) = linspace(TangentialTractions(1),TangentialTractions(2),ImageSize);
    TtSide(2,:) = linspace(TangentialTractions(3),TangentialTractions(4),ImageSize);
    TtSide(3,:) = linspace(TangentialTractions(5),TangentialTractions(6),ImageSize);
    TtSide(4,:) = linspace(TangentialTractions(7),TangentialTractions(8),ImageSize);

    ImageTn = zeros(ImageSize,ImageSize);

    ImageTn(ImageSize,:) = TnSide(1,:);
    ImageTn(:,end) = flip(TnSide(2,:));
    ImageTn(1,:) = flip(TnSide(3,:));
    ImageTn(:,1) = TnSide(4,:);

    ImageTt = zeros(ImageSize,ImageSize);
    ImageTt(ImageSize,:) = TtSide(1,:);
    ImageTt(:,end) = flip(TtSide(2,:));
    ImageTt(1,:) = flip(TtSide(3,:));
    ImageTt(:,1) = TtSide(4,:);

    inputData = zeros(ImageSize,ImageSize,3);

    % Normal Tractions
    inputData(:,:,2) = ImageTn;

    % Tangential Tractions
    inputData(:,:,3) = ImageTt;

end

Stress = zeros(NumEls,6,NumIt);
Densities = zeros(NumEls,NumIt);
KeepFull = 0;
TrueCut = 0;
CounterFull = 0;
FinalIt = NumIt;

for it = 1:NumIt
    Densities(:,it) = Rel_Density;
    if isempty(ProbInfo.Net) || ~mod(it,1500) || it==1 ||KeepFull
        [Sol] = SolveFE(MeshData,Rel_Density.^Penal,NGPpS,NGPpL,D,Tractions,SquareEls,ElemTO);

        % Evaluate Finite Element Stress
        [Stress(:,:,it), VMStress] = EvalStress(Top,XY,Rel_Density.^1,D,Sol,2,SquareEls,ElemTO,'Plane Stress',E,nu);
        StressFE = Stress;
        CounterFull = CounterFull + 1;
        disp('Full Orden Model evaluation')
    else

        inputData(:,:,1) = reshape(Rel_Density',ImageSize,ImageSize);
        imagenPredicha = predict(Net.net, inputData);
        Stress(:,1,it) = reshape(imagenPredicha(:,:,1),[],1)'; %./(Rel_Density.^Penal)';
        Stress(:,2,it) = reshape(imagenPredicha(:,:,2),[],1)'; %./(Rel_Density.^Penal)';
        Stress(:,4,it) = reshape(imagenPredicha(:,:,3),[],1)'; %./(Rel_Density.^Penal)';

        [Sol] = SolveFE(MeshData,Rel_Density.^Penal,NGPpS,NGPpL,D,Tractions,SquareEls,ElemTO);

        % Evaluate Finite Element Stress
        [StressFE(:,:,it), VMStress] = EvalStress(Top,XY,Rel_Density.^1,D,Sol,2,SquareEls,ElemTO,'Plane Stress',E,nu);
    end

    % % Visualizar los 3 canales de Entrada (Tracciones y Densidad)
    % subplot(3,3,1); imshow(inputData(:,:,1), []); title('Input: \rho');
    % colorbar
    % clim([0,1])
    % subplot(3,3,2); imshow(inputData(:,:,2), []); title('Input: t_n');
    % colorbar
    % clim([-1,1])
    % subplot(3,3,3); imshow(inputData(:,:,3), []); title('Input: t_t');
    % colorbar
    % clim([-1,1])
    % 
    % % Visualizar los 3 canales de Salida Predicha (Tensiones)
    % subplot(3,3,4); imshow(reshape(Stress(:,1,it),ImageSize,ImageSize), []); title('Pred: \sigma_x');
    % colorbar
    % %clim([-1,1])
    % subplot(3,3,5); imshow(reshape(Stress(:,2,it),ImageSize,ImageSize), []); title('Pred: \sigma_y');
    % colorbar
    % %clim([-1,1])
    % subplot(3,3,6); imshow(reshape(Stress(:,4,it),ImageSize,ImageSize), []); title('Pred: \tau_{xy}');
    % colorbar
    % %clim([-1,1])
    % 
    % % Visualizar los 3 canales de Salida Real (Tensiones)
    % subplot(3,3,7); imshow(reshape(StressFE(:,1,it),ImageSize,ImageSize), []); title('Real: \sigma_x');
    % colorbar
    % %clim([-1,1])
    % subplot(3,3,8); imshow(reshape(StressFE(:,2,it),ImageSize,ImageSize), []); title('Real: \sigma_y');
    % colorbar
    % %clim([-1,1])
    % subplot(3,3,9); imshow(reshape(StressFE(:,4,it),ImageSize,ImageSize), []); title('Real: \tau_{xy}');
    % colorbar
    % %clim([-1,1])
    % 
    % pause(0.1)

    % Objective function and sensibilidty evaluation
    %[c(it),dc,ce,InfVol] = Opt(Rel_Density,Sol,D,Penal,MeshData,ElemTO,1,2,4,4);

    [c(it),dc,ce,InfVol] = Opt_Stress(Rel_Density,Stress(:,:,it),D,Penal,MeshData,ElemTO,SquareEls,2,4);
    Vol = sum(InfVol);
    %[c2,dc2,ce2,~] = Opt(Rel_Density,Sol,D,Penal,MeshData,ElemTO,1,2,4,4);



    %[FEMc(it),~,~,~] = Opt_Stress(Rel_Density,StressFE(:,:,it),D,Penal,MeshData,ElemTO,SquareEls,2,4);
    
    % Stress(:,1,it) = Stress(:,1,it) .*(Rel_Density.^Penal);
    % Stress(:,2,it) = Stress(:,2,it) .*(Rel_Density.^Penal);
    % Stress(:,4,it) = Stress(:,4,it) .*(Rel_Density.^Penal);

    % Filtering radius
    CharacteristicSize = nthroot(Vol/NumEls,2); %%% Se podría jugar!!!
    Rmin_aux = Rmin*mean(CharacteristicSize); % Más o menos de todos los elementos

    % Sensibility filtering
    dc = Filter(Rel_Density,dc,Rmin_aux,MeshData,ElemTO,2); % de la dc. Promedio de lo de alrededor
    InfVol = Filter(ones(size(Rel_Density)),InfVol,Rmin_aux,MeshData,ElemTO,2); % de del volumen de influencia. Hay que aplicar el mismo filtrado que a dc
    InfVol = InfVol*Vol/sum(InfVol);

    % Optimality criteria
    New_Rel_Density = OC(Rel_Density,dc,InfVol,Relative_Vol_Frac);

    change=max(abs(New_Rel_Density-Rel_Density));
    disp(['Iteration: ' num2str(it) '; c:' num2str(c(it)) '; cvar:' num2str(RelVariation(it)) '; Change:' num2str(change) '; VolFrac:' num2str(sum(Rel_Density.*InfVol)/sum(InfVol)) '; itTime:' num2str(toc) 's']);

    %  if ElemTO
    %      Value = repmat(New_Rel_Density',size(Top,1),1); % repetir matrices
    %  else
    %      Value = New_Rel_Density(Top); % repetir matrices
    % 
    %  end
    %  Xval = reshape(XY(1,Top),size(Top,1),[]);
    %  Yval = reshape(XY(2,Top),size(Top,1),[]);
    % 
    % figure(1)
    % patch(Xval,Yval,-Value);
    % axis equal
    % caxis([-1,0]) %límites de colores
    % colormap gray
    % title(['Iteration: ' num2str(it)])%,'Dirección',num2str(dir));
    % pause(0.1)

     if it == FinalIt
         break
     end

    if change < 0.05 && it>2 %% OJO que se puede cambiar. 130 iteraciomes para 0.05. Más para 0.01
        KeepFull = 1;
        FinalIt = it + 1;
        if TrueCut
            break
        end
        TrueCut = 1;      
    end
    
    if it>15
        meanCompliance = mean(rmoutliers(c(it-4:it)));
        Variation = abs(c(it)-c(it-1));
        RelVariation(it+1) = Variation/meanCompliance;
        if RelVariation(it+1)<0.001
            KeepFull = 1;
            FinalIt = it + 1;
            if TrueCut
                break
            end
            TrueCut = 1;
        end
    end


    Rel_Density = New_Rel_Density;

end

% figure 
% plot(c)

Stress = Stress(:,:,1:it);
Densities = Densities(:,1:it);
c = c(1:it);
%FEMc = FEMc(1:it);
FEMc = zeros(1,it);



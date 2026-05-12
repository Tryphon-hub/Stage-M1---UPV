clear all

%% Carga de datos
Folder = 'Training_002';
load([Folder '\TrainingData.mat'])

%% Prepare data
Tractions_normal = Tractions(1,:,:);
Tractions_tangential = Tractions(2,:,:);

TractionsImage(:,:,1,:) = reshape(Tractions_normal(1,[1 2 5 6],:),2,2,1,[]);
TractionsImage(:,:,2,:) = reshape(Tractions_normal(1,[3 4 7 8],:),2,2,1,[]);

TractionsImage(:,:,3,:) = reshape(Tractions_tangential(1,[1 2 5 6],:),2,2,1,[]);
TractionsImage(:,:,4,:) = reshape(Tractions_tangential(1,[3 4 7 8],:),2,2,1,[]);

%% Entrenamiento de la red tipo U-NET
imageSize = [8 8 4];
numClasses = 5;
encoderDepth = 3;

Layers = unetLayers(imageSize,numClasses,'EncoderDepth',encoderDepth);

TractionsImage = imresize3(TractionsImage,[8 8 4]);

options = trainingOptions("sgdm", ...
    LearnRateSchedule="piecewise", ...
    LearnRateDropFactor=0.02, ...
    LearnRateDropPeriod=5, ...
    MaxEpochs=200, ...
    MiniBatchSize=64, ...
    Plots="training-progress");

features = reshape(Tractions,16,size(Tractions,3))';
responses = Rel_Density';

net = trainNetwork(features,responses,Layers,options);

CalTop = predict(net,features(1,:));

Top = MeshData.Surf.Topology; % Que nodos forma cada elemento
XY = MeshData.XYZ; % coordenadas de los nodos

Xval = reshape(XY(1,Top),size(Top,1),[]);
Yval = reshape(XY(2,Top),size(Top,1),[]);
%Value = repmat(CalTop,size(Top,1),1);
Value = repmat(responses(1,:),size(Top,1),1);
figure(1)
patch(Xval,Yval,-Value);
axis equal
caxis([-1,0]) %límites de colores
colormap gray



% 
% LayerInput = image3dInputLayer(inputSize,Name,Value)([2 2 4 1],'Name','Entrada');
% Conv_01 = convolution3dLayer(2,20,'Name','Conv_01');
% MxPo_01 = maxPooling3dLayer(2,'Stride',2);
% 
% 
% Norm_01 = batchNormalizationLayer;
% Relu_01 = reluLayer;
% 
% 
% 
% 
% layer = geluLayer
% layer = clippedReluLayer(ceiling)


% Layer_01 = fullyConnectedLayer(256,'Name','Capa_01');
% Layer_02 = fullyConnectedLayer(1024,'Name','Capa_02');
% Layer_03_0 = fullyConnectedLayer(4096,'Name','Capa_03_0');
% Layer_03_1 = fullyConnectedLayer(7200,'Name','Capa_03_1');
% Layer_03_2 = fullyConnectedLayer(16000,'Name','Capa_03_2');
% Layer_03_3 = fullyConnectedLayer(16000,'Name','Capa_03_3');
% Layer_03_4 = fullyConnectedLayer(16000,'Name','Capa_03_4');
% Layer_03_5 = fullyConnectedLayer(10000,'Name','Capa_03_5');
% Layer_03_6 = fullyConnectedLayer(6000,'Name','Capa_03_6');
% Layer_04 = fullyConnectedLayer(2704,'Name','Capa_04');
% Layer_RELU = reluLayer('Name','Capa_RELU');
% Layer_06 = sigmoidLayer('Name','Capa_07');


% Layer_Out = regressionLayer('Name','Output');
% 
% layers = [LayerInput Layer_01 Layer_02 Layer_03_0 Layer_03_1 Layer_03_2 Layer_03_3 Layer_03_4 Layer_03_5 Layer_03_6...
%     Layer_04 Layer_06 Layer_Out];




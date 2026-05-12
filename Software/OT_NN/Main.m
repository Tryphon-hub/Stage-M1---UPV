function Main
% Generar imágenes a partir de los datos

DataImage = load('TrainingData_Stress.mat');

numSamples = size(DataImage.Tractions,3);
ImageSize = sqrt(size(DataImage.Rel_Density,1));

X = squeeze(reshape(DataImage.Tractions,16,1,numSamples))';
Y = DataImage.Rel_Density';

numInputs = size(X,2);
numOutputs = size(Y,2);

% Si tienes tus imágenes en una matriz plana de 1000x2704, cámbiales la forma:
Y = reshape(Y', [ImageSize, ImageSize, 1, numSamples]); 

% Dividir datos (usando el método anterior)
idxTrain = 1:floor(0.7*numSamples);
idxValidation = floor(0.7*numSamples)+1:floor(0.9*numSamples);
idxVerification = floor(0.9*numSamples)+1:numSamples;

XTrain = X(idxTrain, :);
YTrain = Y(:, :, :, idxTrain);


XValidation = X(idxValidation, :);
YValidation = Y(:, :, :, idxValidation);

XVerification = X(idxVerification, :);
YVerification= Y(:, :, :, idxVerification);


% 1. Convertir X a formato 'CB' (Canales, Batch)
% MATLAB espera que las muestras estén en la SEGUNDA dimensión para vectores
XT = dlarray(single(XTrain'), 'CB'); 
XV = dlarray(single(XValidation'), 'CB');

% 2. Convertir Y a formato 'SSCB' (Spatial, Spatial, Channel, Batch)
% Si tus imágenes son [52 52 1 1000]
YT = dlarray(single(YTrain), 'SSCB');
YV = dlarray(single(YValidation), 'SSCB');

% Parámetros de diseño
inputDim = numInputs;           % Tus 16 parámetros de entrada
imgSize = [ImageSize ImageSize 1];     % Tamaño objetivo [H, W, C]
baseSize = [8 8];        % Tamaño del "cuello de botella" inicial
numFilters = 64;

layers = [
    featureInputLayer(inputDim, 'Name', 'input')
    
    % Capa densa que proyecta a la base espacial (8x8x64 = 4096 neuronas)
    fullyConnectedLayer(prod(baseSize) * numFilters, 'Name', 'fc_expand')
    
    % Reshape a formato espacial inicial
    reshapeLayer([baseSize numFilters],OperationDimension="spatial-channel") 
    
    % Primera expansión: 8x8 -> 16x16
    transposedConv2dLayer(4, numFilters/2, 'Stride', 2, 'Cropping', 'same', 'Name', 'up1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    
    % Segunda expansión: 16x16 -> 32x32
    transposedConv2dLayer(4, numFilters/4, 'Stride', 2, 'Cropping', 'same', 'Name', 'up2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    
    % Capa final para colapsar canales a 1 (Gris) y asegurar rango [0,1]
    convolution2dLayer(3, imgSize(3), 'Padding', 'same', 'Name', 'final_conv')
    sigmoidLayer('Name', 'output')
];

% 3. Entrenar
options = trainingOptions('adam', ...
    'MaxEpochs', 150, ...
    'ValidationData', {XV, YV}, ...
    'Plots', 'training-progress', ...
    'Verbose', false);

net = trainnet(XT, YT, layers, "mse", options);

for idx = 1:10

    inputData = XVerification(idx, :);
    imagenReal = YVerification(:, :, :, idx);
    % Predecir sobre el conjunto de validación
    imagenPredicha = minibatchpredict(net, inputData);

    subplot(10,2,2*idx-1), imshow(imagenReal), title('Real')
    subplot(10,2,2*idx), imshow(imagenPredicha), title('Predicha')
end

Error = zeros(length(idxVerification),1);
for idx = 1:length(idxVerification)
     inputData = XVerification(idx, :);
    imagenReal = YVerification(:, :, :, idx);
    % Predecir sobre el conjunto de validación
    imagenPredicha = minibatchpredict(net, inputData);
    Error(idx) = norm(imagenReal-imagenPredicha)/ImageSize;
end

ErrorMSEVeri = norm(Error)/sqrt(length(idxVerification))



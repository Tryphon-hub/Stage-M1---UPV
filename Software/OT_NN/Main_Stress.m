function Main_Stress(TrainAgain,FileInput,FileNet)
% Generar imágenes a partir de los datos

DataImage = load(FileInput);

NumSets = size(DataImage.Tractions,3);
ImageSize = sqrt(size(DataImage.Rel_Density,1));

Counter = 0;
for iSet = 1:NumSets
    SamplesPSet = size(DataImage.Densities{iSet},2);
    NormalTractions = DataImage.Tractions(1,:,iSet);
    TangentialTractions = DataImage.Tractions(2,:,iSet);

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


    for iSample = 1:SamplesPSet
        Counter = Counter + 1;
            % Input
            % Densities
            X(:,:,1,Counter) = reshape(DataImage.Densities{iSet}(:,iSample)',ImageSize,ImageSize);

            % Normal Tractions
            X(:,:,2,Counter) = ImageTn;

            % Tangential Tractions
            X(:,:,3,Counter) = ImageTt;

            % Output
            % Sigma x
            Y(:,:,1,Counter) = reshape(DataImage.Stress{iSet}(:,1,iSample)',ImageSize,ImageSize);

            % Sigma y
            Y(:,:,2,Counter) = reshape(DataImage.Stress{iSet}(:,2,iSample)',ImageSize,ImageSize);

            % Tau xy
            Y(:,:,3,Counter) = reshape(DataImage.Stress{iSet}(:,4,iSample)',ImageSize,ImageSize);
    end
end

numSamples = Counter;

% Dividir datos (usando el método anterior)
SizeTraining = round(0.8*numSamples);
idxTrain = randperm(numSamples,SizeTraining);
Restidx = setdiff(1:numSamples,idxTrain);
auxidxValidation = randperm(length(Restidx),round(0.6*length(Restidx)));
idxValidation = Restidx(auxidxValidation);
idxVerification = setdiff(1:numSamples,[idxTrain idxValidation]);

XTrain = X(:, :, :, idxTrain);
YTrain = Y(:, :, :, idxTrain);

XValidation = X(:, :, :, idxValidation);
YValidation = Y(:, :, :, idxValidation);

XVerification = X(:, :, :, idxVerification);
YVerification = Y(:, :, :, idxVerification);




% Configuración de la arquitectura de la red
npx = ImageSize; % Dimensión espacial (32x32) [cite: 209]
n_filters = 15; % Filtros base (mejor resultado según el estudio) [cite: 208, 341]
dropout_prob = 0.3;  % Probabilidad de dropout para bloques profundosv
numInputChannels = 3; % tx, ty, rho [cite: 249, 282]
numOutputChannels = 3; % sigma_x, sigma_y, tau_xy [cite: 186, 286]

lgraph = layerGraph();

% --- BLOQUE DE ENTRADA ---
tempLayers = [
    imageInputLayer([npx npx numInputChannels], 'Name', 'input', 'Normalization', 'none') % [cite: 204, 209]
    convolution2dLayer(3, n_filters, 'Padding', 'same', 'Name', 'enc0_c1') % [cite: 207, 208]
    reluLayer('Name', 'enc0_r1') % [cite: 236]
    convolution2dLayer(3, n_filters, 'Padding', 'same', 'Name', 'enc0_c2')
    reluLayer('Name', 'enc0_r2')
    convolution2dLayer(3, n_filters, 'Padding', 'same', 'Name', 'enc0_c3')
    reluLayer('Name', 'enc0_r3')];
lgraph = addLayers(lgraph, tempLayers);

% --- ENCODER NIVEL 1 (16x16) ---
tempLayers = [
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1') % [cite: 198, 209]
    convolution2dLayer(3, 2*n_filters, 'Padding', 'same', 'Name', 'enc1_c1') % [cite: 208]
    reluLayer('Name', 'enc1_r1')
    convolution2dLayer(3, 2*n_filters, 'Padding', 'same', 'Name', 'enc1_c2')
    reluLayer('Name', 'enc1_r2')
    convolution2dLayer(3, 2*n_filters, 'Padding', 'same', 'Name', 'enc1_c3')
    reluLayer('Name', 'enc1_r3')];
lgraph = addLayers(lgraph, tempLayers);

% --- ENCODER NIVEL 2 (8x8) ---
tempLayers = [
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')
    convolution2dLayer(3, 4*n_filters, 'Padding', 'same', 'Name', 'enc2_c1')
    reluLayer('Name', 'enc2_r1')
    convolution2dLayer(3, 4*n_filters, 'Padding', 'same', 'Name', 'enc2_c2')
    reluLayer('Name', 'enc2_r2')
    convolution2dLayer(3, 4*n_filters, 'Padding', 'same', 'Name', 'enc2_c3')
    reluLayer('Name', 'enc2_r3')];
lgraph = addLayers(lgraph, tempLayers);

% --- BLOQUE NIVEL 3 (4x4) ---
tempLayers = [
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool3')
    convolution2dLayer(3, 8*n_filters, 'Padding', 'same', 'Name', 'enc3_c1')
    reluLayer('Name', 'enc3_r1')
    convolution2dLayer(3, 8*n_filters, 'Padding', 'same', 'Name', 'enc3_c2')
    reluLayer('Name', 'enc3_r2')
    convolution2dLayer(3, 8*n_filters, 'Padding', 'same', 'Name', 'enc3_c3')
    reluLayer('Name', 'enc3_r3')];
lgraph = addLayers(lgraph, tempLayers);

% --- BLOQUE CENTRAL NIVEL 4 (2X2) ---
tempLayers = [
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool4')
    convolution2dLayer(3, 16*n_filters, 'Padding', 'same', 'Name', 'btl_c1')
    reluLayer('Name', 'btl_r1')
    convolution2dLayer(3, 16*n_filters, 'Padding', 'same', 'Name', 'btl_c2')
    reluLayer('Name', 'btl_r2')
    convolution2dLayer(3, 16*n_filters, 'Padding', 'same', 'Name', 'btl_c3')
    reluLayer('Name', 'btl_r3')
    dropoutLayer(dropout_prob, 'Name', 'btl_drop')];
lgraph = addLayers(lgraph, tempLayers);

% --- DECODER NIVEL 3 (4x4) ---
tempLayers = [
    transposedConv2dLayer(4, 8*n_filters, 'Stride', 2, 'Cropping', 'same', 'Name', 'up3') % [cite: 198, 210]
    concatenationLayer(3, 2, 'Name', 'concat3') % [cite: 195, 200]
    convolution2dLayer(3, 8*n_filters, 'Padding', 'same', 'Name', 'dec3_c1')
    reluLayer('Name', 'dec3_r1')
    convolution2dLayer(3, 8*n_filters, 'Padding', 'same', 'Name', 'dec3_c2')
    reluLayer('Name', 'dec3_r2')
    convolution2dLayer(3, 8*n_filters, 'Padding', 'same', 'Name', 'dec3_c3')
    reluLayer('Name', 'dec3_r3')];
lgraph = addLayers(lgraph, tempLayers);



% --- DECODER NIVEL 2 (8x8) ---
tempLayers = [
    transposedConv2dLayer(4, 4*n_filters, 'Stride', 2, 'Cropping', 'same', 'Name', 'up2') % [cite: 198, 210]
    concatenationLayer(3, 2, 'Name', 'concat2') % [cite: 195, 200]
    convolution2dLayer(3, 4*n_filters, 'Padding', 'same', 'Name', 'dec2_c1')
    reluLayer('Name', 'dec2_r1')
    convolution2dLayer(3, 4*n_filters, 'Padding', 'same', 'Name', 'dec2_c2')
    reluLayer('Name', 'dec2_r2')
    convolution2dLayer(3, 4*n_filters, 'Padding', 'same', 'Name', 'dec2_c3')
    reluLayer('Name', 'dec2_r3')];
lgraph = addLayers(lgraph, tempLayers);

% --- DECODER NIVEL 1 (16x16) ---
tempLayers = [
    transposedConv2dLayer(4, 2*n_filters, 'Stride', 2, 'Cropping', 'same', 'Name', 'up1')
    concatenationLayer(3, 2, 'Name', 'concat1')
    convolution2dLayer(3, 2*n_filters, 'Padding', 'same', 'Name', 'dec1_c1')
    reluLayer('Name', 'dec1_r1')
    convolution2dLayer(3, 2*n_filters, 'Padding', 'same', 'Name', 'dec1_c2')
    reluLayer('Name', 'dec1_r2')
    convolution2dLayer(3, 2*n_filters, 'Padding', 'same', 'Name', 'dec1_c3')
    reluLayer('Name', 'dec1_r3')];
lgraph = addLayers(lgraph, tempLayers);

% --- DECODER NIVEL 0 / SALIDA (32x32) ---
tempLayers = [
    transposedConv2dLayer(4, n_filters, 'Stride', 2, 'Cropping', 'same', 'Name', 'up0')
    concatenationLayer(3, 2, 'Name', 'concat0')
    convolution2dLayer(3, n_filters, 'Padding', 'same', 'Name', 'dec0_c1')
    reluLayer('Name', 'dec0_r1')
    convolution2dLayer(3, n_filters, 'Padding', 'same', 'Name', 'dec0_c2')
    reluLayer('Name', 'dec0_r2')
    convolution2dLayer(3, n_filters, 'Padding', 'same', 'Name', 'dec0_c3')
    reluLayer('Name', 'dec0_r3')
    convolution2dLayer(1, numOutputChannels, 'Padding', 'same', 'Name', 'final_conv') % 
    % Nota: Usamos una activación lineal para el campo de tensiones 
];
lgraph = addLayers(lgraph, tempLayers);

% --- CONEXIONES DE SALTO (Skip Connections) ---
% Conectamos encoder con decoder en los mismos niveles dimensionales 
lgraph = connectLayers(lgraph, 'enc0_r3', 'concat0/in2');
lgraph = connectLayers(lgraph, 'enc1_r3', 'concat1/in2');
lgraph = connectLayers(lgraph, 'enc2_r3', 'concat2/in2');
lgraph = connectLayers(lgraph, 'enc3_r3', 'concat3/in2');

% Conexiones secuenciales entre bloques principales
lgraph = connectLayers(lgraph, 'enc0_r3', 'pool1');
lgraph = connectLayers(lgraph, 'enc1_r3', 'pool2');
lgraph = connectLayers(lgraph, 'enc2_r3', 'pool3');
lgraph = connectLayers(lgraph, 'enc3_r3', 'pool4');
lgraph = connectLayers(lgraph, 'btl_drop', 'up3');
lgraph = connectLayers(lgraph, 'dec3_r3', 'up2');
lgraph = connectLayers(lgraph, 'dec2_r3', 'up1');
lgraph = connectLayers(lgraph, 'dec1_r3', 'up0');




% Conversión del grapho de la red a la propia red
net = dlnetwork(lgraph);

% Parámetros extraídos del artículo (Secciones 2.3 y 3.1)
% Tasa de aprendizaje de 0.001 [cite: 242]
% Beta1 = 0.9, Beta2 = 0.999, Epsilon = 1e-7 [cite: 242]
% Batch size de 30 muestras [cite: 244]
% 30,000 pases (iteraciones totales estimadas) [cite: 244]

% options = trainingOptions('adam', ...
%     'InitialLearnRate', 0.001, ...                     % [cite: 242]
%     'GradientDecayFactor', 0.09, ...                    % Equivale a beta1 [cite: 242]
%     'SquaredGradientDecayFactor', 0.999, ...           % Equivale a beta2 [cite: 242]
%     'Epsilon', 1e-7, ...                               % [cite: 242]
%     'MiniBatchSize', 30, ...                           % [cite: 244]
%     'MaxEpochs', 50, ...                               % Ajustar según el tamaño del dataset (ej: 20,000 muestras / 30 batch ~ 666 iter/epoch. 30,000 pases / 666 ~ 45-50 epochs) [cite: 244, 276]
%     'Shuffle', 'every-epoch', ...                      % Recomendado para evitar sesgos en el orden 
%     'ValidationData', {XV, YV}, ...              % El estudio usa un 10% para validación 
%     'ValidationFrequency', 100, ...                    % Frecuencia de chequeo de la pérdida SMAPE
%     'Plots', 'training-progress', ...                  % Para visualizar la convergencia en tiempo real [cite: 218]
%     'Verbose', false);



% Entrenar usando la función personalizada
if ~exist(FileNet,'file') || TrainAgain
    
    % Configuración
    numEpochs = 500;
    switchEpoch = 3; % Cambia a SMAPE en la época 30
    learningRate = 1e-4;
    OnTraining = 1;
    executionEnvironment = 'gpu'; % Recomendado para UNET
        % Inicializar variables para el optimizador Adam
    trailingAvg = [];
    trailingAvgSq = [];

    % Datos de validación
    % Crear el Datastore (con los datos en formato single/double)
    dsX = arrayDatastore(XValidation, 'IterationDimension', 4);
    dsY = arrayDatastore(YValidation, 'IterationDimension', 4);
    dsVal = combine(dsX, dsY);

    % Configurar la cola de validación
    % El minibatchqueue se encargará de volver a ponerles el formato 'SSCB'
    mbqVal = minibatchqueue(dsVal, ...
        'MiniBatchSize', 64, ...
        'MiniBatchFormat', {'SSCB', 'SSCB'}, ...
        'OutputEnvironment', executionEnvironment);

    % Datos para el entrenamiento
    % Crear el Datastore (con los datos en formato single/double)
    dsX_T = arrayDatastore(XTrain, 'IterationDimension', 4);
    dsY_T = arrayDatastore(YTrain, 'IterationDimension', 4);
    dsTrain = combine(dsX_T, dsY_T);

    mbq = minibatchqueue(dsTrain, ...
        'MiniBatchSize', 64, ...
        'MiniBatchFormat', {'SSCB', 'SSCB'}, ...
        'OutputEnvironment', executionEnvironment);

    % Crear el monitor
    monitor = trainingProgressMonitor( ...
        Metrics=["TrainingLoss","ValidationLoss"], ...
        Info=["Epoch","Iteration"], ...
        XLabel="Iteration");

    % Agrupar ambas pérdidas en la misma subgráfica
    groupSubPlot(monitor, "Loss", ["TrainingLoss", "ValidationLoss"]);

    % Escala logarítmica si los valores de SMAPE varían mucho
    yscale(monitor, "Loss", "log");

    % Precompilación de funciones de pérdida
    accelGrads = dlaccelerate(@modelGradients);

    iteration = 0;
    for epoch = 1:numEpochs
        % Decidir si usamos SMAPE según la época (ej. a partir de la 20)
        useSMAPE = epoch >= switchEpoch;

        shuffle(mbq);

        % Bucle de mini-batches
        while hasdata(mbq)
            iteration = iteration + 1;
            [X, T] = next(mbq);

            % Evaluar gradientes
            [loss, gradients] = dlfeval(accelGrads, net, X, T, useSMAPE, OnTraining);

            % Actualizar pesos (ejemplo con Adam)
            [net, trailingAvg, trailingAvgSq] = adamupdate(net, gradients, ...
                trailingAvg, trailingAvgSq, iteration, learningRate);

            % Actualizar el monitor
            % Registrar pérdida de entrenamiento en cada iteración
            recordMetrics(monitor, iteration, TrainingLoss=double(loss));

            % Actualizar info de cabecera
            updateInfo(monitor, Epoch=string(epoch)+" de "+numEpochs, ...
                Iteration=string(iteration));

            % Registrar pérdida de VALIDACIÓN cada cierto tiempo (ej. cada 100 iteraciones)
            if mod(iteration, 100) == 0
                % Aquí debes llamar a una función que pase el set de validación por la red
                lossVal = calculateValidationLoss(net, mbqVal, useSMAPE);
                recordMetrics(monitor, iteration, ValidationLoss=double(lossVal));
            end

            % Comprobar si has pulsado el botón "Stop" en la ventana
            if monitor.Stop
                return;
            end

        end
    end


    %net = trainnet(XT, YT, net_to_train, @(Y, T) smapeLoss(Y, T), options);
    %net = trainnet(XT, YT, net_to_train, 'mse', options);
    save(FileNet,'net')
else
    load(FileNet)
end

ChoosenSamples = randperm(length(idxVerification),10);
for idx = ChoosenSamples

    inputData = XVerification(:,:,:,idx);
    imagenReal = YVerification(:,:,:,idx);
    % Predecir sobre el conjunto de validación
    imagenPredicha = predict(net, inputData);

    figure('Name',[ 'Evaluación del Modelo Subrogado en Verification ' num2str(idxVerification(idx))]);

    % Visualizar los 3 canales de Entrada (Tracciones y Densidad)
    subplot(3,3,1); imshow(inputData(:,:,1), []); title('Input: \rho');
    colorbar
    clim([0,1])
    subplot(3,3,2); imshow(inputData(:,:,2), []); title('Input: t_n');
    colorbar
    clim([-1,1])
    subplot(3,3,3); imshow(inputData(:,:,3), []); title('Input: t_t');
    colorbar
    clim([-1,1])

    % Visualizar los 3 canales de Salida Predicha (Tensiones)
    subplot(3,3,4); imshow(imagenPredicha(:,:,1), []); title('Pred: \sigma_x');
    colorbar
    %clim([-1,1])
    subplot(3,3,5); imshow(imagenPredicha(:,:,2), []); title('Pred: \sigma_y');
    colorbar
    %clim([-1,1])
    subplot(3,3,6); imshow(imagenPredicha(:,:,3), []); title('Pred: \tau_{xy}');
    colorbar
    %clim([-1,1])

    % Visualizar los 3 canales de Salida Real (Tensiones)
    subplot(3,3,7); imshow(imagenReal(:,:,1), []); title('Real: \sigma_x');
    colorbar
    %clim([-1,1])
    subplot(3,3,8); imshow(imagenReal(:,:,2), []); title('Real: \sigma_y');
    colorbar
    %clim([-1,1])
    subplot(3,3,9); imshow(imagenReal(:,:,3), []); title('Real: \tau_{xy}');
    colorbar
    %clim([-1,1])

    colormap jet; % El artículo usa mapas de calor para resaltar gradientes de tensión
end


% ChoosenSamples = randperm(length(idxTrain),10);
% for idx = ChoosenSamples
% 
%     inputData = XTrain(:,:,:,idx);
%     imagenReal = YTrain(:,:,:,idx);
%     % Predecir sobre el conjunto de validación
%     imagenPredicha = predict(net, inputData);
% 
%     figure('Name',[ 'Evaluación del Modelo Subrogado en Training ' num2str(idxTrain(idx))]);
% 
%     % Visualizar los 3 canales de Entrada (Tracciones y Densidad)
%     subplot(3,3,1); imshow(inputData(:,:,1), []); title('Input: \rho');
%     colorbar
%     clim([0,1])
%     subplot(3,3,2); imshow(inputData(:,:,2), []); title('Input: t_n');
%     colorbar
%     clim([-1,1])
%     subplot(3,3,3); imshow(inputData(:,:,3), []); title('Input: t_t');
%     colorbar
%     clim([-1,1])
% 
%     % Visualizar los 3 canales de Salida Predicha (Tensiones)
%     subplot(3,3,4); imshow(imagenPredicha(:,:,1), []); title('Pred: \sigma_x');
%     colorbar
%     clim([-1,1])
%     subplot(3,3,5); imshow(imagenPredicha(:,:,2), []); title('Pred: \sigma_y');
%     colorbar
%     clim([-1,1])
%     subplot(3,3,6); imshow(imagenPredicha(:,:,3), []); title('Pred: \tau_{xy}');
%     colorbar
%     clim([-1,1])
% 
%     % Visualizar los 3 canales de Salida Real (Tensiones)
%     subplot(3,3,7); imshow(imagenReal(:,:,1), []); title('Real: \sigma_x');
%     colorbar
%     clim([-1,1])
%     subplot(3,3,8); imshow(imagenReal(:,:,2), []); title('Real: \sigma_y');
%     colorbar
%     clim([-1,1])
%     subplot(3,3,9); imshow(imagenReal(:,:,3), []); title('Real: \tau_{xy}');
%     colorbar
%     clim([-1,1])
% 
%     colormap jet; % El artículo usa mapas de calor para resaltar gradientes de tensión
% end


Error_xx = zeros(length(idxVerification),1);
Error_yy = zeros(length(idxVerification),1);
Error_xy = zeros(length(idxVerification),1);
LossVerifu = zeros(length(idxVerification),1);

for idx = 1:length(idxVerification)
    inputData = XVerification(:,:,:,idx);
    imagenReal = YVerification(:, :, :, idx);
    % Predecir sobre el conjunto de validación

    imagenPredicha = predict(net, inputData);

    Error_xx(idx) = norm(imagenReal(:,:,1)-imagenPredicha(:,:,1))/ImageSize;
    Error_yy(idx) = norm(imagenReal(:,:,2)-imagenPredicha(:,:,2))/ImageSize;
    Error_xy(idx) = norm(imagenReal(:,:,3)-imagenPredicha(:,:,3))/ImageSize;

    LossVerifu(idx) = modelGradients(net, inputData, imagenReal, 1,0);
end

ErrorMSEVeri_xx = norm(Error_xx)/sqrt(length(idxVerification))
ErrorMSEVeri_yy = norm(Error_yy)/sqrt(length(idxVerification))
ErrorMSEVeri_xy = norm(Error_xy)/sqrt(length(idxVerification))
ErrorLoss = norm(LossVerifu)/sqrt(length(idxVerification))

end
    
% =================== AUXILIAR FUNCTIONS =======================================
% function loss = smapeLoss(YPred, YTarget)
%     % Evitar división por cero añadiendo un valor muy pequeño (epsilon)
%     epsilon = 1e-1;
% 
%     % Calcular el numerador: |Predicho - Real|
%     numerator = abs(YPred - YTarget);
% 
%     % Calcular el denominador: (|Real| + |Predicho|) / 2
%     denominator = (abs(YTarget) + abs(YPred)) / 2 + epsilon;
% 
%     % SMAPE es la media de la división
%     % Multiplicamos por 100 para tener el porcentaje si se desea, 
%     % aunque para optimización el factor 100 es opcional.
%     loss = 100 * mean(numerator ./ denominator, 'all');
% end

function [loss, gradients] = modelGradients(net, X, T, useSMAPE, OnTraining)
    Y = forward(net, X);
    
    % Decidimos la pérdida mediante una lógica simple

    loss = EvaluateLoss(Y,T,useSMAPE);

    if OnTraining
        gradients = dlgradient(loss, net.Learnables);
    else
        gradients = [];
    end
end

function lossVal = calculateValidationLoss(net, mbqVal, useSMAPE)
    % Inicializar acumuladores
    totalLoss = 0;
    numBatches = 0;
    
    % Reiniciar el mini-batch queue de validación al principio
    reset(mbqVal);
    
    % Iterar sobre todo el conjunto de validación
    while hasdata(mbqVal)
        % Leer el siguiente lote
        [X, T] = next(mbqVal);
        
        % Predicción en modo 'test' (importante para Batch Normalization y Dropout)
        % Usamos predict() en lugar de forward() para no guardar info de gradientes
        Y = predict(net, X);
        
        % Calcular pérdida del lote actual
        batchLoss = EvaluateLoss(Y,T,useSMAPE);
        
        % Acumular (convertir a double para liberar memoria dlarray)
        totalLoss = totalLoss + double(extractdata(batchLoss));
        numBatches = numBatches + 1;
    end
    
    % Promedio final de la pérdida de validación
    lossVal = totalLoss / numBatches;
end

function loss = EvaluateLoss(Y,T,useSMAPE)

    if useSMAPE
        epsilon = cast(0.001, 'like', Y); 
        num = abs(Y - T);
        den = (abs(Y) + abs(T) + epsilon) / 2;
        loss = mean(num ./ den, 'all');
    else
        % MSE por defecto
        loss = mean((Y - T).^2, 'all');
    end
end
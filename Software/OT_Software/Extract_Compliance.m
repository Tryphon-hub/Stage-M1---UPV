NumSamples = size(c,1);

MeanIter = zeros(1000,1);
Iters = zeros(1,NumSamples);
StdIter = zeros(1000,1);

RightSamples = ones(1,NumSamples);

for iSample = 1:NumSamples
    cSample = c{iSample}./c{iSample}(1);
    if any(cSample > 1)
        RightSamples(iSample) = 0;
    end
end

ValidSamples = find(RightSamples);
for iIter = 1:1000
    counterSample = 0;
    ComplianceIter = [];
    OneSol = 0;
    for iSample = ValidSamples
        if length(c{iSample}) >= iIter
            OneSol = 1;
            counterSample = counterSample + 1;
            ComplianceIter(counterSample) = c{iSample}(iIter)/c{iSample}(1);
        end
    end
    if OneSol
        MeanIter(iIter) = mean(ComplianceIter);
        StdIter(iIter) = std(ComplianceIter);
    else
        break
    end
end
MeanIter = MeanIter(1:iIter-1);
StdIter = StdIter(1:iIter-1);
MaxVal = MeanIter + StdIter; 
MinVal = MeanIter - StdIter; 
iteraciones = 1:iIter-1;

T = table(iteraciones', MeanIter, MaxVal, MinVal, ...
    'VariableNames', {'iteracion', 'media', 'superior', 'inferior'});

% 3. Escribir el archivo CSV
writetable(T, 'Datos_Hybrid.csv');






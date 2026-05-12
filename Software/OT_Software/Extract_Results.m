load('FOM_Resutls_v03_Verification.mat')
FOM_c = c;
FOM_ItsFull = ItsFull;
FOM_NumIts = NumIts;

load('Hybrid_Resutls_v03_Verification.mat')
Hybrid_c = c;
Hybrid_ItsFull = ItsFull;
Hybrid_NumIts = NumIts;

load('NN_Resutls_v03_Verification.mat')
NN_c = c;
NN_ItsFull = ItsFull;
NN_NumIts = NumIts;

All_c = zeros(size(NN_c,1),1);
Its = zeros(size(NN_c,1),1);
for iSample = 1:size(NN_c,1)

    All_c(iSample,1) = FOM_c{iSample}(end)/FOM_c{iSample}(1);
    All_c(iSample,2) = Hybrid_c{iSample}(end)/Hybrid_c{iSample}(1);
    All_c(iSample,3) = NN_c{iSample}(end)/NN_c{iSample}(1);

    Its(iSample,1) = FOM_ItsFull(iSample)/FOM_NumIts(iSample);
    Its(iSample,2) = Hybrid_ItsFull(iSample)/Hybrid_NumIts(iSample);
    Its(iSample,3) = NN_ItsFull(iSample)/NN_NumIts(iSample);


end

writematrix(All_c,'Compliance.csv');
writematrix(Its,'Iterations.csv');






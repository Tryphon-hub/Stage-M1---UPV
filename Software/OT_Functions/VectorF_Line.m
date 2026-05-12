function F = Vector_Ft(F,XY,Top,NGpL,FEDegree,Tractions)

%% General data
NNpS=size(Top,1); % Number of nodes per elemento
gdlx=1:2:2*NNpS-1; % gdl locales del segmento en x
gdly=2:2:2*NNpS;   % gdl locales del segmento en y
Nodes = unique(Top);

Tn(Nodes) = Tractions(:,1);
Tt(Nodes) = Tractions(:,2);
% Coordenadas locales y pesos de los puntos de Gauss
Dim=1;
[Psi, W]=pgauss(Dim,1,NGpL); 

% Funciones de forma y sus derivadas en los puntos de Gauss
[N,dNdPsi,] = Shapef_1D(Psi,FEDegree,1,'Standard');

% Lazo para todas las condiciones de cargas en lados
for iElm= 1:size(Top,2)

    TopE = Top(:,iElm);
    Tn_e = Tn(TopE)'; % Tensión normal en los nodos del elemento
    Tt_e = Tt(TopE)'; % Tension tangencial en los nodos del elemento
    fe=zeros(2*NNpS,1); % Vector fe de elemento (solo en nodos del segmento)
    
    % Lazon en puntos de Gauss
    for iGauss=1:NGpL

         NpG=N(iGauss,:); % funciones de forma para el punto de Gauss en concreto en todos los nodos del segmento
         
         PpG=NpG*Tn_e; % presion en el punto de Gauss
         TpG=NpG*Tt_e;
         
         dNpG=dNdPsi(iGauss,:); % dN en el punto de Gauss

         NpGsum=zeros(2*NNpS,2); % matriz N traspuesta de funciones de forma en el punto de Gauss
         NpGsum(gdlx,1)=NpG;
         NpGsum(gdly,2)=NpG;        
                  
         dxdpsi=dNpG*XY(1,TopE)'; % jacobiana
         dydpsi=dNpG*XY(2,TopE)';
         
         txp= PpG.*dydpsi; % tracciones en el contorno
         typ=-PpG.*dxdpsi;
         txt= TpG.*dxdpsi;
         tyt= TpG.*dydpsi;
         
         t=[txp+txt;...  % vector de cargas superficiales t  
            typ+tyt];
         
         NpGt=NpGsum*t; % matriz N traspuesta · t
         
         fe=fe+NpGt*W(iGauss); % contribucion al vector fe
         
    end
    
    % grados de libertad globales en el segmento
    gdls(gdlx)=2*TopE-1; 
    gdls(gdly)=2*TopE;
    % vector F ensamblado
    F(gdls)=F(gdls)+fe; 
    
end
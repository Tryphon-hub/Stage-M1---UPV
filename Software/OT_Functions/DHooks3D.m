function D=DHooks3D(E,nu,Mode,Image) 


if nargin == 2
    Mode = 0;
end

switch Mode
    case 0
        D = E/((1+nu)*(1-2*nu))*[
            1-nu,   nu,   nu,          0,          0,          0;...
            nu, 1-nu,   nu,          0,          0,          0;...
            nu,   nu, 1-nu,          0,          0,          0;...
            0,   0,    0, (1-2*nu)/2,          0,          0;...
            0,   0,    0,          0, (1-2*nu)/2,          0;...
            0,   0,    0,          0,          0, (1-2*nu)/2];
    case 1
        load '4DCT_N_15_Neurons_32_32_32.mat'
        
        B = DCT2020(Image);
               
        AuxB = zeros(4,1);
        
        AuxB(1) = B(1,1,1);
        AuxB(2) = B(2,1,1);
        AuxB(3) = B(1,2,1);
        AuxB(4) = B(1,1,2);
        
        Image2 = IDCT2020(AuxB);
        
        AuxB(2:end) = AuxB(2:end)/AuxB(1);
        
        AuxD = net(AuxB(2:end));
        
        D      = zeros(6,6);
        aux    = bsxfun(@ge,(1:6)',1:6);
        D(aux) = AuxD;            %Lower triangle +diag
        D      = D +tril(D,-1)'; %Upper triangle
        
        D = D * AuxB(1);
end

    
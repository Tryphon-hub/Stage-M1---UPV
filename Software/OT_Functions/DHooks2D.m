function D=DHooks2D(E,nu,Mode)
% D=DHooks2D(E,nu,Mode)
% 
% Evaluate D matrix:  Sigma=D·Eps
% Mode:  0 => PlaneStrain
%        1 => PlaneStress

switch Mode
    case 'Plane Strain'  %Plane Strain

        D=E/((1+nu)*(1-2*nu))*[...
            1-nu , nu   , 0          ;...
            nu   , 1-nu , 0          ;...
            0    , 0    , (1-2*nu)/2]; %D matrix for plane strain
    case 'Plane Stress'  %Plane Stress
        D=E/(1-nu^2)*[...
            1  , nu   , 0          ;...
            nu , 1    , 0          ;...
            0  , 0    , (1-nu)/2]; %D matrix for plane stress
end

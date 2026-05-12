function [Stress,VMStress] = EvalStress(Top,XY,Rel_Density,D,Sol,Dim,SquareElm,ElmTOP,StressHypo,E,nu)

% FE Discretization. SPACE
NElem = size(Top,2);
NNpE = size(Top,1);

if length(Rel_Density) == 1
    if ElmTOP
        Rel_Density = Rel_Density*ones(size(Top,2),1);
    else
        Rel_Density = Rel_Density*ones(size(TXYop,2),1);
    end

end


FEDegree = 1;
Stress = zeros(NElem,6);
if Dim == 3
    NFpE = 6;
else
    NFpE = 4;
end

% Stress = [Sxx,Syy,Szz,Sxy,Syz,Szx];


%% Material properties


if Dim == 2
    [N,dNdPsi,dNdEta] = Shapef_2D([0 0],FEDegree,NFpE,1);
else
    [N,dNdPsi,dNdEta,dNdTau] = Shapef_3D([0 0 0],FEDegree,NFpE,1);
end

for iElm = 1:NElem
    Top_Elm = Top(:,iElm);
    if Dim == 2
        DoF = [((Dim*Top_Elm) -1), (Dim*Top_Elm)]';
    else
        DoF = [((Dim*Top_Elm)-2), ((Dim*Top_Elm) -1), (Dim*Top_Elm)]';
    end

    XYZ_Elm = zeros(NNpE,Dim);
    for iDim = 1:Dim
        XYZ_Elm(:,iDim) = XY(iDim,Top_Elm)';
    end
    if ~SquareElm || iElm == 1
        J = zeros(Dim,Dim);

        if Dim == 2
            J(1,1) = dNdPsi(1,:) * XYZ_Elm(:,1);
            J(2,1) = dNdEta(1,:) * XYZ_Elm(:,1);
            J(1,2) = dNdPsi(1,:) * XYZ_Elm(:,2);
            J(2,2) = dNdEta(1,:) * XYZ_Elm(:,2);

            dN = J^(-1) *[dNdPsi(1,:);dNdEta(1,:)];
            dNdx = dN(1,:);
            dNdy = dN(2,:);

            B = [];
            for iNode = 1:NNpE
                AuxB =[dNdx(iNode) 0          ;...
                    0           dNdy(iNode);...
                    dNdy(iNode) dNdx(iNode)];
                B = [B AuxB];
            end

        else
            J(1,1) = dNdPsi(1,:) * XYZ_Elm(:,1);
            J(1,2) = dNdEta(1,:) * XYZ_Elm(:,1);
            J(1,3) = dNdTau(1,:) * XYZ_Elm(:,1);
            J(2,1) = dNdPsi(1,:) * XYZ_Elm(:,2);
            J(2,2) = dNdEta(1,:) * XYZ_Elm(:,2);
            J(2,3) = dNdTau(1,:) * XYZ_Elm(:,2);
            J(3,1) = dNdPsi(1,:) * XYZ_Elm(:,3);
            J(3,2) = dNdEta(1,:) * XYZ_Elm(:,3);
            J(3,3) = dNdTau(1,:) * XYZ_Elm(:,3);

            dN = J^(-1) *[dNdPsi(1,:);dNdEta(1,:);dNdTau(1,:)];
            dNdx = dN(1,:);
            dNdy = dN(2,:);
            dNdz = dN(3,:);

            B = [];
            for iNode = 1:NNpE
                AuxB =[dNdx(iNode) 0           0          ;...
                    0           dNdy(iNode) 0          ;...
                    0           0           dNdz(iNode);...
                    dNdy(iNode) dNdx(iNode) 0          ;...
                    0           dNdz(iNode) dNdy(iNode);...
                    dNdz(iNode) 0           dNdx(iNode)];
                B = [B AuxB];
            end

        end

    end
    if ElmTOP
        if Dim == 2
            Stress(iElm,[1 2 4]) = (Rel_Density(iElm)*D*B*Sol(DoF(:)))';
        else
            Stress(iElm,:) = (Rel_Density(iElm)*D*B*Sol(DoF(:)))';
        end
    else
        Elm_Dens = N(1,:)*Rel_Density(Top_Elm);
        if Dim == 2
            Stress(iElm,[1 2 4]) = (Elm_Dens*D*B*Sol(DoF(:)))';
        else
            Stress(iElm,:) = (Elm_Dens*D*B*Sol(DoF(:)))';
        end
    end

end

if Dim == 2 && strcmp(StressHypo,'Plane Strain')
    Stress(:,3) = nu*(Stress(:,1)+Stress(:,2));
end

VMStress = sqrt(Stress(:,1).^2+Stress(:,2).^2+Stress(:,3).^2 -...
        (Stress(:,1).*Stress(:,2)+Stress(:,2).*Stress(:,3)+Stress(:,3).*Stress(:,1)) + ...
        3*(Stress(:,4).^2+Stress(:,5).^2+Stress(:,6).^2));





end
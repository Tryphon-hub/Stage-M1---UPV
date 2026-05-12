function [K,Vol,Ke] = FEM_Creator_Elasticity(k,XY,Top,NGP,FEDegree,NFpE,Dim,D,SquareElm,ElemTO)

% FE Discretization. SPACE
NElem = size(Top,2);
NNpE = size(Top,1);
NNodes = size(XY,2);

if length(k) == 1
    k = ones(NElem,1) * k;
end

[Xg,Wg]= pgauss(Dim,NFpE,NGP,FEDegree,0);
if Dim == 2
    [N,dNdPsi,dNdEta] = Shapef_2D(Xg,FEDegree,NFpE,1);
else
    [N,dNdPsi,dNdEta,dNdTau] = Shapef_3D(Xg,FEDegree,NFpE,1);
end

% Stiffness matrix

i = [];
j = [];
v = [];
Ke = zeros(Dim*NNpE,Dim*NNpE,NElem);
Vol = 0;

B = zeros(size(D,1),Dim*NNpE,NGP);
Jacobian = zeros(NGP,1);

for iElem = 1:NElem


    Top_Elm = Top(:,iElem);
    if Dim == 2
        DoF = [((Dim*Top_Elm) -1), (Dim*Top_Elm)]';
    else
        DoF = [((Dim*Top_Elm)-2), ((Dim*Top_Elm) -1), (Dim*Top_Elm)]';
    end

    XYZ_Elm = zeros(NNpE,Dim);
    Ve = 0;
    for iDim = 1:Dim
        XYZ_Elm(:,iDim) = XY(iDim,Top_Elm)';
    end


    for iGP = 1:NGP
        if ~SquareElm || iElem == 1
            J = zeros(Dim,Dim);

            if Dim == 2
                J(1,1) = dNdPsi(iGP,:) * XYZ_Elm(:,1);
                J(2,1) = dNdEta(iGP,:) * XYZ_Elm(:,1);
                J(1,2) = dNdPsi(iGP,:) * XYZ_Elm(:,2);
                J(2,2) = dNdEta(iGP,:) * XYZ_Elm(:,2);

                dN = J^(-1) *[dNdPsi(iGP,:);dNdEta(iGP,:)];
                dNdx = dN(1,:);
                dNdy = dN(2,:);


                for iNode = 1:NNpE
                    AuxB =[dNdx(iNode) 0          ;...
                        0           dNdy(iNode);...
                        dNdy(iNode) dNdx(iNode)];
                    B(:,1+Dim*(iNode-1):Dim*(iNode),iGP) = AuxB;
                end

            else
                J(1,1) = dNdPsi(iGP,:) * XYZ_Elm(:,1);
                J(1,2) = dNdEta(iGP,:) * XYZ_Elm(:,1);
                J(1,3) = dNdTau(iGP,:) * XYZ_Elm(:,1);
                J(2,1) = dNdPsi(iGP,:) * XYZ_Elm(:,2);
                J(2,2) = dNdEta(iGP,:) * XYZ_Elm(:,2);
                J(2,3) = dNdTau(iGP,:) * XYZ_Elm(:,2);
                J(3,1) = dNdPsi(iGP,:) * XYZ_Elm(:,3);
                J(3,2) = dNdEta(iGP,:) * XYZ_Elm(:,3);
                J(3,3) = dNdTau(iGP,:) * XYZ_Elm(:,3);

                dN = J^(-1) *[dNdPsi(iGP,:);dNdEta(iGP,:);dNdTau(iGP,:)];
                dNdx = dN(1,:);
                dNdy = dN(2,:);
                dNdz = dN(3,:);


                for iNode = 1:NNpE
                    AuxB =[dNdx(iNode) 0           0          ;...
                        0           dNdy(iNode) 0          ;...
                        0           0           dNdz(iNode);...
                        dNdy(iNode) dNdx(iNode) 0          ;...
                        0           dNdz(iNode) dNdy(iNode);...
                        dNdz(iNode) 0           dNdx(iNode)];
                    B(:,1+Dim*(iNode-1):Dim*(iNode),iGP) = AuxB;
                end

            end


            Jacobian(iGP) = det(J);
            if Jacobian(iGP) <=0
                disp(['Please check element ' num2str(iElem) '. The Jacobian takes negative valies']);
            end
        end

        if ElemTO
            Ke(:,:,iElem) = Ke(:,:,iElem) + k(iElem) * B(:,:,iGP)' * D * B(:,:,iGP) * Jacobian(iGP) * Wg(iGP);
        else
            Elm_Dens = k(Top_Elm);
            Ke(:,:,iElem) = Ke(:,:,iElem) + N(iGP,:)*Elm_Dens * B(:,:,iGP)' * D * B(:,:,iGP) * Jacobian(iGP) * Wg(iGP);
        end
        Ve = Ve + Jacobian(iGP) * Wg(iGP);


    end

    Vol = Vol + Ve;

    DoF = DoF(:);
    Index_i = repmat(DoF,1,size(DoF,1)); Index_i = Index_i(:);
    Index_j = repmat(DoF',size(DoF,1),1); Index_j = Index_j(:);
    i = [i ; Index_i];
    j = [j ; Index_j];
    v = [v ; reshape(Ke(:,:,iElem),[],1,1)];


end

K = sparse(i,j,v,NNodes*Dim,NNodes*Dim);
K = (K'+K)/2;





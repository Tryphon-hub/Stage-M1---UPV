function [c,dc,ce,InfVol] = Opt(k,U,D,Penal,MeshData,ElemTO,SquareElm,Dim,NGP,NFpE)

%Esta función devuelve la derivada de la energía de deformación en cada
%elemento, a partir de los desplazamientos y la matriz de rigidez
%de cada elemento

FEDegree = 1;

[Xg,Wg]= pgauss(Dim,NFpE,NGP,FEDegree,0);
if Dim == 2
    [N,dNdPsi,dNdEta] = Shapef_2D(Xg,FEDegree,NFpE,1);
    NodCon = MeshData.Surf.NodCon;
    Top = MeshData.Surf.Topology;
else
    [N,dNdPsi,dNdEta,dNdTau] = Shapef_3D(Xg,FEDegree,NFpE,1);
    NodCon = MeshData.Vol.NodCon;
    Top = MeshData.Vol.Topology;
end


NNpE = size(Top,1);
XY = MeshData.XYZ;

if ElemTO
    LoopLength  = size(Top,2);
else
    LoopLength  = size(XY,2);
end

dc = zeros(LoopLength,1);
InfVol = zeros(LoopLength,1); % Influence volume, if is element then is the element volume or if si nodal si the influence volume of a node
ce = zeros(size(Top,2),1);

B = zeros(size(D,1),Dim*NNpE,NGP);
Jacobian = zeros(NGP,1);

for i = 1:LoopLength
    if ElemTO
        Elements = i;
    else
        Position = NodCon(:,2,i); Position = Position(Position>0);
        Elements = NodCon(:,1,i); Elements = Elements(Elements>0);
        %[Pos,Els] = find(Top==i);
    end
    kPos = 0;
    for iElem = Elements'
        kPos = kPos+1;
        Top_Elm = Top(:,iElem);

        if Dim == 2
            DoF = [((Dim*Top_Elm) -1), (Dim*Top_Elm)]';
        else
            DoF = [((Dim*Top_Elm)-2), ((Dim*Top_Elm) -1), (Dim*Top_Elm)]';
        end
        Ue = U(DoF); Ue = Ue(:);

        XYZ_Elm = zeros(NNpE,Dim);
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
                Density = k(iElem)^Penal;
                derDensity = Penal*k(iElem)^(Penal-1);
                InfVol(i) = InfVol(i) + Jacobian(iGP) * Wg(iGP);
            else
                Elm_Dens = k(Top_Elm);
                Density = N(iGP,Position(kPos))*Elm_Dens(Position(kPos)).^Penal;
                derDensity = N(iGP,Position(kPos))*Penal*Elm_Dens(Position(kPos)).^(Penal-1);

                InfVol(i) = InfVol(i) + N(iGP,Position(kPos))*Jacobian(iGP) * Wg(iGP);
            end
            dc(i) = dc(i) - derDensity * Ue'* B(:,:,iGP)' * D * B(:,:,iGP) * Ue* Jacobian(iGP) * Wg(iGP);
            ce(iElem) = ce(iElem) + Density * Ue'* B(:,:,iGP)' * D * B(:,:,iGP) * Ue* Jacobian(iGP) * Wg(iGP);
            
        end

    end
end
c = sum(ce);
end
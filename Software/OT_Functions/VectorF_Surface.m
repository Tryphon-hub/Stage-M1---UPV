function F = VectorF_Surface(F,Topology,XYCoords,tx,ty,tz,ElmList,NGP2D,NGPpS,FEDegree)

[Xg,Wg]= pgauss(2,NGP2D,NGPpS,FEDegree,0);
[NG,dNGdPsi,dNGdEta] = Shapef_2D(Xg,FEDegree,NGPpS,1);

% Integrate over surface elements
for iElm = ElmList
    ElmTop = Topology(:,iElm);
    XY_Elm = XYCoords(:,ElmTop)';

    % Evaluate the fine element traction vector
    te(1:3:12,1) = tx(ElmTop);
    te(2:3:12,1) = ty(ElmTop);
    te(3:3:12,1) = tz(ElmTop);

    % Integration loop
    Fe = zeros(12,1);
    for iGP = 1:NGP2D

        % Evaluate Jacobian (not strictly necessary for perfect
        % square elements
        J(1,1) = dNGdPsi(iGP,:) * XY_Elm(:,1);
        J(2,1) = dNGdEta(iGP,:) * XY_Elm(:,1);
        J(1,2) = dNGdPsi(iGP,:) * XY_Elm(:,2);
        J(2,2) = dNGdEta(iGP,:) * XY_Elm(:,2);

        % Generate shape functions matrix
        Nmat(1,(1:3:12)) = NG(iGP,:);
        Nmat(2,(2:3:12)) = NG(iGP,:);
        Nmat(3,(3:3:12)) = NG(iGP,:);

        Jacobian = abs(det(J));

        % Evaluate the integrand for one element
        Fe = Fe + Nmat' * Nmat * te * Jacobian * Wg(iGP);
    end
    Index = [3*ElmTop-2,3*ElmTop-1,3*ElmTop]';
    Index = Index(:);

    F(Index) = F(Index) + Fe;
end
function [Tractions,Center,Scale,List,Relative_Vol_Frac] = LoadMacroProblemData(File,AutoEquilibrate)

load(File)

% Load forces per face
Fxy = reshape(full(Iter.FEMAnalysis.FElmSide(:,ActiveElements)),2,8,[]);

% Convert forces into tractions. 
SideLength = 2*2.^(-Elm.Level(CalcMesh.ActiveElements));
Center = squeeze(mean(CalcMesh.GlbElmCoord,2));
ElmLength = squeeze(CalcMesh.GlbElmCoord(1,2,:) -  CalcMesh.GlbElmCoord(1,1,:));
Scale = ElmLength/2;
List = 1:CalcMesh.NumEle;
Relative_Vol_Frac = OptTop.HistorialWeights(ActiveElements,end);
Integral = permute(0.5*SideLength,[3,2,1]);
Txy = bsxfun(@rdivide,Fxy,Integral);

Tractions = zeros(2,8,size(Txy,3));

for iElement = 1:size(Txy,3)
    for iSide = 1:4

        switch iSide
            case 1
                Tractions(1,1:2,iElement) = -Txy(2,1:2,iElement); % Normal component
                Tractions(2,1:2,iElement) =  Txy(1,1:2,iElement); % Tangential component
            case 2
                Tractions(1,3:4,iElement) =  Txy(1,3:4,iElement);
                Tractions(2,3:4,iElement) =  Txy(2,3:4,iElement);
            case 3
                Tractions(1,5:6,iElement) =  Txy(2,5:6,iElement);
                Tractions(2,5:6,iElement) =  -Txy(1,5:6,iElement);
            case 4
                Tractions(1,7:8,iElement) = -Txy(1,7:8,iElement);
                Tractions(2,7:8,iElement) = -Txy(2,7:8,iElement);
        end
    end
end


if AutoEquilibrate
    Tractions = reshape(Tractions,16,[]);

    % Equilibrate the traction proposal

    EqualMatrix = eye(16);
    ConstraintMatrix = zeros(3,16);

    % Equilibrium in x direction
    ConstraintMatrix(1,:) = [0 1 0 1   1 0 1 0  0 -1 0 -1  -1 0 -1 0 ];

    % Equilibrium in y direction
    ConstraintMatrix(2,:) = [-1 0 -1 0   0 1 0 1   1 0 1 0  0 -1 0 -1];

    % Moment equilibrium
    ConstraintMatrix(3,:) = [0 1 0 1     0 1 0 1   0 1 0 1  0 1 0 1] + ...
        1/3*[1 0 -1 0   1 0 -1 0  1 0 -1 0  1 0 -1 0];

    EquilibrationMatrix = [EqualMatrix ConstraintMatrix'; ConstraintMatrix zeros(3)];

    EqTractions = EquilibrationMatrix\[Tractions; zeros(3,size(Tractions,2))];

    EqTractions = EqTractions(1:16,:);

    EqTractions = bsxfun(@times,EqTractions,1./sqrt(sum(EqTractions.^2,1)));

    Tractions = reshape(EqTractions,2,8,size(Txy,3));

end

function [Sol,Vol] = SolveFE(MeshData,Rel_Density,NGPpS,NGPpL,D,Tvert,SquareElm,ElemTO)


%% Evaluate stiffness matrix 
Dim = 2;
Top = MeshData.Surf.Topology;
XY = MeshData.XYZ;
FEDegree = 1;
NSpE = size(Top,1);


[K,Vol] = FEM_Creator_Elasticity(Rel_Density,XY,Top,NGPpS,FEDegree,NSpE,Dim,D,SquareElm,ElemTO);

%% Generate force vector
F = zeros(size(K,1),1);%% Posible modificación
for iPN = 1:length(MeshData.PhysicalNames)
    switch MeshData.PhysicalNames(iPN).Name{1}
        case '"Abajo"'
            Neuman_Tag = MeshData.PhysicalNames(iPN).Tag;
            for iCv = 1:numel(MeshData.Curve)
                if find(MeshData.Curve(iCv).PhyTag == Neuman_Tag)
                    NeumanCv = iCv;
                    break
                end
            end
            NodesSide = unique(MeshData.Curve(NeumanCv).Topology);
            xnode = MeshData.XYZ(1,NodesSide);
            xmin = min(xnode);
            xmax = max(xnode);
            tn = Tvert(1,1)*(xmax-xnode)/(xmax-xmin) + Tvert(1,2)*(xnode-xmin)/(xmax-xmin);
            tt = Tvert(2,1)*(xmax-xnode)/(xmax-xmin) + Tvert(2,2)*(xnode-xmin)/(xmax-xmin);
            Tractions = [tn',tt'];
            F = VectorF_Line(F,MeshData.XYZ,MeshData.Curve(NeumanCv).Topology,NGPpL,FEDegree,Tractions);
        case '"Derecha"'
            Neuman_Tag = MeshData.PhysicalNames(iPN).Tag;
            for iCv = 1:numel(MeshData.Curve)
                if find(MeshData.Curve(iCv).PhyTag == Neuman_Tag)
                    NeumanCv = iCv;
                    break
                end
            end
            NodesSide = unique(MeshData.Curve(NeumanCv).Topology);
            ynode = MeshData.XYZ(2,NodesSide);
            ymin = min(ynode);
            ymax = max(ynode);
            tn = Tvert(1,3)*(ymax-ynode)/(ymax-ymin) + Tvert(1,4)*(ynode-ymin)/(ymax-ymin);
            tt = Tvert(2,3)*(ymax-ynode)/(ymax-ymin) + Tvert(2,4)*(ynode-ymin)/(ymax-ymin);
            Tractions = [tn',tt'];
            F = VectorF_Line(F,MeshData.XYZ,MeshData.Curve(NeumanCv).Topology,NGPpL,FEDegree,Tractions);

        case '"Arriba"'
            Neuman_Tag = MeshData.PhysicalNames(iPN).Tag;
            for iCv = 1:numel(MeshData.Curve)
                if find(MeshData.Curve(iCv).PhyTag == Neuman_Tag)
                    NeumanCv = iCv;
                    break
                end
            end
            NodesSide = unique(MeshData.Curve(NeumanCv).Topology);
            xnode = MeshData.XYZ(1,NodesSide);
            xmin = min(xnode);
            xmax = max(xnode);
            tn = Tvert(1,6)*(xmax-xnode)/(xmax-xmin) + Tvert(1,5)*(xnode-xmin)/(xmax-xmin);
            tt = Tvert(2,6)*(xmax-xnode)/(xmax-xmin) + Tvert(2,5)*(xnode-xmin)/(xmax-xmin);
            Tractions = [tn',tt'];
            F = VectorF_Line(F,MeshData.XYZ,MeshData.Curve(NeumanCv).Topology,NGPpL,FEDegree,Tractions);

        case '"Izquierda"'
            Neuman_Tag = MeshData.PhysicalNames(iPN).Tag;
            for iCv = 1:numel(MeshData.Curve)
                if find(MeshData.Curve(iCv).PhyTag == Neuman_Tag)
                    NeumanCv = iCv;
                    break
                end
            end
            NodesSide = unique(MeshData.Curve(NeumanCv).Topology);
            ynode = MeshData.XYZ(2,NodesSide);
            ymin = min(ynode);
            ymax = max(ynode);
            tn = Tvert(1,8)*(ymax-ynode)/(ymax-ymin) + Tvert(1,7)*(ynode-ymin)/(ymax-ymin);
            tt = Tvert(2,8)*(ymax-ynode)/(ymax-ymin) + Tvert(2,7)*(ynode-ymin)/(ymax-ymin);
            Tractions = [tn',tt'];
            F = VectorF_Line(F,MeshData.XYZ,MeshData.Curve(NeumanCv).Topology,NGPpL,FEDegree,Tractions);
    end
end

% figure
% x = MeshData.XYZ(1,:)';
% y = MeshData.XYZ(2,:)';
% Fx = F(1:2:end);
% Fy = F(2:2:end);
% quiver(x,y,Fx,Fy);

%% Apply Dirichlet Boundary conditions
Sol = zeros(size(K,1),1);
for iPN = 1:length(MeshData.PhysicalNames)
   if strcmp(MeshData.PhysicalNames(iPN).Name{1},'"1"')
       Pivote_Tag = MeshData.PhysicalNames(iPN).Tag;
       break
   end
end

% Find Point pivote number
for iPt = 1:numel(MeshData.Point)
    if find(MeshData.Point(iPt).PhyTag == Pivote_Tag)
        Pivote_Point = iPt; 
        break
    end
end

for iPN = 1:length(MeshData.PhysicalNames)
   if strcmp(MeshData.PhysicalNames(iPN).Name{1},'"2"')
       Apoyo_Tag = MeshData.PhysicalNames(iPN).Tag;
       break
   end
end

% Find Point pivote number
for iPt = 1:numel(MeshData.Point)
    if find(MeshData.Point(iPt).PhyTag == Apoyo_Tag)
        Apoyo_Point = iPt; 
        break
    end
end


AllDof = 1:2*MeshData.NumNodes;
Dofr = [2*MeshData.Point(Pivote_Point).Nodes-1 2*MeshData.Point(Pivote_Point).Nodes 2*MeshData.Point(Apoyo_Point).Nodes]; % gdl con restriccion
Dofr = unique(Dofr);
Dofl = setdiff(AllDof,Dofr); % Aquí pone los que no tienen restricciones

Sol(Dofr) = zeros(length(Dofr),1); % 42 restringidos 

%% Solve problem
Sol(Dofl) = K(Dofl,Dofl)\(F(Dofl) - K(Dofl,Dofr)*Sol(Dofr)); % libres . Tema 1 y 2 de AMEF
end
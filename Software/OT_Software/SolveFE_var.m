function [Sol,Vol,React] = SolveFE_var(MeshData,Rel_Density,NGPpS,NGPpL,D,Tvert,SquareElm,ElemTO,Supports)
% Same as SolveFE but with PARAMETRIC supports so they can be moved/rotated.
%
% Supports : N x 4 matrix, one row per support:
%              [x, y, fixX, fixY]
%            (x,y)  : target position; the nearest mesh node is constrained.
%            fixX   : 1 to block the x-displacement DOF, 0 otherwise.
%            fixY   : 1 to block the y-displacement DOF, 0 otherwise.
%
% The original SolveFE hard-codes a pin at Point "1" (-1,-1) [fix x,y] and a
% roller at Point "2" (1,-1) [fix y]; passing
%   Supports = [-1 -1 1 1; 1 -1 0 1]
% reproduces it exactly.

%% Evaluate stiffness matrix
Dim = 2;
Top = MeshData.Surf.Topology;
XY = MeshData.XYZ;
FEDegree = 1;
NSpE = size(Top,1);

[K,Vol] = FEM_Creator_Elasticity(Rel_Density,XY,Top,NGPpS,FEDegree,NSpE,Dim,D,SquareElm,ElemTO);

%% Generate force vector (identical to SolveFE)
F = zeros(size(K,1),1);
for iPN = 1:length(MeshData.PhysicalNames)
    switch MeshData.PhysicalNames(iPN).Name{1}
        case '"Abajo"'
            Neuman_Tag = MeshData.PhysicalNames(iPN).Tag;
            for iCv = 1:numel(MeshData.Curve)
                if find(MeshData.Curve(iCv).PhyTag == Neuman_Tag)
                    NeumanCv = iCv; break
                end
            end
            NodesSide = unique(MeshData.Curve(NeumanCv).Topology);
            xnode = MeshData.XYZ(1,NodesSide); xmin = min(xnode); xmax = max(xnode);
            tn = Tvert(1,1)*(xmax-xnode)/(xmax-xmin) + Tvert(1,2)*(xnode-xmin)/(xmax-xmin);
            tt = Tvert(2,1)*(xmax-xnode)/(xmax-xmin) + Tvert(2,2)*(xnode-xmin)/(xmax-xmin);
            Tractions = [tn',tt'];
            F = VectorF_Line(F,MeshData.XYZ,MeshData.Curve(NeumanCv).Topology,NGPpL,FEDegree,Tractions);
        case '"Derecha"'
            Neuman_Tag = MeshData.PhysicalNames(iPN).Tag;
            for iCv = 1:numel(MeshData.Curve)
                if find(MeshData.Curve(iCv).PhyTag == Neuman_Tag)
                    NeumanCv = iCv; break
                end
            end
            NodesSide = unique(MeshData.Curve(NeumanCv).Topology);
            ynode = MeshData.XYZ(2,NodesSide); ymin = min(ynode); ymax = max(ynode);
            tn = Tvert(1,3)*(ymax-ynode)/(ymax-ymin) + Tvert(1,4)*(ynode-ymin)/(ymax-ymin);
            tt = Tvert(2,3)*(ymax-ynode)/(ymax-ymin) + Tvert(2,4)*(ynode-ymin)/(ymax-ymin);
            Tractions = [tn',tt'];
            F = VectorF_Line(F,MeshData.XYZ,MeshData.Curve(NeumanCv).Topology,NGPpL,FEDegree,Tractions);
        case '"Arriba"'
            Neuman_Tag = MeshData.PhysicalNames(iPN).Tag;
            for iCv = 1:numel(MeshData.Curve)
                if find(MeshData.Curve(iCv).PhyTag == Neuman_Tag)
                    NeumanCv = iCv; break
                end
            end
            NodesSide = unique(MeshData.Curve(NeumanCv).Topology);
            xnode = MeshData.XYZ(1,NodesSide); xmin = min(xnode); xmax = max(xnode);
            tn = Tvert(1,6)*(xmax-xnode)/(xmax-xmin) + Tvert(1,5)*(xnode-xmin)/(xmax-xmin);
            tt = Tvert(2,6)*(xmax-xnode)/(xmax-xmin) + Tvert(2,5)*(xnode-xmin)/(xmax-xmin);
            Tractions = [tn',tt'];
            F = VectorF_Line(F,MeshData.XYZ,MeshData.Curve(NeumanCv).Topology,NGPpL,FEDegree,Tractions);
        case '"Izquierda"'
            Neuman_Tag = MeshData.PhysicalNames(iPN).Tag;
            for iCv = 1:numel(MeshData.Curve)
                if find(MeshData.Curve(iCv).PhyTag == Neuman_Tag)
                    NeumanCv = iCv; break
                end
            end
            NodesSide = unique(MeshData.Curve(NeumanCv).Topology);
            ynode = MeshData.XYZ(2,NodesSide); ymin = min(ynode); ymax = max(ynode);
            tn = Tvert(1,8)*(ymax-ynode)/(ymax-ymin) + Tvert(1,7)*(ynode-ymin)/(ymax-ymin);
            tt = Tvert(2,8)*(ymax-ynode)/(ymax-ymin) + Tvert(2,7)*(ynode-ymin)/(ymax-ymin);
            Tractions = [tn',tt'];
            F = VectorF_Line(F,MeshData.XYZ,MeshData.Curve(NeumanCv).Topology,NGPpL,FEDegree,Tractions);
    end
end

%% Apply parametric Dirichlet boundary conditions
Sol = zeros(size(K,1),1);
Dofr = [];
for is = 1:size(Supports,1)
    px = Supports(is,1); py = Supports(is,2);
    d2 = (MeshData.XYZ(1,:)-px).^2 + (MeshData.XYZ(2,:)-py).^2;
    [~,node] = min(d2);                      % nearest mesh node to the support
    if Supports(is,3); Dofr(end+1) = 2*node-1; end  %#ok<AGROW>  block x
    if Supports(is,4); Dofr(end+1) = 2*node;   end  %#ok<AGROW>  block y
end
Dofr = unique(Dofr);

AllDof = 1:2*MeshData.NumNodes;
Dofl = setdiff(AllDof,Dofr);
Sol(Dofr) = 0;

%% Solve
Sol(Dofl) = K(Dofl,Dofl)\(F(Dofl) - K(Dofl,Dofr)*Sol(Dofr));

%% Support reaction (should be ~0 for a load that induces no reaction)
React = K(Dofr,:)*Sol - F(Dofr);
end

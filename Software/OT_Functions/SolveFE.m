function [Sol,Vol] = SolveFE(MeshData,Rel_Density,NGPpV,NGPpS,D,TnTt,SquareElm,ElemTO)  


%% Evaluate stiffness matrix 
Dim = 3;
Top = MeshData.Vol.Topology;
XYZ = MeshData.XYZ;
FEDegree = 1;
NFpE = 6;
NGP2D = 4;

[K,Vol] = FEM_Creator_Elasticity(Rel_Density,XYZ,Top,NGPpV,FEDegree,NFpE,Dim,D,SquareElm,ElemTO);

%% Generate force vector
F = zeros(size(K,1),1);%% Posible modificación

for iPN = 1:length(MeshData.PhysicalNames)
    switch MeshData.PhysicalNames(iPN).Name{1}
        case '"Xmin"'
            % Surface selection
            Neuman_Tag = MeshData.PhysicalNames(iPN).Tag;
            for iSf = 1:numel(MeshData.Surf)
                if find(MeshData.Surf(iSf).PhyTag == Neuman_Tag)
                    NeumanSf = iSf;
                    break
                end
            end

            % Get topology and nodal coordinates of the surface elements
            NodesSide = unique(MeshData.Surf(NeumanSf).Topology);
            yznode = MeshData.XYZ([2,3],NodesSide);
            yzmin = min(yznode,[],2);
            yzmax = max(yznode,[],2);

            % Get the nodal tractions of the macro element
            TnTtSurf = TnTt(:,:,1);
                        
            % Interpolate the nodal tractions to the fine element nodes
            yMed = (yzmax(1)+yzmin(1))/2; h_y = (yzmax(1)-yzmin(1))/2;
            zMed = (yzmax(2)+yzmin(2))/2; h_z = (yzmax(2)-yzmin(2))/2;
            yLocal = (yznode(1,:)-yMed)/h_y;
            zLocal = (yznode(2,:)-zMed)/h_z;
            
            N = Shapef_2D([yLocal;zLocal]',1,4,0);
            
            % Apply interpolation and rotation (from normal and tangential)
            % to XYZ. If the surface is not flat, a different rotation must
            % be performed at each node. A normal to a node must be
            % evaluated.
            tx = []; ty = []; tz = [];
            tx(NodesSide,1) = - N*TnTtSurf(1,:)';
            ty(NodesSide,1) = - N*TnTtSurf(2,:)';
            tz(NodesSide,1) =   N*TnTtSurf(3,:)';
            
            % Asign element data
            NumSurfElms = size(MeshData.Surf(NeumanSf).Topology,2);
            Topology = MeshData.Surf(NeumanSf).Topology;
            YZCoords = MeshData.XYZ([2,3],:);
            ElmList = 1:NumSurfElms;
            
            % Evaluate global vector
            F = VectorF_Surface(F,Topology,YZCoords,tx,ty,tz,ElmList,NGP2D,NGPpS,FEDegree);
            
           
        case '"Xmax"'
             % Surface selection
            Neuman_Tag = MeshData.PhysicalNames(iPN).Tag;
            for iSf = 1:numel(MeshData.Surf)
                if find(MeshData.Surf(iSf).PhyTag == Neuman_Tag)
                    NeumanSf = iSf;
                    break
                end
            end

            % Get topology and nodal coordinates of the surface elements
            NodesSide = unique(MeshData.Surf(NeumanSf).Topology);
            yznode = MeshData.XYZ([2,3],NodesSide);
            yzmin = min(yznode,[],2);
            yzmax = max(yznode,[],2);

            % Get the nodal tractions of the macro element
            TnTtSurf = TnTt(:,:,2);
                        
            % Interpolate the nodal tractions to the fine element nodes
            yMed = (yzmax(1)+yzmin(1))/2; h_y = (yzmax(1)-yzmin(1))/2;
            zMed = (yzmax(2)+yzmin(2))/2; h_z = (yzmax(2)-yzmin(2))/2;
            yLocal = (yznode(1,:)-yMed)/h_y;
            zLocal = (yznode(2,:)-zMed)/h_z;
            
            N = Shapef_2D([yLocal;zLocal]',1,4,0);
            
            % Apply interpolation and rotation (from normal and tangential)
            % to XYZ. If the surface is not flat, a different rotation must
            % be performed at each node. A normal to a node must be
            % evaluated.
            tx = []; ty = []; tz = [];
            tx(NodesSide,1) =   N*TnTtSurf(1,:)';
            ty(NodesSide,1) =   N*TnTtSurf(2,:)';
            tz(NodesSide,1) =   N*TnTtSurf(3,:)';
            
            % Asign element data
            NumSurfElms = size(MeshData.Surf(NeumanSf).Topology,2);
            Topology = MeshData.Surf(NeumanSf).Topology;
            YZCoords = MeshData.XYZ([2,3],:);
            ElmList = 1:NumSurfElms;
            
            % Evaluate global vector
            F = VectorF_Surface(F,Topology,YZCoords,tx,ty,tz,ElmList,NGP2D,NGPpS,FEDegree);

        case '"Ymin"'
            % Surface selection
            Neuman_Tag = MeshData.PhysicalNames(iPN).Tag;
            for iSf = 1:numel(MeshData.Surf)
                if find(MeshData.Surf(iSf).PhyTag == Neuman_Tag)
                    NeumanSf = iSf;
                    break
                end
            end

            % Get topology and nodal coordinates of the surface elements
            NodesSide = unique(MeshData.Surf(NeumanSf).Topology);
            xznode = MeshData.XYZ([1,3],NodesSide);
            xzmin = min(xznode,[],2);
            xzmax = max(xznode,[],2);

            % Get the nodal tractions of the macro element
            TnTtSurf = TnTt(:,:,3);
                        
            % Interpolate the nodal tractions to the fine element nodes
            xMed = (xzmax(1)+xzmin(1))/2; h_x = (xzmax(1)-xzmin(1))/2;
            zMed = (xzmax(2)+xzmin(2))/2; h_z = (xzmax(2)-xzmin(2))/2;
            xLocal = (xznode(1,:)-xMed)/h_x;
            zLocal = (xznode(2,:)-zMed)/h_z;
            
            N = Shapef_2D([xLocal;zLocal]',1,4,0);
            
            % Apply interpolation and rotation (from normal and tangential)
            % to XYZ. If the surface is not flat, a different rotation must
            % be performed at each node. A normal to a node must be
            % evaluated.
            tx = []; ty = []; tz = [];
            tx(NodesSide,1) =   N*TnTtSurf(2,:)';
            ty(NodesSide,1) = - N*TnTtSurf(1,:)';
            tz(NodesSide,1) =   N*TnTtSurf(3,:)';
            
            % Asign element data
            NumSurfElms = size(MeshData.Surf(NeumanSf).Topology,2);
            Topology = MeshData.Surf(NeumanSf).Topology;
            XZCoords = MeshData.XYZ([1,3],:);
            ElmList = 1:NumSurfElms;
            
            % Evaluate global vector
            F = VectorF_Surface(F,Topology,XZCoords,tx,ty,tz,ElmList,NGP2D,NGPpS,FEDegree);

        case '"Ymax"'
             % Surface selection
            Neuman_Tag = MeshData.PhysicalNames(iPN).Tag;
            for iSf = 1:numel(MeshData.Surf)
                if find(MeshData.Surf(iSf).PhyTag == Neuman_Tag)
                    NeumanSf = iSf;
                    break
                end
            end

            % Get topology and nodal coordinates of the surface elements
            NodesSide = unique(MeshData.Surf(NeumanSf).Topology);
            xznode = MeshData.XYZ([1,3],NodesSide);
            xzmin = min(xznode,[],2);
            xzmax = max(xznode,[],2);

            % Get the nodal tractions of the macro element
            TnTtSurf = TnTt(:,:,4);
                        
            % Interpolate the nodal tractions to the fine element nodes
            xMed = (xzmax(1)+xzmin(1))/2; h_x = (xzmax(1)-xzmin(1))/2;
            zMed = (xzmax(2)+xzmin(2))/2; h_z = (xzmax(2)-xzmin(2))/2;
            xLocal = (xznode(1,:)-xMed)/h_x;
            zLocal = (xznode(2,:)-zMed)/h_z;
            
            N = Shapef_2D([xLocal;zLocal]',1,4,0);
            
            % Apply interpolation and rotation (from normal and tangential)
            % to XYZ. If the surface is not flat, a different rotation must
            % be performed at each node. A normal to a node must be
            % evaluated.
            tx = []; ty = []; tz = [];
            tx(NodesSide,1) = - N*TnTtSurf(2,:)';
            ty(NodesSide,1) =   N*TnTtSurf(1,:)';
            tz(NodesSide,1) =   N*TnTtSurf(3,:)';
            
            % Asign element data
            NumSurfElms = size(MeshData.Surf(NeumanSf).Topology,2);
            Topology = MeshData.Surf(NeumanSf).Topology;
            XZCoords = MeshData.XYZ([1,3],:);
            ElmList = 1:NumSurfElms;
            
            % Evaluate global vector
            F = VectorF_Surface(F,Topology,XZCoords,tx,ty,tz,ElmList,NGP2D,NGPpS,FEDegree);
        case '"Zmin"'
             % Surface selection
            Neuman_Tag = MeshData.PhysicalNames(iPN).Tag;
            for iSf = 1:numel(MeshData.Surf)
                if find(MeshData.Surf(iSf).PhyTag == Neuman_Tag)
                    NeumanSf = iSf;
                    break
                end
            end

            % Get topology and nodal coordinates of the surface elements
            NodesSide = unique(MeshData.Surf(NeumanSf).Topology);
            xynode = MeshData.XYZ([1,2],NodesSide);
            xymin = min(xynode,[],2);
            xymax = max(xynode,[],2);

            % Get the nodal tractions of the macro element
            TnTtSurf = TnTt(:,:,5);
                        
            % Interpolate the nodal tractions to the fine element nodes
            xMed = (xymax(1)+xymin(1))/2; h_x = (xymax(1)-xymin(1))/2;
            yMed = (xymax(2)+xymin(2))/2; h_y = (xymax(2)-xymin(2))/2;
            xLocal = (xynode(1,:)-xMed)/h_x;
            yLocal = (xynode(2,:)-yMed)/h_y;
            
            N = Shapef_2D([xLocal;yLocal]',1,4,0);
            
            % Apply interpolation and rotation (from normal and tangential)
            % to XYZ. If the surface is not flat, a different rotation must
            % be performed at each node. A normal to a node must be
            % evaluated.
            tx = []; ty = []; tz = [];
            tx(NodesSide,1) = - N*TnTtSurf(2,:)';
            ty(NodesSide,1) =   N*TnTtSurf(3,:)';
            tz(NodesSide,1) = -  N*TnTtSurf(1,:)';
            
            % Asign element data
            NumSurfElms = size(MeshData.Surf(NeumanSf).Topology,2);
            Topology = MeshData.Surf(NeumanSf).Topology;
            XZCoords = MeshData.XYZ([1,2],:);
            ElmList = 1:NumSurfElms;
            
            % Evaluate global vector
            F = VectorF_Surface(F,Topology,XZCoords,tx,ty,tz,ElmList,NGP2D,NGPpS,FEDegree);
        case '"Zmax"'
             % Surface selection
            Neuman_Tag = MeshData.PhysicalNames(iPN).Tag;
            for iSf = 1:numel(MeshData.Surf)
                if find(MeshData.Surf(iSf).PhyTag == Neuman_Tag)
                    NeumanSf = iSf;
                    break
                end
            end

            % Get topology and nodal coordinates of the surface elements
            NodesSide = unique(MeshData.Surf(NeumanSf).Topology);
            xynode = MeshData.XYZ([1,2],NodesSide);
            xymin = min(xynode,[],2);
            xymax = max(xynode,[],2);

            % Get the nodal tractions of the macro element
            TnTtSurf = TnTt(:,:,6);
                        
            % Interpolate the nodal tractions to the fine element nodes
            xMed = (xymax(1)+xymin(1))/2; h_x = (xymax(1)-xymin(1))/2;
            yMed = (xymax(2)+xymin(2))/2; h_y = (xymax(2)-xymin(2))/2;
            xLocal = (xynode(1,:)-xMed)/h_x;
            yLocal = (xynode(2,:)-yMed)/h_y;
            
            N = Shapef_2D([xLocal;yLocal]',1,4,0);
            
            % Apply interpolation and rotation (from normal and tangential)
            % to XYZ. If the surface is not flat, a different rotation must
            % be performed at each node. A normal to a node must be
            % evaluated.
            tx = []; ty = []; tz = [];
            tx(NodesSide,1) =  N*TnTtSurf(2,:)';
            ty(NodesSide,1) =  N*TnTtSurf(3,:)';
            tz(NodesSide,1) =  N*TnTtSurf(1,:)';
            
            % Asign element data
            NumSurfElms = size(MeshData.Surf(NeumanSf).Topology,2);
            Topology = MeshData.Surf(NeumanSf).Topology;
            XZCoords = MeshData.XYZ([1,2],:);
            ElmList = 1:NumSurfElms;
            
            % Evaluate global vector
            F = VectorF_Surface(F,Topology,XZCoords,tx,ty,tz,ElmList,NGP2D,NGPpS,FEDegree);
    end
end

%% Apply Dirichlet Boundary conditions

% en GMSH colocar los nombres a los vértices de la geometría según número
% de punto
Sol = zeros(size(K,1),1);
for iPN = 1:length(MeshData.PhysicalNames)
   if strcmp(MeshData.PhysicalNames(iPN).Name{1},'"1"')
       PivoteXYZ_Tag = MeshData.PhysicalNames(iPN).Tag;
       break
   end
end

% Find Point pivote number
for iPt = 1:numel(MeshData.Point)
    if find(MeshData.Point(iPt).PhyTag == PivoteXYZ_Tag)
        PivoteXYZ_Point = iPt; 
        break
    end
end

for iPN = 1:length(MeshData.PhysicalNames)
   if strcmp(MeshData.PhysicalNames(iPN).Name{1},'"2"')
       PivoteYZ_Tag = MeshData.PhysicalNames(iPN).Tag;
       break
   end
end

% Find Point pivote number
for iPt = 1:numel(MeshData.Point)
    if find(MeshData.Point(iPt).PhyTag == PivoteYZ_Tag)
        PivoteYZ_Point = iPt; 
        break
    end
end

for iPN = 1:length(MeshData.PhysicalNames)
   if strcmp(MeshData.PhysicalNames(iPN).Name{1},'"3"')
       PivoteZ_Tag = MeshData.PhysicalNames(iPN).Tag;
       break
   end
end

% Find Point pivote number
for iPt = 1:numel(MeshData.Point)
    if find(MeshData.Point(iPt).PhyTag == PivoteZ_Tag)
        PivoteZ_Point = iPt; 
        break
    end
end


AllDof = 1:Dim*MeshData.NumNodes;
Dofr = [3*MeshData.Point(PivoteXYZ_Point).Nodes-2 3*MeshData.Point(PivoteXYZ_Point).Nodes-1 3*MeshData.Point(PivoteXYZ_Point).Nodes ...
    3*MeshData.Point(PivoteYZ_Point).Nodes-1 3*MeshData.Point(PivoteYZ_Point).Nodes...
    3*MeshData.Point(PivoteZ_Point).Nodes]; % gdl con restriccion
Dofr = unique(Dofr);
Dofl = setdiff(AllDof,Dofr); % Aquí pone los que no tienen restricciones

Sol(Dofr) = zeros(length(Dofr),1); % 42 restringidos 

% Plots
% AuxF = reshape(F,3,[]);
% quiver3(MeshData.XYZ(1,:)',MeshData.XYZ(2,:)',MeshData.XYZ(3,:)',AuxF(1,:)',AuxF(2,:)',AuxF(3,:)')

%% Solve problem
Sol(Dofl) = K(Dofl,Dofl)\(F(Dofl) - K(Dofl,Dofr)*Sol(Dofr)); % libres . Tema 1 y 2 de AMEF
end
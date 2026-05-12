function [Output] = ReadGMSH(FileName)

fid = fopen(FileName);

Continue = 1;
ReadMeshFormat = 0;
ReadPhysicalNames = 0;
ReadEntities = 0;
ReadNodes = 0;
ReadElements = 0;
Line = 0;
while Continue
    Line = Line + 1;
    tline = fgetl(fid); 
    if strcmp(tline,'$MeshFormat')
        ReadMeshFormat = 1;
        continue
    end
    if strcmp(tline,'$EndMeshFormat')
        ReadMeshFormat = 0;
        continue
    end
    if strcmp(tline,'$PhysicalNames')
        ReadPhysicalNames = 1;
        
    end
    if strcmp(tline,'$EndPhysicalNames')
        ReadPhysicalNames = 0;
        
    end
    if strcmp(tline,'$Entities')
        ReadEntities = 1;
        
    end
    if strcmp(tline,'$EndEntities')
        ReadEntities = 0;
        
    end
    if strcmp(tline,'$Nodes')
        ReadNodes = 1;
        
    end
    if strcmp(tline,'$EndNodes')
        ReadNodes = 0;
        
    end
    if strcmp(tline,'$Elements')
        ReadElements = 1;
        
    end
    if strcmp(tline,'$EndElements')
        ReadElements = 0;
        break
    end
    
    
    if ReadMeshFormat
        Output.Mesformat = tline;
    end
    if ReadPhysicalNames
        Line = Line + 1;
        tline = fgetl(fid);
       NumPhysNames = str2double(tline);
       for iPN = 1:NumPhysNames
           Line = Line + 1;
           tline = fgetl(fid);
           AuxLineText = textscan(tline,'%s','Delimiter',' ');
           Output.PhysicalNames(iPN).Dimension = str2double(AuxLineText{1}(1));
           Output.PhysicalNames(iPN).Tag = str2double(AuxLineText{1}(2));
           Output.PhysicalNames(iPN).Name = AuxLineText{1}(3);
       end
        
    end
    if ReadEntities
        Line = Line + 1;
         tline = fgetl(fid);
         AuxLineText = textscan(tline,'%f','Delimiter',' ');
         NumberPoints = AuxLineText{1}(1);
         Output.Entities.NumPoints = NumberPoints;
         NumberCurves = AuxLineText{1}(2);
         Output.Entities.NumCurves = NumberCurves;
         NumberSurfaces = AuxLineText{1}(3);
         Output.Entities.NumSurf = NumberSurfaces;
         NumberVolumes = AuxLineText{1}(4);
         Output.Entities.NumVOl = NumberVolumes;     
         
         for iPt = 1:NumberPoints
             Line = Line + 1;
             tline = fgetl(fid);
             AuxLineText = textscan(tline,'%f','Delimiter',' ');
             Output.Point(AuxLineText{1}(1)).XYZ(1,1) = AuxLineText{1}(2);
             Output.Point(AuxLineText{1}(1)).XYZ(2,1) = AuxLineText{1}(3);
             Output.Point(AuxLineText{1}(1)).XYZ(3,1) = AuxLineText{1}(4);
             NumPhTags = AuxLineText{1}(5);
             for iTag = 1:NumPhTags
                 Output.Point(AuxLineText{1}(1)).PhyTag(iTag) = AuxLineText{1}(5+iTag);
             end
         end
    
         for iCv = 1:NumberCurves
             Line = Line + 1;
             tline = fgetl(fid);
             AuxLineText = textscan(tline,'%f','Delimiter',' ');
             Output.Curve(AuxLineText{1}(1)).minX = AuxLineText{1}(2);
             Output.Curve(AuxLineText{1}(1)).minY = AuxLineText{1}(3);
             Output.Curve(AuxLineText{1}(1)).minZ = AuxLineText{1}(4);
             Output.Curve(AuxLineText{1}(1)).maxX = AuxLineText{1}(5);
             Output.Curve(AuxLineText{1}(1)).maxY = AuxLineText{1}(6);
             Output.Curve(AuxLineText{1}(1)).maxZ = AuxLineText{1}(7);
             NumPhTags = AuxLineText{1}(8);
             for iTag = 1:NumPhTags
                 Output.Curve(AuxLineText{1}(1)).PhyTag(iTag) = AuxLineText{1}(8+iTag);
             end
             Next = 8+NumPhTags+1;
             NumberPoints = AuxLineText{1}(Next);
             for iPt = 1:NumberPoints
                 Output.Curve(AuxLineText{1}(1)).Points(iPt) = AuxLineText{1}(Next+iPt);
             end
         end
         
         for iSf = 1:NumberSurfaces
             Line = Line + 1;
             tline = fgetl(fid);
             AuxLineText = textscan(tline,'%f','Delimiter',' ');
             Output.Surf(AuxLineText{1}(1)).minX = AuxLineText{1}(2);
             Output.Surf(AuxLineText{1}(1)).minY = AuxLineText{1}(3);
             Output.Surf(AuxLineText{1}(1)).minZ = AuxLineText{1}(4);
             Output.Surf(AuxLineText{1}(1)).maxX = AuxLineText{1}(5);
             Output.Surf(AuxLineText{1}(1)).maxY = AuxLineText{1}(6);
             Output.Surf(AuxLineText{1}(1)).maxZ = AuxLineText{1}(7);
             NumPhTags = AuxLineText{1}(8);
             for iTag = 1:NumPhTags
                 Output.Surf(AuxLineText{1}(1)).PhyTag(iTag) = AuxLineText{1}(8+iTag);
             end
             Next = 8+NumPhTags+1;
             NumberCvs = AuxLineText{1}(Next);
             for iCv = 1:NumberCvs
                 Output.Surf(AuxLineText{1}(1)).Curvs(iCv) = AuxLineText{1}(Next+iCv);
             end
         end
         
         for iVol = 1:NumberVolumes
             Line = Line + 1;
             tline = fgetl(fid);
             AuxLineText = textscan(tline,'%f','Delimiter',' ');
             Output.Vol( AuxLineText{1}(1)).minX = AuxLineText{1}(2);
             Output.Vol( AuxLineText{1}(1)).minY = AuxLineText{1}(3);
             Output.Vol( AuxLineText{1}(1)).minZ = AuxLineText{1}(4);
             Output.Vol( AuxLineText{1}(1)).maxX = AuxLineText{1}(5);
             Output.Vol( AuxLineText{1}(1)).maxY = AuxLineText{1}(6);
             Output.Vol( AuxLineText{1}(1)).maxZ = AuxLineText{1}(7);
             NumPhTags = AuxLineText{1}(8);
             for iTag = 1:NumPhTags
                 Output.Vol( AuxLineText{1}(1)).PhyTag(iTag) = AuxLineText{1}(8+iTag);
             end
             Next = 8+NumPhTags+1;
             NumberSfs = AuxLineText{1}(Next);
             for iSf = 1:NumberSfs
                 Output.Vol( AuxLineText{1}(1)).Surfs(iSf) = AuxLineText{1}(Next+iSf);
             end
         end
    end
    
    if ReadNodes
        Line = Line + 1;
        tline = fgetl(fid);
        AuxLineText = textscan(tline,'%f','Delimiter',' ');
        NumberEntities = AuxLineText{1}(1);
        NumberNodes = AuxLineText{1}(2);
        MinNodeTag = AuxLineText{1}(3);
        MaxNodeTag = AuxLineText{1}(4);
        Output.NumNodes = NumberNodes;
        Output.MinNode = MinNodeTag;
        Output.MaxNode = MaxNodeTag;
        
        for iEnt = 1:NumberEntities
            Line = Line + 1;
            tline = fgetl(fid);
            AuxLineText = textscan(tline,'%f','Delimiter',' ');
            Dim = AuxLineText{1}(1);
            Tag = AuxLineText{1}(2);
            Parametric = AuxLineText{1}(3);
            NumNodes = AuxLineText{1}(4);
            Nodes = [];
            for iNode = 1:NumNodes
                Line = Line + 1;
                tline = fgetl(fid);
                AuxLineText = textscan(tline,'%f','Delimiter',' ');
                Nodes(iNode) = AuxLineText{1}(1);
            end
            XYZ = [];
            for iNode = 1:NumNodes
                Line = Line + 1;
                tline = fgetl(fid);
                AuxLineText = textscan(tline,'%f','Delimiter',' ');
                XYZ(1,iNode) = AuxLineText{1}(1);
                XYZ(2,iNode) = AuxLineText{1}(2);
                XYZ(3,iNode) = AuxLineText{1}(3);
            end
            
            if Dim == 0
                Output.Point(Tag).Parametric = Parametric;
                Output.Point(Tag).Nodes = Nodes;
                Output.Point(Tag).XYZ = XYZ;
            end
            if Dim == 1
                Output.Curve(Tag).Parametric = Parametric;
                Output.Curve(Tag).Nodes = Nodes;
                Output.Curve(Tag).XYZ = XYZ;
            end
            if Dim == 2
                Output.Surf(Tag).Parametric = Parametric;
                Output.Surf(Tag).Nodes = Nodes;
                Output.Surf(Tag).XYZ = XYZ;
            end
            if Dim == 3
                Output.Vol(Tag).Parametric = Parametric;
                Output.Vol(Tag).Nodes = Nodes;
                Output.Vol(Tag).XYZ = XYZ;
            end
            Output.XYZ(:,Nodes) = XYZ;
        end 
    end
    if ReadElements
        Line = Line + 1;
        tline = fgetl(fid);
        AuxLineText = textscan(tline,'%f','Delimiter',' ');
        NumberEntities = AuxLineText{1}(1);
        NumberElements = AuxLineText{1}(2);
        MinElemTag = AuxLineText{1}(3);
        MaxElemTag = AuxLineText{1}(4);
        Output.NumElements = NumberElements;
        Output.MinElement = MinElemTag;
        Output.MaxElement = MaxElemTag;
        
        for iEnt = 1:NumberEntities
            Line = Line + 1;
            tline = fgetl(fid);
            AuxLineText = textscan(tline,'%f','Delimiter',' ');
            Dim = AuxLineText{1}(1);
            Tag = AuxLineText{1}(2);
            ElmType = AuxLineText{1}(3);
            NumEls = AuxLineText{1}(4);
            Elements = [];
            Topology = [];
            for iElm = 1:NumEls
                Line = Line + 1;
                tline = fgetl(fid);
                AuxLineText = textscan(tline,'%f','Delimiter',' ');
                Elements(iElm) = AuxLineText{1}(1);
                Topology(:,iElm) = AuxLineText{1}(2:end);
            end
            
            if Dim == 1
                Output.Curve(Tag).ElmType = ElmType;
                Output.Curve(Tag).Elements = Elements;
                Output.Curve(Tag).Topology = Topology;
                Output.Curve(Tag).NodCon = EvalNodCon(Topology);

            end
            if Dim == 2
                Output.Surf(Tag).ElmType = ElmType;
                Output.Surf(Tag).Elements = Elements;
                Output.Surf(Tag).Topology = Topology;
                Output.Surf(Tag).NodCon = EvalNodCon(Topology);
            end
            if Dim == 3
                Output.Vol(Tag).ElmType = ElmType;
                Output.Vol(Tag).Elements = Elements;
                Output.Vol(Tag).Topology = Topology;
                Output.Vol(Tag).NodCon = EvalNodCon(Topology);
            end
        end
    end
end
fclose(fid);
end

function NodCon = EvalNodCon(Topology)

Nodes = unique(Topology)';

for iNode = Nodes

    [Position,Elements] = find(Topology==iNode);

    Num = numel(Position);

    NodCon(1:Num,:,iNode) = [Elements,Position];
end
end
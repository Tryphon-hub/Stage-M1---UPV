function dc_new = Filter(Rel_Density,dc,rmin,MeshData,ElemTO,Dim)

if Dim == 3
    Top = MeshData.Vol.Topology;
else
    Top = MeshData.Surf.Topology;
end
XYZ = MeshData.XYZ;

if ElemTO
    Centroid(1,:) = mean(reshape(XYZ(1,Top),size(Top,1),[]),1);% 4*2000
    Centroid(2,:) = mean(reshape(XYZ(2,Top),size(Top,1),[]),1);% 4*2000
    Centroid(3,:) = mean(reshape(XYZ(3,Top),size(Top,1),[]),1);% 4*2000
else
    Centroid = XYZ;
end


[Idx,D] = rangesearch(Centroid',Centroid',rmin);% Operador de convolución

dc_new = zeros(size(dc));
NumItems = length(dc);
for i = 1:NumItems
    dc_new(i) = 1/(Rel_Density(i)*sum(rmin-D{i})) * sum((rmin-D{i})'.*Rel_Density(Idx{i}).*dc(Idx{i}));% (2.22)
end
end

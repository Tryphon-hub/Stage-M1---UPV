function plot_Traction(DataFile,Num)
%% 
scale_force=10;
cadre=floor(scale_force);


load(DataFile)
figure('Name', 'Topology Optimization Result')

%% Relative density

ImageSize = sqrt(length(Rel_Density(:,Num)));
img = reshape(Rel_Density(:,Num), ImageSize, ImageSize);
imagesc(img)
colormap(flipud(gray))

% Forcer le rendu pour obtenir les positions réelles
drawnow

ax = gca;
axPos = ax.Position;  % [x y largeur hauteur] de l'axe

% La colorbar doit occuper la même hauteur que l'image (32/42 de l'axe)
imgFrac = ImageSize / (ImageSize + 2*cadre);
cbY = axPos(2) + axPos(4) * (cadre + 0.5) / (ImageSize + 2*cadre);
cbHeight = axPos(4) * (ImageSize - 1) / (ImageSize + 2*cadre);

cb = colorbar('Ticks', 0:0.2:1, 'Position', [axPos(3), cbY, 0.02, cbHeight]);
cb.Limits = [0 1];

axis equal
xlim([-cadre ImageSize+cadre])
ylim([-cadre ImageSize+cadre])
hold on


%% Force vectors

% (xmin,ymax) <—— Segment 3 ———— (xmax,ymax)
%      |                              |
%   Segment 4                      Segment 2
%      |                              |
% (xmin,ymin) ——— Segment 1 ———> (xmax,ymin)


T_scale=transpose(Tractions(:,:,Num))*scale_force;

Points=[
    0 ImageSize;
    ImageSize ImageSize;
    ImageSize ImageSize;
    ImageSize 0;
    ImageSize 0;
    0 0;
    0 0;
    0 ImageSize;
    ];

for i=1:8
    sx=Points(i,1)+0.5;
    sy=Points(i,2)+0.5;

    tx=T_scale(i,1);
    ty=T_scale(i,2);
    q=quiver(sx, sy, tx, ty, 0, 'r', 'LineWidth', 2, 'MaxHeadSize', 5);
end


hold on

%% Normal Force distributions

for i=1:2:8 % each segment
    edge=(Points(i+1,:)-Points(i,:))';
    normal=[-edge(2); edge(1)]/norm(edge); % oriented toward the outside

    border_1=T_scale(i,:)*normal;
    border_2=T_scale(i+1,:)*normal;

    b=quiver(Points(i,1)+0.5, Points(i,2)+0.5,  border_1*normal(1), border_1*normal(2),  'b-', 'LineWidth', 1.5, 'MaxHeadSize', 7);

    b=quiver(Points(i+1,1)+0.5, Points(i+1,2)+0.5,  border_2*normal(1), border_2*normal(2),  'b-', 'LineWidth', 1.5, 'MaxHeadSize', 7);

    b=plot([Points(i,1)+0.5+border_1*normal(1),  Points(i+1,1)+0.5+border_2*normal(1)], ...
        [Points(i,2)+0.5+border_1*normal(2),   Points(i+1,2)+0.5+border_2*normal(2)], ...
        'b-', 'LineWidth', 1);
end

legend([b, q], {'Normal force distributions', 'Side nodal forces'}, 'FontSize', 14)



%% Image name
title('Relative Density (OT Result)', 'FontSize', 16)
axis off



end
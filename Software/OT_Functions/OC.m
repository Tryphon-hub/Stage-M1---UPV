function New_Rel_Density = OC(Rel_Density,dc,Ae,VolFrac)
% Ver TFM
%Esta función es el corazón del programa de optimización, el algoritmo
%matemático en sí. Se calculan los nuevos pesos de cada elemento
%(actualización de las variables peso) a partir de la derivada de la
%energía de deformación (numerador) y un valor en el denominador que es un
%multiplicador de lagrange por la derivada del volumen respecto la vble
%peso. este valor es desconocido y se calcula con un algoritmo de
%biseccion.%%


l1=1e-50;
l2 = 10e100;
move = 0.2;

while (100*(l2-l1)/l1 > 1e-7)
    lmid = 0.5*(l2+l1);
     New_Rel_Density = max(0.001,max(Rel_Density-move,min(1.,min...
        (Rel_Density+move,Rel_Density.*(-dc./(lmid*Ae)).^(1/2)))));

    if sum(New_Rel_Density.*Ae) - VolFrac*sum(Ae) > 0
        l1 = lmid;
    else
        l2 = lmid;
    end
end
end
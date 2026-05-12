function [N,dNdx,IdxH] = Shapef_1D(X,Degree,Deriv,Type)



X = X(:);


if nargin < 4
    Type = 'Standard';    IdxH = [];
elseif ~strcmp(Type,'Hermite')
    IdxH = [];
end

if ~Deriv
    dNdx = [];
end

Ones = ones(size(X));

switch Type
    case 'Standard'
        switch Degree
            case 1
                N = [0.5 - X*1/2 , X*1/2 + 0.5];
                if Deriv
                    dNdx = [-1*Ones , 1*Ones]*1/2;
                end
            case 2
                
                N = [-X+X.^2 , 2-2*X.^2 , X+X.^2]/2;
                if Deriv
                    dNdx = [-1*Ones+2*X , -4*X , 1+2*X]/2;
                end
            otherwise
                err('Not implemented')
        end
        
    case 'Hierarchical'
        
        N = [0.5 - X*1/2 , X*1/2 + 0.5];
        if Deriv
            dNdx = [-1*Ones , 1*Ones]*1/2;
        end
        if Degree > 1
            
            for iDegree = 2:Degree
                if floor(iDegree/2) == iDegree/2 % is even
                    %N = [N, 1/factorial(iDegree) * (X.^iDegree - 1)];
                    N = [N, (X.^iDegree - 1)];
                    if Deriv
                        % dNdx = [dNdx , 1/factorial(iDegree) * (iDegree * X.^(iDegree -1))];
                        dNdx = [dNdx , (iDegree * X.^(iDegree -1))];
                    end
                else
                    % N = [N, 1/factorial(iDegree) * (X.^iDegree - X)];
                    N = [N, (X.^iDegree - X)];
                    if Deriv
                        %dNdx = [dNdx , 1/factorial(iDegree) * (iDegree * X.^(iDegree -1) - 1)];
                        dNdx = [dNdx , (iDegree * X.^(iDegree -1) - 1)];
                    end
                end
            end
        end
        
    case 'Armonic'
        N = [0.5 - X*1/2 , X*1/2 + 0.5];
        if Deriv
            dNdx = [-1*Ones , 1*Ones]*1/2;
        end
        if Degree > 1
            Omega = [0 1 1 2 2 3 3 4 4 5 5];
            theta = (X + 1)*pi();
            for iDegree = 2:Degree
                N = [N,sin(11*Omega(Degree)*theta)];
                if Deriv
                    dNdx = [dNdx , cos(11*Omega(Degree)*theta)];
                end
            end
        end
        
    case 'Material'
        N = [0.5 - X*1/2 , X*1/2 + 0.5];
        if Deriv
            dNdx = [-1*Ones , 1*Ones]*1/2;
        end
        if Degree > 1
            InterfJump = -1:2/11:1;
            InterfJump_Ini = InterfJump(1:end-1);
            InterfJump_End = InterfJump(2:end);
            for Position = 1:10
                AuxN = zeros(length(X),1);
                for iP = 1:length(X)
                    if X(iP)>InterfJump_Ini(Position) && X(iP)<=InterfJump_End(Position+1)
                        if X(iP) >= InterfJump(Position+1)
                            AuxN(iP) = 1-(X(iP)-InterfJump(Position+1))/(2/11);
                        else
                            AuxN(iP) = (X(iP)-InterfJump(Position+1))/(2/11) + 1;
                        end
                    else
                        AuxN(iP) = 0;
                    end
                end
                N = [N,AuxN];
            end
        end
    case 'Hermite'
        switch Degree
            case 3
                H1 = @(s)  1/4*(1-s).^2.*(2+s);
                M1 = @(s)  1/8*(1-s).^2.*(1+s);
                H2 = @(s)  1/4*(1+s).^2.*(2-s);
                M2 = @(s) -1/8*(1+s).^2.*(1-s);
                
                N = [H1(X) M1(X) H2(X) M2(X)];
                IdxH = [2 4];
                
                if Deriv
                    dH1 = @(s)  1/4*(2*(s-1).*(s+2) + (1-s).^2);
                    dM1 = @(s)  1/8*(2*(s-1).*(1+s) + (1-s).^2);
                    dH2 = @(s)  1/4*(2*(s+1).*(2-s) - (1+s).^2);
                    dM2 = @(s) -1/8*(2*(1+s).*(1-s) - (1+s).^2);
                    
                    dNdx = [dH1(X) dM1(X) dH2(X) dM2(X)];
                end
            otherwise
                err('Not implemented')
        end
    otherwise
        err('Not implemented')
   
end





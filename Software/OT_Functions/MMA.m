function [New_Rel_Density,L,U] = MMA(c,Rel_Density,dc,Ae,Relative_Vol_Frac,iter,Rel_Density_old,LOld,UOld)
%% DESCRIPTION
%{
 =========================================================================
 METHOD OF MOVING ASYMPTOTES
 =========================================================================
 This script performs one MMA-iteration, aimed at solving the nonlinear
 programming problem:
         
    Minimize  f_0(x) + a_0*z + sum(c_i*y_i + 0.5*d_i*(y_i)^2)

  subject to  f_i(x) - a_i*z - y_i <= 0,   i = 1,...,m
              xmin_j <= x_j <= xmax_j,     j = 1,...,n
              z >= 0, y_i >= 0,            i = 1,...,m
 
 Recomended lectures on this subject:
 
 1. K. Svanberg (1987): The method of moving asymptotes—a new method
    for structural optimization. Int J Numer Methods Eng 24(2):359–373.
 2. K. Svanberg (2007): MMA and GCMMA, versions September 2007. 
    Optimization and Systems Theory 104.
 
 *In every comment, CV stands for "Column Vector"*
 =========================================================================
 Version 3                                                Date: 22/03/2021
 =========================================================================
%}
%% INICIALIZATION


vf = Relative_Vol_Frac;
V0 = sum(Ae);             %Volume not considering weights                
V  = sum(Rel_Density.*Ae)/(vf*V0);   %Relative volume: V = V_real/V_desired
dV = (V-1)*Ae'/(vf*V0);

% Input of the original function 
%-------------------------------------------------------------------------
xval = Rel_Density;     %CV with current values of variables x_j.

if iter > 1
    xold1 = Rel_Density_old(:,1);     %xval, 1 iter. ago.
    L = LOld(:,1);   %CV with lower asymptotes from previous iter.
    U = UOld(:,1);   %CV with upper asymptotes from previous iter.
    if iter > 2
        xold2 = Rel_Density_old(:,2); %xval, 2 iter. ago.
    end
end

f0val = c;   % Value of objective function f_0 at xval.
df0dx = dc;  % CV with derivatives of objective function 
                          % f_0 with respect to variables x_j, 
                          % calculated at xval.
df0dx2 = 0*df0dx;         % CV with non-mixed second derivatives of 
                          % objective function f_0 with respect to 
                          % variables x_j, calculated at xval.

% df0dx2(j) = the second derivative of f_0 with respect to x_j (twice).        
% Important note: If not available, simply let df0dx2 = 0*df0dx.

% df0dx = min(max(df0dx(~isoutlier(df0dx))),df0dx); Interesa????
% df0dx = max(min(df0dx(~isoutlier(df0dx))),df0dx);

%{
 ******************************
 fval:  CV with values of constraint functions f_i, calculated at xval.
 dfdx:  [m x n]-matrix with derivatives of constraint functions f_i with 
        respect to variables x_j, calculated at xval.
        dfdx(i,j) = derivative of f_i with respect to x_j.
 dfdx2: [m x n]-matrix with non-mixed second derivatives of constraint 
        functions f_i with respect to variables x_j, calculated at xval. 
        dfdx2(i,j) = second derivative of f_i with respect to x_j (twice). 
        Important note: If not available, simply let dfdx2 = 0*dfdx.
 ******************************
%}

% Volume restriction
fval(1,:)  = 0.5*(V-1)^2;          %Effective whenever >0 -> V_real>V_desired
dfdx(1,:)  = dV;
dfdx2(1,:) = 0*dfdx(1,:);


% second restriction
%fval(2)    = ;
%dfdx(2,:)  = ;
%dfdx2(2,:) = 0*dfdx;

m = size(fval,1);     %Nº of general constraints.
n = size(xval,1);     %Nº of design variables (x_j = MNESA).

een   = ones(n,1);    %Auxiliar functions.
eem   = ones(m,1);
zeron = zeros(n,1);
zerom = zeros(m,1);

x_min = 1e-3*een;        %Lower bound for variables x_j.
x_max = een;             %Upper bound for variables x_j.
xmami = x_max-x_min;     %Range of admissible design variables.
xmamieps = 0.00001*een;  %Minimum x_range.

a0 = 1;               %Constants a_0 in a_0*z.
a  = 0*eem;           %CV with constants a_i in a_i*z.
c  = 10000*eem;       %CV with constants c_i in c_i*y_i.
d  = 1*eem;           %CV with constants d_i in 0.5*d_i*(y_i)^2.

% Factors for further calculations
%-------------------------------------------------------------------------
% L,U
init = 0.5;   %Applied when iter <= 2.
incr = 1.2;   %Applied when iter > 2. incr = [1.1, 1.3]
decr = 0.7;   %Applied when iter > 2. decr = [0.5, 0.7]

%Important note:
%incr*decr < 1, so that one single "relaxation" of the asymptotes
%never fully compensates for one "tightening".

% Alfa, Beta
move = 1;       %Factor for calculation of alfa and beta.
albefa = 0.1;     %Factor for calculation of alfa and beta.

% p0, q0, P, Q, b
raa0 = 0.00001;

%% MMA-ITERATION

% Calculation of the lower and upper asymptotes:
%-------------------------------------------------------------------------
if iter < 2.5
  L = xval - init*(xmami);      %CV with the lower asymptotes
  U = xval + init*(xmami);      %CV with the upper asymptotes
  
else
  zzz = (xval-xold1).*(xold1-xold2);  %Aux. for checking opt. behavior.
  factor = een;
  factor(find(zzz > 0)) = incr;
  factor(find(zzz < 0)) = decr;
  
  L = xval - factor.*(xold1 - L);
  Lmin = xval - 10*(xmami);
  Lmax = xval - 0.01*(xmami);
  L = min(max(L,Lmin),Lmax);          %CV with the lower asymptotes.
  
  U = xval + factor.*(U - xold1);
  Umin = xval + 0.01*(xmami);
  Umax = xval + 10*(xmami);
  U = max(min(U,Umax),Umin);          %CV with the upper asymptotes.
  
end

% Calculation of the bounds alfa and beta:
%-------------------------------------------------------------------------
zzz1 = L + albefa*(xval-L);
zzz2 = xval - move*(xmami);
alfa = max(max(zzz1,zzz2),x_min);

zzz1 = U - albefa*(U-xval);
zzz2 = xval + move*(xmami);
beta = min(min(zzz1,zzz2),x_max);

% Calculation of p0, q0, P, Q and b:
%-------------------------------------------------------------------------
xmami = max(xmami,xmamieps);
xmamiinv = een./xmami;
ux1 = U-xval;
ux2 = ux1.*ux1;
xl1 = xval-L;
xl2 = xl1.*xl1;
uxinv = een./ux1;
xlinv = een./xl1;

p0 = zeron;
q0 = zeron;
p0 = max(df0dx,0);
q0 = max(-df0dx,0);
%p0(find(df0dx > 0)) =  df0dx(find(df0dx > 0));
%q0(find(df0dx < 0)) = -df0dx(find(df0dx < 0));
pq0 = 0.001*(p0 + q0) + raa0*xmamiinv;
p0 = p0 + pq0;
q0 = q0 + pq0;
p0 = p0.*ux2;
q0 = q0.*xl2;

P = sparse(m,n);
Q = sparse(m,n);
P = max(dfdx,0);
Q = max(-dfdx,0);
%P(find(dfdx > 0)) =  dfdx(find(dfdx > 0));
%Q(find(dfdx < 0)) = -dfdx(find(dfdx < 0));
PQ = 0.001*(P + Q) + raa0*eem*xmamiinv.';
P = P + PQ;
Q = Q + PQ;
P = P * spdiags(ux2,0,n,n);
Q = Q * spdiags(xl2,0,n,n);
b = P*uxinv + Q*xlinv - fval(:) ;

%% SUBPROBLEM (primal-dual Newton method)
%{
  This section solves the MMA subproblem:
         
    minimize  SUM[p0j/(Uj-xj) + q0j/(xj-Lj)  + a0*z +
              + SUM[ci*yi + 0.5*di*(yi)^2]
 
  subject to  SUM[pij/(Uj-xj) + qij/(xj-Lj)] - ai*z - yi <= bi,
              alfaj <=  xj <=  betaj,  yi >= 0,  z >= 0.
%}
%epsimin = sqrt(m+n)*10^(-9);
epsimin = 10^(-7); 
epsi = 1;             % TOLERANCE
epsvecn = epsi*een;
epsvecm = epsi*eem;
x = 0.5*(alfa+beta);
y = eem;
z = 1;
lam = eem;
xsi = een./(x-alfa);
xsi = max(xsi,een);
eta = een./(beta-x);
eta = max(eta,een);
mu  = max(eem,0.5*c);
zet = 1;
s = eem;
itera = 0;

while epsi > epsimin
  epsvecn = epsi*een;
  epsvecm = epsi*eem;
  ux1 = U-x;
  xl1 = x-L;
  ux2 = ux1.*ux1;
  xl2 = xl1.*xl1;
  uxinv1 = een./ux1;
  xlinv1 = een./xl1;

  plam = p0 + P'*lam ;
  qlam = q0 + Q'*lam ;
  gvec = P*uxinv1 + Q*xlinv1;
  dpsidx = plam./ux2 - qlam./xl2 ;

  rex = dpsidx - xsi + eta;
  rey = c + d.*y - mu - lam;
  rez = a0 - zet - a'*lam;
  relam = gvec - a*z - y + s - b;
  rexsi = xsi.*(x-alfa) - epsvecn;
  reeta = eta.*(beta-x) - epsvecn;
  remu = mu.*y - epsvecm;
  rezet = zet*z - epsi;
  res = lam.*s - epsvecm;

  residu1 = [rex' rey' rez']';
  residu2 = [relam' rexsi' reeta' remu' rezet res']';
  residu = [residu1' residu2']';
  residunorm = sqrt(residu'*residu);
  residumax = max(abs(residu));

  ittt = 0;
  while residumax > 0.9*epsi & ittt < 100
    ittt=ittt + 1;
    itera=itera + 1;

    ux1 = U-x;
    xl1 = x-L;
    ux2 = ux1.*ux1;
    xl2 = xl1.*xl1;
    ux3 = ux1.*ux2;
    xl3 = xl1.*xl2;
    uxinv1 = een./ux1;
    xlinv1 = een./xl1;
    uxinv2 = een./ux2;
    xlinv2 = een./xl2;
    plam = p0 + P'*lam ;
    qlam = q0 + Q'*lam ;
    gvec = P*uxinv1 + Q*xlinv1;
    GG = P*spdiags(uxinv2,0,n,n) - Q*spdiags(xlinv2,0,n,n);
    dpsidx = plam./ux2 - qlam./xl2 ;
    delx = dpsidx - epsvecn./(x-alfa) + epsvecn./(beta-x);
    dely = c + d.*y - lam - epsvecm./y;
    delz = a0 - a'*lam - epsi/z;
    dellam = gvec - a*z - y - b + epsvecm./lam;
    diagx = plam./ux3 + qlam./xl3;
    diagx = 2*diagx + xsi./(x-alfa) + eta./(beta-x);
    diagxinv = een./diagx;
    diagy = d + mu./y;
    diagyinv = eem./diagy;
    diaglam = s./lam;
    diaglamyi = diaglam+diagyinv;

    if m < n
      blam = dellam + dely./diagy - GG*(delx./diagx);
      bb = [blam' delz']';
      Alam = spdiags(diaglamyi,0,m,m) + GG*spdiags(diagxinv,0,n,n)*GG';
      AA = [Alam     a
            a'    -zet/z ];
      solut = AA\bb;
      dlam = solut(1:m);
      dz = solut(m+1);
      dx = -delx./diagx - (GG'*dlam)./diagx;
    else
      diaglamyiinv = eem./diaglamyi;
      dellamyi = dellam + dely./diagy;
      Axx = spdiags(diagx,0,n,n) + GG'*spdiags(diaglamyiinv,0,m,m)*GG;
      azz = zet/z + a'*(a./diaglamyi);
      axz = -GG'*(a./diaglamyi);
      bx = delx + GG'*(dellamyi./diaglamyi);
      bz  = delz - a'*(dellamyi./diaglamyi);
      AA = [Axx   axz
            axz'  azz ];
      bb = [-bx' -bz]';
      solut = AA\bb;
      dx  = solut(1:n);
      dz = solut(n+1);
      dlam = (GG*dx)./diaglamyi - dz*(a./diaglamyi) + dellamyi./diaglamyi;
    end

    dy = -dely./diagy + dlam./diagy;
    dxsi = -xsi + epsvecn./(x-alfa) - (xsi.*dx)./(x-alfa);
    deta = -eta + epsvecn./(beta-x) + (eta.*dx)./(beta-x);
    dmu  = -mu + epsvecm./y - (mu.*dy)./y;
    dzet = -zet + epsi/z - zet*dz/z;
    ds   = -s + epsvecm./lam - (s.*dlam)./lam;
    xx  = [ y'  z  lam'  xsi'  eta'  mu'  zet  s']';
    dxx = [dy' dz dlam' dxsi' deta' dmu' dzet ds']';
    
    stepxx = -1.01*dxx./xx;
    stmxx  = max(stepxx);
    stepalfa = -1.01*dx./(x-alfa);
    stmalfa = max(stepalfa);
    stepbeta = 1.01*dx./(beta-x);
    stmbeta = max(stepbeta);
    stmalbe  = max(stmalfa,stmbeta);
    stmalbexx = max(stmalbe,stmxx);
    stminv = max(stmalbexx,1);
    steg = 1/stminv;

    xold   = x;
    yold   = y;
    zold   = z;
    lamold = lam;
    xsiold = xsi;
    etaold = eta;
    muold  = mu;
    zetold = zet;
    sold   = s;

    itto = 0;
    resinew = 2*residunorm;
    while resinew > residunorm & itto < 50
    itto = itto+1;

    x   =   xold + steg*dx;
    y   =   yold + steg*dy;
    z   =   zold + steg*dz;
    lam = lamold + steg*dlam;
    xsi = xsiold + steg*dxsi;
    eta = etaold + steg*deta;
    mu  =  muold + steg*dmu;
    zet = zetold + steg*dzet;
    s   =   sold + steg*ds;
    ux1 = U-x;
    xl1 = x-L;
    ux2 = ux1.*ux1;
    xl2 = xl1.*xl1;
    uxinv1 = een./ux1;
    xlinv1 = een./xl1;
    plam = p0 + P'*lam ;
    qlam = q0 + Q'*lam ;
    gvec = P*uxinv1 + Q*xlinv1;
    dpsidx = plam./ux2 - qlam./xl2 ;

    rex = dpsidx - xsi + eta;
    rey = c + d.*y - mu - lam;
    rez = a0 - zet - a'*lam;
    relam = gvec - a*z - y + s - b;
    rexsi = xsi.*(x-alfa) - epsvecn;
    reeta = eta.*(beta-x) - epsvecn;
    remu = mu.*y - epsvecm;
    rezet = zet*z - epsi;
    res = lam.*s - epsvecm;

    residu1 = [rex' rey' rez]';
    residu2 = [relam' rexsi' reeta' remu' rezet res']';
    residu = [residu1' residu2']';
    resinew = sqrt(residu'*residu);
    steg = steg/2;
    end
  residunorm=resinew;
  residumax = max(abs(residu));
  steg = 2*steg;
  end
  if ittt > 198
      epsi;
      ittt;
  end
epsi = 0.1*epsi;
end

% Output of the original function
%-------------------------------------------------------------------------
x_mma   = x;    %CV with the optimal values of the variables x_j
                %in the current MMA subproblem.
y_mma   = y;    %CV with the optimal values of the variables y_i
                %in the current MMA subproblem.
z_mma   = z;    %Scalar with the optimal value of the variable z
                %in the current MMA subproblem.
lam_mma = lam;  %Lagrange mult. for the m general MMA constraints.
xsi_mma = xsi;  %Lagrange mult. for the n constraints alfa_j - x_j <= 0.
eta_mma = eta;  %Lagrange mult. for the n constraints x_j - beta_j <= 0.
mu_mma  = mu;   %Lagrange mult. for the m constraints -y_i <= 0.
zet_mma = zet;  %Lagrange mult. for the single constraint -z <= 0.
s_mma   = s;    %Slack variables for the m general MMA constraints.

% Update of design variables and asymptotes
%-------------------------------------------------------------------------
New_Rel_Density = x_mma;

%% CREDITS
%{
 Written in May 1999 and modified in August 2007 by
 Krister Svanberg <krille@math.kth.se>
 Department of Mathematics
 SE-10044 Stockholm, Sweden.

 Modified in January 2018 by
 David Muñoz Pellicer <damuopel@upv.es>
 Department of Mechanical and Materials Engineering
 Camí de Vera, s/n, Valencia, Spain.

 Modified in April 2021 by
 Marc Bosch Galera <marbosga@etsid.upv.es>
 Department of Mechanical and Materials Engineering
 Camí de Vera, s/n, Valencia, Spain.
%}
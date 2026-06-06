function thesis_zero_crossing_validation_N2000_short()


clc; close all;

%% -------------------- model parameters --------------------
par.k = 6;
par.z = 0.10;
par.u = 1 - par.z;
par.mu = 1e-3;
par.N = 2000;
par.pZZ = par.z^2;
par.tol_raw = 1e-10;
par.fdY = 1e-6;
par.tiny = 1e-14;
par.max_pass = 50;
par.pass_time = 50*par.N;
par.max_step = 5*par.N;
par.coeff = make_coeff(par.k);

base.R = 1.20;
base.S = -0.40;
base.T = 1.60;
base.P = 0.00;

nSweep = 41;
sweepSpan = 1.0;
beta_small = [0.0025 0.005 0.01];

%% -------------------- neutral steady state --------------------
x_init = 0.50;
pC_init = par.u*x_init;
Yinit = sanitize([x_init; pC_init^2; par.z*pC_init],par);

par0 = set_payoffs(par,0,0,0,0);
[Y0,ok0,F0] = solve_ss(0,Yinit,par0);
x0 = Y0(1);

J = num_jac(Y0,par0);
[U,V] = structural_vectors(Y0,par);

% Structural coefficients
Amu = -[1 0 0]*(J\U);
Bmu = -[1 0 0]*(J\V);

%% -------------------- choose payoff entry and sweep --------------------
% Sweep the payoff entry with the stronger structural coefficient.
if abs(Bmu) >= abs(Amu)
    sweepName = 'S';
    cross_pred = base.P - (Amu/Bmu)*(base.R-base.T);
    sweep = linspace(cross_pred-sweepSpan,cross_pred+sweepSpan,nSweep);
    theory = Amu*(base.R-base.T) + Bmu*(sweep-base.P);
    xlab = 'Swept payoff entry S';
else
    sweepName = 'T';
    cross_pred = base.R + (Bmu/Amu)*(base.S-base.P);
    sweep = linspace(cross_pred-sweepSpan,cross_pred+sweepSpan,nSweep);
    theory = Amu*(base.R-sweep) + Bmu*(base.S-base.P);
    xlab = 'Swept payoff entry T';
end

%% -------------------- numerical weak-selection slopes --------------------
slope = zeros(length(beta_small),nSweep);
fit_slope = zeros(1,nSweep);
conv_count = 0;
max_res = 0;

for i = 1:nSweep
    if strcmp(sweepName,'S')
        p = set_payoffs(par,base.R,sweep(i),base.T,base.P);
    else
        p = set_payoffs(par,base.R,base.S,sweep(i),base.P);
    end

    Yseed = Y0;
    for b = 1:length(beta_small)
        beta = beta_small(b);
        [Yss,ok,res] = solve_ss(beta,Yseed,p);
        slope(b,i) = (Yss(1)-x0)/beta;
        Yseed = Yss;
        conv_count = conv_count + ok;
        max_res = max(max_res,res);
    end

    y = beta_small(:).*slope(:,i);
    fit_slope(i) = sum(beta_small(:).*y)/sum(beta_small(:).^2);
end

num_cross = zero_cross(sweep,fit_slope);
the_cross = zero_cross(sweep,theory);

%% -------------------- print short summary --------------------
fprintf('\nZero-crossing validation with N = %d\n',par.N);
fprintf('k = %d, z = %.2f, mu = %.1e\n',par.k,par.z,par.mu);
fprintf('Neutral convergence = %d, raw residual = %.3e\n',ok0,F0);
fprintf('x0_mu = %.10f\n',x0);
fprintf('A_mu = %.10e, B_mu = %.10e\n',Amu,Bmu);
fprintf('Sweep entry = %s\n',sweepName);
fprintf('Predicted crossing = %.10f\n',cross_pred);
fprintf('Theory crossing     = %.10f\n',the_cross);
fprintf('Numerical crossing  = %.10f\n',num_cross);
fprintf('Converged slope runs = %d / %d\n',conv_count,numel(slope));
fprintf('Largest raw residual = %.3e\n\n',max_res);

%% -------------------- plot --------------------
figure('Color','w','Position',[100 100 1150 560]);
hold on; box on; grid on;

xLimits = [min(sweep) max(sweep)];
yLimits = [-0.2 0.2];

plot(sweep,theory,'k-','LineWidth',2.5);
plot(sweep,slope(1,:),'b--','LineWidth',2);
plot(sweep,slope(2,:),'r-.','LineWidth',2);
plot(sweep,slope(3,:),'g:','LineWidth',2.0);

xlim(xLimits);
ylim(yLimits);

plot(xLimits,[0 0],'k:','LineWidth',1.2);
plot([cross_pred cross_pred],yLimits,'--','Color',[0.35 0.35 0.35],'LineWidth',1.4);

xlabel(xlab,'FontWeight','bold');
ylabel('Weak-selection slope / exact predictor','FontWeight','bold');

legend({'Theory','Slope, \beta = 0.0025','Slope, \beta = 0.005', ...
        'Slope, \beta = 0.01','Zero line','Predicted crossing'}, ...
        'Location','eastoutside');

set(gca,'FontName','Times New Roman','FontSize',18);
saveas(gcf,'zero_crossing_validation_N2000_short.png');

end

%% ==========================================================
function [Y,ok,res] = solve_ss(beta,Y0,p)

opts = odeset('RelTol',1e-9,'AbsTol',1e-11,'MaxStep',p.max_step);
Y = sanitize(Y0,p);
ok = false;
res = Inf;

for pass = 1:p.max_pass
    [~,Ysol] = ode15s(@(t,y) rhs(y,beta,p),[0 p.pass_time],Y,opts);
    Y = sanitize(Ysol(end,:).',p);

    % Convergence uses the unscaled RHS, not the RHS divided by N.
    res = norm(rhs_raw(Y,beta,p),inf);
    if res < p.tol_raw
        ok = true;
        return;
    end
end

end

%% ==========================================================
function F = rhs(Y,beta,p)
% Actual ODE RHS with N written explicitly.

F0 = rhs_raw(Y,beta,p);
F = [F0(1)/p.N; F0(2)/p.N; F0(3)/p.N];

end

%% ==========================================================
function F = rhs_raw(Y,beta,p)
% Unscaled RHS. This is also used for steady-state residual checking.

Y = sanitize(Y,p);
x = Y(1);
pCC = Y(2);
pZC = Y(3);

k = p.k;
z = p.z;
u = p.u;
mu = p.mu;

pC = u*x;
pD = u*(1-x);
pCD = max(pC-pCC-pZC,0);
pZD = max(z-p.pZZ-pZC,0);

[qC_C,qZ_C,qD_C] = cond_probs(pCC,pZC,pC,p.tiny);
[qC_D,qZ_D,qD_D] = cond_probs(pCD,pZD,pD,p.tiny);

Tplus = 0;
Tminus = 0;
GCCp = 0;
GCCm = 0;
GZCp = 0;
GZCm = 0;

for kZ = 0:k
    for kC = 0:(k-kZ)
        kD = k-kZ-kC;
        c = p.coeff(kZ+1,kC+1);

        A = (kZ+kC)/k;
        B = kD/k;
        PiC = A*p.R + B*p.S;
        PiD = A*p.T + B*p.P;
        Delta = PiC - PiD;

        fplus = fermi(beta*Delta);
        fminus = fermi(-beta*Delta);

        phip = mu + (1-mu)*A*fplus;
        phim = mu + (1-mu)*B*fminus;

        probD = c*(qZ_D^kZ)*(qC_D^kC)*(qD_D^kD);
        Tplus = Tplus + probD*phip;
        GCCp = GCCp + probD*kC*phip;
        GZCp = GZCp + probD*kZ*phip;

        probC = c*(qZ_C^kZ)*(qC_C^kC)*(qD_C^kD);
        Tminus = Tminus + probC*phim;
        GCCm = GCCm + probC*kC*phim;
        GZCm = GZCm + probC*kZ*phim;
    end
end

F = [((1-x)*Tplus - x*Tminus); ...
     2*(pD*GCCp - pC*GCCm)/k; ...
     2*(pD*GZCp - pC*GZCm)/k];

end

%% ==========================================================
function [U,V] = structural_vectors(Y0,p)

k = p.k;
mu = p.mu;
N = p.N;
x = Y0(1);
pCC = Y0(2);
pZC = Y0(3);

pC = p.u*x;
pD = p.u*(1-x);
pCD = max(pC-pCC-pZC,0);
pZD = max(p.z-p.pZZ-pZC,0);

[cC,zC,r0] = cond_probs(pCC,pZC,pC,p.tiny);
[cD,zD,q0] = cond_probs(pCD,pZD,pD,p.tiny);

ED_A2 = (1-q0)^2 + q0*(1-q0)/k;
ED_AB = q0*(1-q0)*(1-1/k);
EC_AB = r0*(1-r0)*(1-1/k);
EC_B2 = r0^2 + r0*(1-r0)/k;

FD = k - ((k-1)*(2*k-1)/k)*q0 + ((k-1)*(k-2)/k)*q0^2;
HD = ((k-1)*q0/k)*((k-1)-(k-2)*q0);
FC = ((k-1)*r0/k)*((k-1)-(k-2)*r0);
HC = ((k-1)*r0/k)*(1+(k-2)*r0);

U = [(1-mu)*((1-x)*ED_A2 + x*EC_AB)/(4*N); ...
     (1-mu)*(pD*cD*FD + pC*cC*FC)/(2*k*N); ...
     (1-mu)*(pD*zD*FD + pC*zC*FC)/(2*k*N)];

V = [(1-mu)*((1-x)*ED_AB + x*EC_B2)/(4*N); ...
     (1-mu)*(pD*cD*HD + pC*cC*HC)/(2*k*N); ...
     (1-mu)*(pD*zD*HD + pC*zC*HC)/(2*k*N)];

end

%% ==========================================================
function J = num_jac(Y,p)

n = length(Y);
J = zeros(n,n);

for j = 1:n
    h = p.fdY*max(1,abs(Y(j)));
    e = zeros(n,1);
    e(j) = 1;

    Yp = sanitize(Y+h*e,p);
    Ym = sanitize(Y-h*e,p);
    den = Yp(j)-Ym(j);
    if abs(den) < 1e-16
        den = 2*h;
    end

    J(:,j) = (rhs(Yp,0,p)-rhs(Ym,0,p))/den;
end

end

%% ==========================================================
function p = set_payoffs(p,R,S,T,P)

p.R = R;
p.S = S;
p.T = T;
p.P = P;

end

%% ==========================================================
function y = fermi(a)

if a >= 0
    y = 1/(1+exp(-a));
else
    ea = exp(a);
    y = ea/(1+ea);
end

end

%% ==========================================================
function [qC,qZ,qD] = cond_probs(pairC,pairZ,total,tiny)

if total > tiny
    qC = max(pairC/total,0);
    qZ = max(pairZ/total,0);
    s = qC+qZ;
    if s > 1
        qC = qC/s;
        qZ = qZ/s;
        qD = 0;
    else
        qD = 1-s;
    end
else
    qC = 0;
    qZ = 0;
    qD = 1;
end

end

%% ==========================================================
function Y = sanitize(Y,p)

epsx = 1e-12;
x = min(max(Y(1),epsx),1-epsx);
pC = p.u*x;

pZC = min(max(Y(3),0),min(pC,max(p.z-p.pZZ,0)));
pCC = min(max(Y(2),0),max(pC-pZC,0));

Y = [x; pCC; pZC];

end

%% ==========================================================
function coeff = make_coeff(k)

coeff = zeros(k+1,k+1);
for kZ = 0:k
    for kC = 0:(k-kZ)
        coeff(kZ+1,kC+1) = nchoosek(k,kZ)*nchoosek(k-kZ,kC);
    end
end

end

%% ==========================================================
function xc = zero_cross(x,y)

xc = NaN;
for i = 1:length(x)-1
    if y(i) == 0
        xc = x(i);
        return;
    elseif y(i)*y(i+1) < 0
        xc = x(i)-y(i)*(x(i+1)-x(i))/(y(i+1)-y(i));
        return;
    end
end

end
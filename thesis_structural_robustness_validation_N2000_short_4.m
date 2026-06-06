function thesis_structural_robustness_validation_N2000_short()


clc; clear; close all;

%% -------------------- main settings --------------------
base.k  = 6;
base.z  = 0.10;
base.mu = 1e-3;

k_values  = [4 6 8];
z_values  = [0.05 0.10 0.20];
mu_values = [1e-4 1e-3 1e-2];

game.R = 1.00;
game.S = 0.20;
game.P = -0.20;

beta_small = [0.001 0.002 0.004];
sweepSpan  = 0.10;
nSweep     = 7;

colors = [0.000 0.447 0.741; ...
          0.850 0.325 0.098; ...
          0.000 0.550 0.180];
markers = {'o','s','d'};

figure('Color','w','Position',[80 120 1650 650]);

fprintf('\nStructural robustness validation with N = 2000\n');
fprintf('Fixed game: R = %.2f, S = %.2f, P = %.2f, swept payoff = T\n', ...
    game.R,game.S,game.P);

make_panel(1,k_values ,'k' ,base,game,beta_small,sweepSpan,nSweep,colors,markers);
make_panel(2,z_values ,'z' ,base,game,beta_small,sweepSpan,nSweep,colors,markers);
make_panel(3,mu_values,'mu',base,game,beta_small,sweepSpan,nSweep,colors,markers);

saveas(gcf,'structural_robustness_validation_N2000_short.png');

end

%% ==========================================================
function make_panel(panel_id,values,mode,base,game,beta_small,sweepSpan,nSweep,colors,markers)

panel_pos = [0.060 0.170 0.205 0.700; ...
             0.320 0.170 0.205 0.700; ...
             0.580 0.170 0.205 0.700];

legend_pos = [0.830 0.695 0.145 0.205; ...
              0.830 0.395 0.145 0.205; ...
              0.830 0.095 0.145 0.205];

caption_pos = [0.830 0.905 0.145 0.035; ...
               0.830 0.605 0.145 0.035; ...
               0.830 0.305 0.145 0.035];

axes('Position',panel_pos(panel_id,:)); hold on; box on; grid on;
set(gca,'FontName','Times New Roman','FontSize',10,'LineWidth',1.0);

h = [];
leg = {};

for i = 1:length(values)
    if strcmp(mode,'k')
        p = make_par(values(i),base.z,base.mu);
        tag = sprintf('k = %d',p.k);
        legtag = sprintf('%d',p.k);
    elseif strcmp(mode,'z')
        p = make_par(base.k,values(i),base.mu);
        tag = sprintf('z = %.2f',p.z);
        legtag = sprintf('%.2f',p.z);
    else
        p = make_par(base.k,base.z,values(i));
        tag = ['\mu = ' sprintf('%.0e',p.mu)];
        legtag = sprintf('%.0e',p.mu);
    end

    res = validate_one(p,game,beta_small,sweepSpan,nSweep);

    h1 = plot(res.xplot,res.theory,'-','Color',colors(i,:),'LineWidth',2.3);
    h2 = plot(res.xplot,res.slope,'LineStyle','none','Marker',markers{i}, ...
        'MarkerSize',6,'MarkerEdgeColor',colors(i,:),'MarkerFaceColor','w','LineWidth',1.3);

    h = [h h1 h2]; %#ok<AGROW>
    leg{end+1} = [legtag ' theory']; %#ok<AGROW>
    leg{end+1} = [legtag ' numeric']; %#ok<AGROW>

    fprintf('%s: A_mu = %.4e, B_mu = %.4e, theory T* = %.6f, numeric T* = %.6f\n', ...
        tag,res.Amu,res.Bmu,safe_num(res.crossTheory),safe_num(res.crossNum));
end

if strcmp(mode,'z')
    xlim([-0.15 0.15]);
    set(gca,'XTick',-0.15:0.05:0.15);
end

xl = xlim;
yl = ylim;
plot(xl,[0 0],'k:','LineWidth',1.1);
plot([0 0],yl,'--','Color',[0.35 0.35 0.35],'LineWidth',1.1);
xlim(xl); ylim(yl);

xlabel('T - T^*_{theory}', ...
    'FontWeight','bold','Interpreter','tex','FontSize',20);

if panel_id == 1
    ylabel('Weak-selection slope','FontWeight','bold','FontSize',20);
else
    ylabel('');
end

if strcmp(mode,'k')
    title({'Degree robustness','Varying k'},'FontWeight','bold','Interpreter','tex','FontSize',18 );
    legend_caption = 'k';
elseif strcmp(mode,'z')
    title({'Zealot-fraction robustness','Varying z'},'FontWeight','bold','Interpreter','tex','FontSize',18 );
    legend_caption = 'z';
else
    title({'Mutation-rate robustness','Varying \mu'},'FontWeight','bold','Interpreter','tex','FontSize',18 );
    legend_caption = '\mu';
end

lgd = legend(h,leg,'Interpreter','tex','FontSize',14);
set(lgd,'Position',legend_pos(panel_id,:));

annotation('textbox',caption_pos(panel_id,:), ...
    'String',legend_caption, ...
    'EdgeColor','none', ...
    'HorizontalAlignment','center', ...
    'FontWeight','bold', ...
    'FontSize',14, ...
    'Interpreter','tex');

end

%% ==========================================================
function res = validate_one(p,game,beta_small,sweepSpan,nSweep)

xinit = 0.50;
pC0 = p.u*xinit;
Yinit = sanitize([xinit; pC0^2; p.z*pC0],p);

p0 = set_payoffs(p,0,0,0,0);
Y0 = solve_ss(0,Yinit,p0);
x0 = Y0(1);

J = num_jac(Y0,p0);
[U,V] = structural_vectors(Y0,p);

Amu = -[1 0 0]*(J\U);
Bmu = -[1 0 0]*(J\V);

Tstar = game.R + (Bmu/Amu)*(game.S-game.P);
Tvec = linspace(Tstar-sweepSpan,Tstar+sweepSpan,nSweep);

theory = zeros(1,nSweep);
slope  = zeros(1,nSweep);
YseedT = Y0;

for i = 1:nSweep
    Tcur = Tvec(i);
    pG = set_payoffs(p,game.R,game.S,Tcur,game.P);
    theory(i) = Amu*(game.R-Tcur) + Bmu*(game.S-game.P);

    local = zeros(length(beta_small),1);
    Yseed = YseedT;

    for b = 1:length(beta_small)
        beta = beta_small(b);
        Yss = solve_ss(beta,Yseed,pG);
        local(b) = (Yss(1)-x0)/beta;
        Yseed = Yss;
    end

    slope(i) = mean(local);
    YseedT = Yseed;
end

res.Amu = Amu;
res.Bmu = Bmu;
res.xplot = Tvec - Tstar;
res.theory = theory;
res.slope = slope;
res.crossTheory = zero_cross(Tvec,theory);
res.crossNum = zero_cross(Tvec,slope);

end

%% ==========================================================
function p = make_par(k,z,mu)

p.k = k;
p.z = z;
p.u = 1-z;
p.mu = mu;
p.N = 2000;
p.pZZ = z^2;
p.tol_raw = 1e-10;
p.fdY = 1e-6;
p.tiny = 1e-14;
p.max_pass = 40;
p.pass_time = 50*p.N;
p.max_step = 5*p.N;
p.coeff = make_coeff(k);

end

%% ==========================================================
function Y = solve_ss(beta,Y0,p)

opts = odeset('RelTol',1e-9,'AbsTol',1e-11,'MaxStep',p.max_step);
Y = sanitize(Y0,p);

for pass = 1:p.max_pass
    [~,Ysol] = ode15s(@(t,y) rhs(y,beta,p),[0 p.pass_time],Y,opts);
    Y = sanitize(Ysol(end,:).',p);

    % Check convergence by unscaled RHS, not by RHS divided by N.
    if norm(rhs_raw(Y,beta,p),inf) < p.tol_raw
        return;
    end
end

warning('Steady state did not fully converge.');

end

%% ==========================================================
function F = rhs(Y,beta,p)
% Actual ODE RHS with N written explicitly.

F0 = rhs_raw(Y,beta,p);
F = [F0(1)/p.N; F0(2)/p.N; F0(3)/p.N];

end

%% ==========================================================
function F = rhs_raw(Y,beta,p)
% Unscaled RHS used for residual checking.

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

    % Jacobian of the scaled RHS. U and V also include 1/N.
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

%% ==========================================================
function out = safe_num(x)

if isnan(x)
    out = 0;
else
    out = x;
end

end
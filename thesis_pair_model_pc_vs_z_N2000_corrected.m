function thesis_pair_model_pc_vs_z_N2000()


clear; clc; close all;

%% -------------------- parameters --------------------
par.N  = 2000;
par.k  = 6;
par.mu = 0.0;      % change to 0.00 or 0.20 if needed
par.x0 = 0.40;

par.rhs_tol   = 1e-8;
par.tail_tol  = 1e-7;
par.bound_tol = 1e-8;
par.tiny      = 1e-14;

par.pass_time = 400*par.N;
par.max_step  = 10*par.N;
par.max_pass  = 120;
par.min_pass  = 3;
par.coeff     = make_coeff(par.k);

z_vec    = linspace(0,0.4,16);
beta_vec = [0.1 0.2 1 2 5 10];

games = {
    struct('name','PD','R',1.0,'S',-0.5,'T',2.0,'P',0.0)
    struct('name','SH','R',1.0,'S',-0.5,'T',0.5,'P',0.0)
    struct('name','CG','R',1.0,'S', 0.5,'T',2.0,'P',0.0)
};

curve_colors = [
    0.0000 0.4470 0.7410
    0.8500 0.3250 0.0980
    0.9290 0.6940 0.1250
    0.4940 0.1840 0.5560
    0.4660 0.6740 0.1880
    0.3010 0.7450 0.9330
];

legend_text = cell(length(beta_vec),1);
for bi = 1:length(beta_vec)
    legend_text{bi} = sprintf('\\beta = %.2g',beta_vec(bi));
end

%% -------------------- figure setup --------------------
fig = figure('Color','w','Position',[100 100 1500 470]);
ax_pos = [0.060 0.160 0.240 0.720;
          0.340 0.160 0.240 0.720;
          0.620 0.160 0.240 0.720];

n_rhs   = 0;
n_tail  = 0;
n_bound = 0;
n_slow  = 0;

%% -------------------- main computation --------------------
for g = 1:length(games)
    ax = axes('Position',ax_pos(g,:));
    hold(ax,'on'); box(ax,'on'); grid(ax,'on');

    for bi = 1:length(beta_vec)
        beta = beta_vec(bi);
        pc_z = zeros(size(z_vec));

        for zi = 1:length(z_vec)
            z = z_vec(zi);
            [pc_z(zi),method] = steady_pc_ode15s(beta,z,games{g},par);

            if method == 1
                n_rhs = n_rhs + 1;
            elseif method == 2
                n_tail = n_tail + 1;
            elseif method == 3
                n_bound = n_bound + 1;
            else
                n_slow = n_slow + 1;
            end
        end

        plot(ax,z_vec,pc_z,'o-', ...
            'LineWidth',2.0, ...
            'MarkerSize',5, ...
            'Color',curve_colors(bi,:), ...
            'MarkerFaceColor',curve_colors(bi,:), ...
            'MarkerEdgeColor',curve_colors(bi,:));
    end
xlabel(ax,'Zealot fraction (z)','FontWeight','bold','FontSize',18);

if g == 1
    ylabel(ax,'Fraction of cooperators (p_C^*)', ...
        'FontWeight','bold','FontSize',18,'Interpreter','tex');
end

title(ax,sprintf('%s',games{g}.name), ...
    'FontWeight','bold','FontSize',17,'Interpreter','tex');
ylim(ax,[-0.05 1.05]);
set(ax,'YTick',0:0.2:1);
set(ax,'FontSize',18);
   
end

%% -------------------- single legend --------------------
leg_ax = axes('Position',[0.885 0.280 0.100 0.440],'Visible','off');
hold(leg_ax,'on');

hleg = zeros(length(beta_vec),1);
for bi = 1:length(beta_vec)
    hleg(bi) = plot(leg_ax,nan,nan,'o-', ...
        'LineWidth',2.0, ...
        'MarkerSize',5, ...
        'Color',curve_colors(bi,:), ...
        'MarkerFaceColor',curve_colors(bi,:), ...
        'MarkerEdgeColor',curve_colors(bi,:));
end

legend(leg_ax,hleg,legend_text, ...
    'Location','northwest', ...
    'FontSize',16, ...
    'Box','on', ...
    'Interpreter','tex');

%% -------------------- output summary --------------------
fprintf('\nSteady-state acceptance summary\n');
fprintf('Raw RHS accepted points       = %d\n',n_rhs);
fprintf('Terminal-tail accepted points = %d\n',n_tail);
fprintf('Boundary accepted points      = %d\n',n_bound);
fprintf('Maximum-pass final points     = %d\n',n_slow);
if n_slow == 0
    fprintf('All points accepted.\n');
else
    fprintf('Some points are slow. Increase max_pass/pass_time if needed.\n');
end

safe_save(fig,'pair_model_pc_vs_z_N2000_corrected.png');

end

%% ==========================================================
function [pc,method] = steady_pc_ode15s(beta,z,game,par)
% method = 1: raw RHS norm small
% method = 2: terminal tail stable
% method = 3: absorbing boundary accepted
% method = 4: maximum pass reached

par = set_z(par,z);
par.game = game;
Y = initial_state(par);

opts = odeset('RelTol',1e-8,'AbsTol',1e-10,'MaxStep',par.max_step);
method = 4;

for pass = 1:par.max_pass
    Yold = Y;
    [~,Ysol] = ode15s(@(t,y) rhs(y,beta,par),[0 par.pass_time],Y,opts);
    Y = sanitize(Ysol(end,:).',par);

    Fraw = rhs_raw(Y,beta,par);
    raw_norm = norm(Fraw,inf);
    tail_change = max(abs(Y - Yold));

    if raw_norm < par.rhs_tol
        method = 1;
        break;
    end

    if pass >= par.min_pass && tail_change < par.tail_tol
        method = 2;
        break;
    end

    if boundary_accepted(Y,Fraw,par)
        method = 3;
        break;
    end
end

pc = par.z + par.u*Y(1);
pc = min(max(pc,0),1);

end

%% ==========================================================
function Y = initial_state(par)

pC0 = par.u*par.x0;
Y = sanitize([par.x0; pC0^2; par.z*pC0],par);

end

%% ==========================================================
function ok = boundary_accepted(Y,Fraw,par)
% Used only for mu = 0 absorbing boundary states.

x = Y(1);
xdot_raw = Fraw(1);

near_zero = x < par.bound_tol && xdot_raw <= par.rhs_tol;
near_one  = x > 1 - par.bound_tol && xdot_raw >= -par.rhs_tol;

ok = (par.mu == 0) && (near_zero || near_one);

end

%% ==========================================================
function par = set_z(par,z)

par.z   = z;
par.u   = 1 - z;
par.pZZ = z^2;

end

%% ==========================================================
function F = rhs(Y,beta,par)
% Actual ODE RHS with N written explicitly.

F0 = rhs_raw(Y,beta,par);
F = [F0(1)/par.N; F0(2)/par.N; F0(3)/par.N];

end

%% ==========================================================
function F = rhs_raw(Y,beta,par)
% Unscaled RHS used for convergence checking.

Y = sanitize(Y,par);
x = Y(1);
pCC = Y(2);
pZC = Y(3);

k = par.k;
z = par.z;
u = par.u;
mu = par.mu;
R = par.game.R;
S = par.game.S;
T = par.game.T;
P = par.game.P;

pC  = u*x;
pD  = u*(1-x);
pCD = max(pC - pCC - pZC,0);
pZD = max(z - par.pZZ - pZC,0);

[qC_C,qZ_C,qD_C] = cond_probs(pCC,pZC,pC,par.tiny);
[qC_D,qZ_D,qD_D] = cond_probs(pCD,pZD,pD,par.tiny);

Tplus = 0;
Tminus = 0;
GCCp = 0;
GCCm = 0;
GZCp = 0;
GZCm = 0;

for kZ = 0:k
    for kC = 0:(k-kZ)
        kD = k - kZ - kC;
        c  = par.coeff(kZ+1,kC+1);

        A = (kZ+kC)/k;
        B = kD/k;
        PiC = A*R + B*S;
        PiD = A*T + B*P;
        Delta = PiC - PiD;

        probD = c*(qZ_D^kZ)*(qC_D^kC)*(qD_D^kD);
        phip  = mu + (1-mu)*A*fermi(beta*Delta);
        Tplus = Tplus + probD*phip;
        GCCp  = GCCp  + probD*kC*phip;
        GZCp  = GZCp  + probD*kZ*phip;

        probC = c*(qZ_C^kZ)*(qC_C^kC)*(qD_C^kD);
        phim  = mu + (1-mu)*B*fermi(-beta*Delta);
        Tminus = Tminus + probC*phim;
        GCCm   = GCCm   + probC*kC*phim;
        GZCm   = GZCm   + probC*kZ*phim;
    end
end

xdot_raw   = ((1-x)*Tplus - x*Tminus);
pCCdot_raw = 2*(pD*GCCp - pC*GCCm)/k;
pZCdot_raw = 2*(pD*GZCp - pC*GZCm)/k;

F = [xdot_raw; pCCdot_raw; pZCdot_raw];

end

%% ==========================================================
function y = fermi(a)

if a >= 0
    y = 1/(1 + exp(-a));
else
    ea = exp(a);
    y = ea/(1 + ea);
end

end

%% ==========================================================
function [qC,qZ,qD] = cond_probs(pairC,pairZ,total,tiny)

if total > tiny
    qC = max(pairC/total,0);
    qZ = max(pairZ/total,0);
    s = qC + qZ;

    if s > 1
        qC = qC/s;
        qZ = qZ/s;
        qD = 0;
    else
        qD = 1 - s;
    end
else
    qC = 0;
    qZ = 0;
    qD = 1;
end

end

%% ==========================================================
function Y = sanitize(Y,par)

x = min(max(Y(1),0),1);
pC = par.u*x;

pZC = max(Y(3),0);
pZC = min(pZC,max(par.z-par.pZZ,0));
pZC = min(pZC,pC);

pCC = max(Y(2),0);
pCC = min(pCC,max(pC-pZC,0));

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
function safe_save(fig,filename)

if ishandle(fig)
    drawnow;
    pause(0.05);
    try
        set(fig,'PaperPositionMode','auto');
        print(fig,filename,'-dpng','-r300');
    catch
        try
            saveas(fig,filename);
        catch
            warning(['Could not save figure: ' filename]);
        end
    end
end

end

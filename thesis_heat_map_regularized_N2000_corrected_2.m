function [all_maps_pC, all_maps_x] = thesis_heat_map_regularized_N2000_corrected(beta)
% Heat-map code for the pair-approximation model with N = 2000.
clc; close all;

if nargin < 1 || isempty(beta)
    beta = 1.0;
end

%% -------------------- parameters --------------------
par.N  = 2000;
par.k  = 6;
par.mu = 0.10;      % change to 0.00 or 0.20 if needed
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

R = 1;
P = 0;
z_vals = [0 0.05 0.10 0.20 0.30 0.40];
S_vals = linspace(-1,1,41);
T_vals = linspace(0,3,41);

nS = length(S_vals);
nT = length(T_vals);
nZ = length(z_vals);

all_maps_pC = zeros(nS,nT,nZ);
all_maps_x  = zeros(nS,nT,nZ);

n_rhs   = 0;
n_tail  = 0;
n_bound = 0;
n_slow  = 0;

%% -------------------- main computation --------------------
for zi = 1:nZ
    z = z_vals(zi);
    fprintf('Running z = %.2f, beta = %.3f, mu = %.2f, N = %d\n', ...
        z,beta,par.mu,par.N);

    for si = 1:nS
        for ti = 1:nT
            S = S_vals(si);
            T = T_vals(ti);

            [pC_star,x_star,method] = steady_state_ode15s(beta,z,S,T,R,P,par);

            all_maps_pC(si,ti,zi) = pC_star;
            all_maps_x(si,ti,zi)  = x_star;

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
    end
end

%% -------------------- plot heat maps --------------------
fig1 = plot_heatmaps(all_maps_pC,T_vals,S_vals,z_vals,beta,par.mu,par.N,'p_C^*');

fig2 = plot_heatmaps(all_maps_x,T_vals,S_vals,z_vals,beta,par.mu,par.N,'x^*');

safe_save(fig1,'heat_map_total_pC_N2000_corrected.png');
safe_save(fig2,'heat_map_ordinary_x_N2000_corrected.png');

%% -------------------- output summary --------------------
fprintf('\nSteady-state acceptance summary\n');
fprintf('Raw RHS accepted points       = %d\n',n_rhs);
fprintf('Terminal-tail accepted points = %d\n',n_tail);
fprintf('Boundary accepted points      = %d\n',n_bound);
fprintf('Maximum-pass final points     = %d\n',n_slow);
if n_slow == 0
    fprintf('All heat-map points accepted.\n');
else
    fprintf('Some points are slow. Increase max_pass/pass_time if needed.\n');
end

save('heat_map_regularized_N2000_corrected_results.mat','all_maps_pC','all_maps_x', ...
    'S_vals','T_vals','z_vals','beta','par');

disp('Heat maps complete.');

end

%% ==========================================================
function [pC_star,x_star,method] = steady_state_ode15s(beta,z,S,T,R,P,par)
% method = 1: raw RHS norm small
% method = 2: terminal tail stable
% method = 3: absorbing boundary accepted
% method = 4: maximum pass reached

par = set_local_parameters(par,z,S,T,R,P);
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

x_star  = min(max(Y(1),0),1);
pC_star = par.z + par.u*x_star;
pC_star = min(max(pC_star,0),1);

end

%% ==========================================================
function par = set_local_parameters(par,z,S,T,R,P)

par.z   = z;
par.u   = 1 - z;
par.pZZ = z^2;
par.S   = S;
par.T   = T;
par.R   = R;
par.P   = P;

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
R = par.R;
S = par.S;
T = par.T;
P = par.P;

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
function fig = plot_heatmaps(M,T_vals,S_vals,z_vals,beta,mu,N,colorbar_label)

fig = figure('Color','w','Position',[40 50 1300 720]);

plot_w = 0.205;
plot_h = 0.315;
x_pos  = [0.070 0.335 0.600];
y_pos  = [0.560 0.125];
cb_pos = [0.845 0.125 0.018 0.750];

last_ax = [];

for zi = 1:length(z_vals)
    row = ceil(zi/3);
    col = zi - 3*(row-1);
    ax_pos = [x_pos(col) y_pos(row) plot_w plot_h];
    ax = axes('Position',ax_pos);
    last_ax = ax;

    imagesc(T_vals,S_vals,M(:,:,zi));
    set(ax,'YDir','normal','Layer','top');
    caxis(ax,[0 1]);
    hold(ax,'on');

    line(ax,[min(T_vals) max(T_vals)],[0 0],'Color','k','LineWidth',1.3);
    line(ax,[1 1],[min(S_vals) max(S_vals)],'Color','k','LineWidth',1.3);

    text(ax,0.28, 0.68,'TR','FontWeight','bold','FontSize',18,'Color','k');
    text(ax,1.55, 0.68,'CG','FontWeight','bold','FontSize',18,'Color','k');
    text(ax,0.28,-0.72,'SH','FontWeight','bold','FontSize',18,'Color','k');
    text(ax,1.55,-0.72,'PD','FontWeight','bold','FontSize',18,'Color','k');

    if row == 2
        xlabel(ax,'T','FontWeight','bold');
    else
        xlabel(ax,'');
    end

    if col == 1
        ylabel(ax,'S','FontWeight','bold');
    else
        ylabel(ax,'');
    end

    title(ax,sprintf('z = %.2f',z_vals(zi)), ...
        'FontWeight','bold','Interpreter','tex');

    xlim(ax,[min(T_vals) max(T_vals)]);
    ylim(ax,[min(S_vals) max(S_vals)]);
    set(ax,'FontSize',18);

    hold(ax,'off');
end

hcb = colorbar('peer',last_ax);
set(hcb,'Position',cb_pos);
ylabel(hcb,colorbar_label,'FontWeight','bold','Interpreter','tex');

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
function thesis_branch_bistability_merging_N2000_short()


clc; clear; close all;

%% -------------------- parameters --------------------
p.k  = 6;
p.mu = 0;
p.N  = 2000;

p.rhs_tol   = 1e-10;
p.tail_tol  = 1e-7;
p.bound_tol = 1e-8;
p.tiny      = 1e-14;

p.pass_time = 350*p.N;
p.max_step  = 5*p.N;
p.max_pass  = 1000;
p.min_pass  = 3;
p.coeff     = make_coeff(p.k);

betaList  = [0.01 0.05 0.1 0.5 1 2 3 5 7 10];
showBetas = [0.01 2 7];
zList     = 0:0.005:0.50;

gapTol      = 1e-3;
mergeTol    = 1e-3;
xTarget     = 1 - 1e-8;
minBiPoints = 2;

games(1) = struct('short','PD','R',1,'S',-0.5,'T',1.5,'P',0);
games(2) = struct('short','SH','R',1,'S',-0.5,'T',0.5,'P',0);
games(3) = struct('short','CG','R',1,'S',0.5,'T',1.5,'P',0);

nG = length(games);
nB = length(betaList);
nZ = length(zList);

xLowAll  = zeros(nG,nB,nZ);
xHighAll = zeros(nG,nB,nZ);
gapAll   = zeros(nG,nB,nZ);
zCrit    = NaN(nG,nB);
hasCME   = false(nG,nB);
hasFull  = false(nG,nB);
hasBi    = false(nG,nB);

n_rhs   = 0;
n_tail  = 0;
n_bound = 0;
n_slow  = 0;

fprintf('\nBranch-bistability test with N = %d\n',p.N);
fprintf('pass_time = %.0f, max_pass = %d\n',p.pass_time,p.max_pass);
fprintf('%6s %8s %10s %10s %12s\n','Game','beta','maxGap','zCrit','class');

%% -------------------- main computation --------------------
for g = 1:nG
    pg = p;
    pg.R = games(g).R;
    pg.S = games(g).S;
    pg.T = games(g).T;
    pg.P = games(g).P;

    for b = 1:nB
        beta = betaList(b);
        [xLow,xHigh,methods] = sweep_game(beta,pg,zList);

        n_rhs   = n_rhs   + sum(methods(:) == 1);
        n_tail  = n_tail  + sum(methods(:) == 2);
        n_bound = n_bound + sum(methods(:) == 3);
        n_slow  = n_slow  + sum(methods(:) == 4);

        gap  = abs(xHigh - xLow);
        xMin = min(xLow,xHigh);

        xLowAll(g,b,:)  = xLow;
        xHighAll(g,b,:) = xHigh;
        gapAll(g,b,:)   = gap;

        biIdx = find((gap > gapTol) & (zList > 0));
        hasBi(g,b) = length(biIdx) >= minBiPoints;
        hasFull(g,b) = any(xMin >= xTarget);

        if hasBi(g,b)
            startIdx = biIdx(end) + 1;
            if startIdx <= nZ
                idx = find((gap(startIdx:end) <= mergeTol) & ...
                           (xMin(startIdx:end) >= xTarget),1,'first');
                if ~isempty(idx)
                    zCrit(g,b) = zList(startIdx + idx - 1);
                    hasCME(g,b) = true;
                end
            end
        end

        if hasCME(g,b)
            cls = 'CME';
        elseif hasFull(g,b)
            cls = 'smoothFull';
        elseif hasBi(g,b)
            cls = 'bistableNoFull';
        else
            cls = 'noCME';
        end

        fprintf('%6s %8.3g %10.3e %10s %12s\n', ...
            games(g).short,beta,max(gap),fmt_z(zCrit(g,b)),cls);
    end
end

fprintf('\nSteady-state acceptance summary\n');
fprintf('Raw RHS accepted points       = %d\n',n_rhs);
fprintf('Terminal-tail accepted points = %d\n',n_tail);
fprintf('Boundary accepted points      = %d\n',n_bound);
fprintf('Maximum-pass final points     = %d\n',n_slow);
if n_slow > 0
    fprintf('Note: A few points still reached maximum passes. Check if they affect zCrit.\n');
else
    fprintf('All points accepted.\n');
end

%% -------------------- branch figures --------------------
for g = 1:nG
    plot_branch_figure(g,games,betaList,showBetas,zList, ...
        xLowAll,xHighAll,zCrit,hasCME,hasFull,hasBi,p.N);
end

%% -------------------- branch-gap heat map --------------------
fig = figure('Color','w','Position',[80 120 1450 430]);
yLog = log10(betaList);
yLabs = cell(size(betaList));
for i = 1:length(betaList)
    yLabs{i} = num2str(betaList(i));
end

last_ax = [];

for g = 1:nG
    ax = subplot(1,3,g);
    last_ax = ax;

    imagesc(zList,yLog,squeeze(gapAll(g,:,:)));
    set(gca,'YDir','normal','YTick',yLog,'YTickLabel',yLabs, ...
        'FontName','Times New Roman','FontSize',18,'LineWidth',1.0);
    colormap(gca,parula(256));
    caxis([0 max(gapAll(:))]);

    xlabel('Zealot fraction (z)','FontWeight','bold','FontSize',18);
    if g == 1
        ylabel('Selection intensity (\beta)','FontWeight','bold','Interpreter','tex','FontSize',18);
    else
        ylabel('');
    end
    title([games(g).short ': branch gap'], ...
        'FontWeight','bold','Interpreter','tex');
    xlim([min(zList) max(zList)]);
    ylim([min(yLog) max(yLog)]);
end

hcb = colorbar('peer',last_ax);
set(hcb,'Position',[0.930 0.160 0.015 0.700]);
ylabel(hcb,'Branch gap','FontWeight','bold','FontSize',16);

safe_save(fig,'branch_gap_N2000_short.png');

end

%% ==========================================================
function [xLow,xHigh,methods] = sweep_game(beta,p,zList)

nZ = length(zList);
xLow = zeros(1,nZ);
xHigh = zeros(1,nZ);
methods = zeros(2,nZ);

Ylow = [];
Yhigh = [];

for i = 1:nZ
    p.z = zList(i);
    p.u = 1 - p.z;
    p.pZZ = p.z^2;

    if isempty(Ylow)
        Ylow = init_state(1e-4,p);
    else
        Ylow = sanitize(Ylow,p);
    end
    [Ylow,methods(1,i)] = solve_ss(beta,Ylow,p);
    xLow(i) = Ylow(1);

    if isempty(Yhigh)
        Yhigh = init_state(1-1e-4,p);
    else
        Yhigh = sanitize(Yhigh,p);
    end
    [Yhigh,methods(2,i)] = solve_ss(beta,Yhigh,p);
    xHigh(i) = Yhigh(1);
end

end

%% ==========================================================
function [Y,method] = solve_ss(beta,Y0,p)
% method = 1: raw RHS small
% method = 2: terminal tail stable
% method = 3: absorbing boundary accepted
% method = 4: maximum pass reached

opts = odeset('RelTol',1e-9,'AbsTol',1e-11,'MaxStep',p.max_step);
Y = sanitize(Y0,p);
method = 4;

for pass = 1:p.max_pass
    Yold = Y;
    [~,Ysol] = ode15s(@(t,y) rhs(y,beta,p),[0 p.pass_time],Y,opts);
    Y = sanitize(Ysol(end,:).',p);

    Fraw = rhs_raw(Y,beta,p);
    raw_norm = norm(Fraw,inf);
    tail_change = max(abs(Y-Yold));

    if raw_norm < p.rhs_tol
        method = 1;
        return;
    end

    if pass >= p.min_pass && tail_change < p.tail_tol
        method = 2;
        return;
    end

    if boundary_accepted(Y,Fraw,p)
        method = 3;
        return;
    end
end

end

%% ==========================================================
function ok = boundary_accepted(Y,Fraw,p)

x = Y(1);
xdot_raw = Fraw(1);

near_zero = (x < p.bound_tol) && (xdot_raw <= p.rhs_tol);
near_one  = (x > 1-p.bound_tol) && (xdot_raw >= -p.rhs_tol);

ok = (p.mu == 0) && (near_zero || near_one);

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

pC = p.u*x;
pD = p.u*(1-x);
pCD = max(pC-pCC-pZC,0);
pZD = max(p.z-p.pZZ-pZC,0);

[qC_C,qZ_C,qD_C] = cond_probs(pCC,pZC,pC,p.tiny);
[qC_D,qZ_D,qD_D] = cond_probs(pCD,pZD,pD,p.tiny);

Tplus = 0;
Tminus = 0;
GCCp = 0;
GCCm = 0;
GZCp = 0;
GZCm = 0;

for kZ = 0:p.k
    for kC = 0:(p.k-kZ)
        kD = p.k-kZ-kC;
        c = p.coeff(kZ+1,kC+1);

        A = (kZ+kC)/p.k;
        B = kD/p.k;
        PiC = A*p.R + B*p.S;
        PiD = A*p.T + B*p.P;
        Delta = PiC - PiD;

        fplus = fermi(beta*Delta);
        fminus = fermi(-beta*Delta);

        phip = p.mu + (1-p.mu)*A*fplus;
        phim = p.mu + (1-p.mu)*B*fminus;

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
     2*(pD*GCCp - pC*GCCm)/p.k; ...
     2*(pD*GZCp - pC*GZCm)/p.k];

end

%% ==========================================================
function plot_branch_figure(g,games,betaList,showBetas,zList, ...
    xLowAll,xHighAll,zCrit,hasCME,hasFull,hasBi,N)

colLow = [0.000 0.250 0.650];
colHigh = [0.850 0.180 0.050];
colCrit = [0.000 0.520 0.250];
dzShow = 0.0015;

fig = figure('Color','w','Position',[80 100 1650 500]);

ax_pos = [0.060 0.150 0.235 0.720; ...
          0.335 0.150 0.235 0.720; ...
          0.610 0.150 0.235 0.720];

branch_ylim = [-0.05 1.02];

legend_hHigh = [];
legend_hLow  = [];
legend_hCrit = [];

for j = 1:length(showBetas)
    [~,b] = min(abs(betaList-showBetas(j)));
    xLow = squeeze(xLowAll(g,b,:)).';
    xHigh = squeeze(xHighAll(g,b,:)).';

    axes('Position',ax_pos(j,:)); hold on; box on; grid on;
    hLow = plot(zList,xLow,'-o','Color',colLow,'LineWidth',2.1, ...
        'MarkerSize',4.5,'MarkerFaceColor',colLow);
    hHigh = plot(zList+dzShow,xHigh,'--s','Color',colHigh,'LineWidth',2.0, ...
        'MarkerSize',5.5,'MarkerFaceColor','w');

    hCrit = [];
    if ~isnan(zCrit(g,b))
        hCrit = plot([zCrit(g,b) zCrit(g,b)],branch_ylim,'-.' ,'Color',colCrit,'LineWidth',1.8);
    end

    if isempty(legend_hHigh)
        legend_hHigh = hHigh;
        legend_hLow  = hLow;
    end
    if isempty(legend_hCrit) && ~isempty(hCrit)
        legend_hCrit = hCrit;
    end

    if hasCME(g,b)
        label = ['CME, z_c = ' num2str(zCrit(g,b),'%.3f')];
    elseif hasFull(g,b)
        label = 'smooth full';
    elseif hasBi(g,b)
        label = 'bistable, no merge';
    else
        label = 'no CME';
    end

    title([games(g).short '(\beta = ' num2str(betaList(b)) ')'], ...
        'FontWeight','bold','Interpreter','tex','FontSize',16);
    xlabel('Zealot fraction (z)','FontWeight','bold','FontSize',20);
    if j == 1
        ylabel('Ordinary cooperation (x^*)','FontWeight','bold','FontSize',20);
    else
        ylabel('');
    end
    xlim([min(zList) max(zList)+dzShow]);
    ylim(branch_ylim);
    set(gca,'FontName','Times New Roman','FontSize',18);
end

if isempty(legend_hCrit)
    lgd = legend([legend_hHigh legend_hLow], ...
        {'High initial','Low initial'}, ...
        'FontSize',14, ...
        'Interpreter','tex');
else
    lgd = legend([legend_hHigh legend_hLow legend_hCrit], ...
        {'High initial','Low initial','Transition line'}, ...
        'FontSize',14, ...
        'Interpreter','tex');
end
set(lgd,'Position',[0.865 0.385 0.115 0.230]);

safe_save(fig,[games(g).short '_CME_N2000_short.png']);

end

%% ==========================================================
function safe_save(fig,filename)
% Safe save for MATLAB versions that sometimes throw invalid figure warnings.

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

%% ==========================================================
function Y0 = init_state(x0,p)

pC = p.u*x0;
Y0 = sanitize([x0; pC^2; p.z*pC],p);

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

x = min(max(Y(1),1e-12),1-1e-12);
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
function s = fmt_z(z)

if isnan(z)
    s = '---';
else
    s = sprintf('%.3f',z);
end

end
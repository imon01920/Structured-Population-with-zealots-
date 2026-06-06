function MC_weak_selection_PRE_exact_AB_2016a()


% This code does TWO linked tasks:
%
% PART I: Exact analytical coefficient calculation
% ------------------------------------------------------------
%   1. Solve the neutral regularized PA steady state Y0_mu
%   2. Compute the neutral Jacobian J_mu
%   3. Compute structural vectors U_mu and V_mu
%   4. Compute
%
%        A_mu = -[1 0 0]*(J_mu\U_mu)
%        B_mu = -[1 0 0]*(J_mu\V_mu)
%
%   5. Construct the analytical weak-selection predictor
%
%        x1_mu = A_mu(R-T) + B_mu(S-P)
%
% PART II: Explicit-network Monte Carlo validation
% ------------------------------------------------------------
%   1. Build random k-regular graphs
%   2. Simulate stochastic updating with zealots
%   3. Compare beta>0 and beta=0 paired trajectories
%   4. Estimate
%
%        [x_mu^*(beta) - x_{0,mu}^*]/beta
%
%   5. Fit the weak-selection MC slope across several small beta
%   6. Compare the MC zero crossing with the analytical crossing
clc;
clear;
close all;
rng(1);

%% ============================================================
% PART I. PARAMETERS FOR EXACT PA ANALYTICAL CALCULATION
%% ============================================================

par.k  = 6;
par.z  = 0.10;
par.u  = 1 - par.z;
par.mu = 1e-4;
par.N  = 2000;

par.pZZ = par.z^2;

par.tol_raw  = 1e-10;
par.fdY      = 1e-6;
par.tiny     = 1e-14;
par.max_pass = 50;
par.pass_time = 50 * par.N;
par.max_step  = 5 * par.N;

% Safety tolerance for accepting the neutral steady state.
% This is deliberately weaker than par.tol_raw, because par.tol_raw is
% very strict. If the residual is above this level, A_mu and B_mu may be
% computed at a non-steady state and the validation may become unreliable.
par.neutral_residual_accept = 1e-8;

par.coeff = make_coeff(par.k);

% Base payoff parameters
base.R = 1.20;
base.S = -0.40;
base.T = 1.60;
base.P = 0.00;

% Small beta values used for MC weak-selection validation
beta_small = [0.0025 0.005 0.010];

% Sweep design
nSweep = 41;
sweepSpan = 1.0;

%% ============================================================
% PART I-A. NEUTRAL REGULARIZED STEADY STATE
%% ============================================================

x_init = 0.50;
pC_init = par.u * x_init;

Yinit = sanitize([x_init; pC_init^2; par.z*pC_init],par);

par0 = set_payoffs(par,0,0,0,0);

[Y0,ok0,F0] = solve_ss(0,Yinit,par0);

% ------------------------------------------------------------
% CRITICAL SAFETY CHECK
% ------------------------------------------------------------
% Do not silently continue if the neutral PA steady state is not accurate.
% The weak-selection coefficients A_mu and B_mu depend on Y0, J_mu, U_mu,
% and V_mu. If Y0 is not close to a true neutral steady state, the analytical
% crossing can be wrong and the Monte Carlo validation becomes misleading.
% ------------------------------------------------------------
if F0 > par.neutral_residual_accept
    error(['Neutral PA steady state is not accurate enough. ', ...
           'Residual F0 = %.4e exceeds acceptable tolerance %.4e. ', ...
           'Increase par.max_pass, par.pass_time, or relax solver settings.'], ...
           F0,par.neutral_residual_accept);
elseif ~ok0
    warning(['Neutral solver did not meet the strict tolerance %.4e, ', ...
             'but residual F0 = %.4e is below acceptable tolerance %.4e. ', ...
             'Proceeding with caution.'], ...
             par.tol_raw,F0,par.neutral_residual_accept);
end

x0_mu = Y0(1);

%% ============================================================
% PART I-B. JACOBIAN, STRUCTURAL VECTORS, A_mu, B_mu
%% ============================================================

J = num_jac(Y0,par0);

% Additional diagnostic: A_mu and B_mu require solving linear systems with J.
% A very large condition number means the coefficients may be numerically
% unstable even if the neutral residual is small.
condJ = cond(J);

[U,V] = structural_vectors(Y0,par);

Amu = -[1 0 0]*(J\U);
Bmu = -[1 0 0]*(J\V);

if condJ > 1e10
    warning(['Neutral Jacobian is ill-conditioned: cond(J) = %.4e. ', ...
             'A_mu and B_mu may be numerically unstable.'],condJ);
end

%% ============================================================
% PART I-C. CHOOSE PAYOFF SWEEP AND ANALYTICAL CROSSING
%% ============================================================

if abs(Bmu) >= abs(Amu)

    if abs(Bmu) < 1e-14
        error('B_mu is too close to zero for a stable S-crossing calculation.');
    end

    sweepName = 'S';

    cross_pred = base.P - (Amu/Bmu)*(base.R-base.T);

    sweepVals = linspace( ...
        cross_pred-sweepSpan, ...
        cross_pred+sweepSpan, ...
        nSweep);

    theoryCurve = Amu*(base.R-base.T) + ...
                  Bmu*(sweepVals-base.P);

    xlab = 'Swept payoff entry S';

else

    if abs(Amu) < 1e-14
        error('A_mu is too close to zero for a stable T-crossing calculation.');
    end

    sweepName = 'T';

    cross_pred = base.R + (Bmu/Amu)*(base.S-base.P);

    sweepVals = linspace( ...
        cross_pred-sweepSpan, ...
        cross_pred+sweepSpan, ...
        nSweep);

    theoryCurve = Amu*(base.R-sweepVals) + ...
                  Bmu*(base.S-base.P);

    xlab = 'Swept payoff entry T';
end

theory_cross = zero_crossing_2016a(sweepVals,theoryCurve);

%% ============================================================
% PRINT EXACT ANALYTICAL SUMMARY
%% ============================================================

fprintf('\n============================================================\n');
fprintf('EXACT ANALYTICAL STRUCTURAL COEFFICIENT CALCULATION\n');
fprintf('============================================================\n');
fprintf('Neutral convergence flag         = %d\n',ok0);
fprintf('Neutral raw residual             = %.4e\n',F0);
fprintf('Neutral residual accept tol      = %.4e\n',par.neutral_residual_accept);
fprintf('cond(J_mu)                       = %.4e\n',condJ);
fprintf('x0_mu                            = %.10f\n',x0_mu);
fprintf('A_mu                             = %.12e\n',Amu);
fprintf('B_mu                             = %.12e\n',Bmu);
fprintf('Sweep entry                      = %s\n',sweepName);
fprintf('Predicted analytical crossing    = %.10f\n',cross_pred);
fprintf('Theory curve zero crossing       = %.10f\n',theory_cross);
fprintf('============================================================\n\n');

%% ============================================================
% PART II. MONTE CARLO SETTINGS
%% ============================================================

mc.N  = par.N;
mc.k  = par.k;
mc.z  = par.z;
mc.mu = par.mu;

mc.R = base.R;
mc.S = base.S;
mc.T = base.T;
mc.P = base.P;

% PRE-level Monte Carlo settings
nRep = 30;

burnSweeps    = 2500;
measureSweeps = 2500;
maxSweeps     = burnSweeps + measureSweeps;

sampleEvery = 10;
swapFactor  = 15;

% Use analytical neutral ordinary cooperation as initial MC condition
mc.xInit = x0_mu;

% Tail drift diagnostic
tailDriftWarn = 0.003;

nB = length(beta_small);
nT = length(sweepVals);

%% ============================================================
% PART II-A. STORAGE
%% ============================================================

xBetaMeanAll = NaN(nRep,nB,nT);
xZeroMeanAll = NaN(nRep,nB,nT);

deltaXAll = NaN(nRep,nB,nT);
slopeAll  = NaN(nRep,nB,nT);

fitSlopeRep = NaN(nRep,nT);

tailDriftBetaAll = NaN(nRep,nB,nT);
tailDriftZeroAll = NaN(nRep,nB,nT);

crossRep = NaN(nRep,1);
elapsedByRep = NaN(nRep,1);

checkpointFile = 'MC_weak_PRE_exactAB_checkpoint_2016a.mat';

%% ============================================================
% MONTE CARLO HEADER
%% ============================================================

fprintf('\n============================================================\n');
fprintf('PRE-LEVEL MONTE CARLO WEAK-SELECTION VALIDATION\n');
fprintf('============================================================\n');
fprintf('N                         = %d\n',mc.N);
fprintf('k                         = %d\n',mc.k);
fprintf('z                         = %.3f\n',mc.z);
fprintf('mu                        = %.4g\n',mc.mu);
fprintf('Initial MC x              = %.10f\n',mc.xInit);
fprintf('beta values               = ');
fprintf('%.4g ',beta_small);
fprintf('\n');
fprintf('Number of sweep points    = %d\n',nT);
fprintf('Sweep variable            = %s\n',sweepName);
fprintf('Sweep range               = %.5f to %.5f\n', ...
    min(sweepVals),max(sweepVals));
fprintf('Repetitions               = %d\n',nRep);
fprintf('Burn-in sweeps            = %d\n',burnSweeps);
fprintf('Measurement sweeps        = %d\n',measureSweeps);
fprintf('Sampling every            = %d sweeps\n',sampleEvery);
fprintf('Edge-swap factor          = %d\n',swapFactor);
fprintf('============================================================\n\n');

globalTic = tic;

%% ============================================================
% PART II-B. MONTE CARLO MAIN LOOP
%% ============================================================

for rep = 1:nRep

    repTic = tic;

    fprintf('\n------------------------------------------------------------\n');
    fprintf('Repetition %d / %d\n',rep,nRep);
    fprintf('------------------------------------------------------------\n');

    % --------------------------------------------------------
    % Random k-regular graph
    % --------------------------------------------------------
    neighbors = random_regular_neighbors_switching_2016a( ...
        mc.N,mc.k,swapFactor);

    % --------------------------------------------------------
    % Fixed zealot placement for this repetition
    % --------------------------------------------------------
    zealotOrder = randperm(mc.N);

    stateInitial = initialize_state_zero_crossing_2016a( ...
        mc.N,mc.z,mc.xInit,zealotOrder);

    for b = 1:nB

        beta = beta_small(b);

        fprintf('   beta = %.4g\n',beta);

        for ti = 1:nT

            sweepNow = sweepVals(ti);

            if mod(ti,5) == 1 || ti == nT
                fprintf('      %s = %.5f   (%d/%d)\n', ...
                    sweepName,sweepNow,ti,nT);
            end

            % ----------------------------------------------------
            % Current payoff values
            % ----------------------------------------------------
            if strcmp(sweepName,'S')
                Rnow = base.R;
                Snow = sweepNow;
                Tnow = base.T;
                Pnow = base.P;
            else
                Rnow = base.R;
                Snow = base.S;
                Tnow = sweepNow;
                Pnow = base.P;
            end

            % ----------------------------------------------------
            % Transition probabilities:
            % beta > 0 chain
            % ----------------------------------------------------
            [probBetaD2C,probBetaC2D] = ...
                precompute_switch_probs_2016a( ...
                beta,mc,Rnow,Snow,Tnow,Pnow);

            % ----------------------------------------------------
            % beta = 0 neutral chain
            % Payoffs do not matter at beta = 0, but we pass
            % the same values for clarity.
            % ----------------------------------------------------
            [probZeroD2C,probZeroC2D] = ...
                precompute_switch_probs_2016a( ...
                0,mc,Rnow,Snow,Tnow,Pnow);

            % ----------------------------------------------------
            % Common random numbers.
            % Seed independent of sweep point to reduce noise
            % across the payoff sweep.
            % ----------------------------------------------------
            rngSeed = 1000000 + 10000*rep + 100*b;
            rng(rngSeed);

            [xBetaMean,xZeroMean,tailBeta,tailZero] = ...
                run_paired_MC_weak_slope_2016a( ...
                stateInitial,neighbors, ...
                probBetaD2C,probBetaC2D, ...
                probZeroD2C,probZeroC2D, ...
                mc,maxSweeps,burnSweeps,sampleEvery);

            xBetaMeanAll(rep,b,ti) = xBetaMean;
            xZeroMeanAll(rep,b,ti) = xZeroMean;

            deltaXAll(rep,b,ti) = xBetaMean - xZeroMean;
            slopeAll(rep,b,ti) = ...
                (xBetaMean - xZeroMean)/beta;

            tailDriftBetaAll(rep,b,ti) = tailBeta;
            tailDriftZeroAll(rep,b,ti) = tailZero;

        end
    end

    % --------------------------------------------------------
    % Fit MC weak-selection slope across beta:
    %
    % deltaX(beta) approx beta * slope_fit
    %
    % slope_fit = sum(beta * deltaX)/sum(beta^2)
    % --------------------------------------------------------
    for ti = 1:nT

        dvec = zeros(nB,1);

        for b = 1:nB
            dvec(b) = deltaXAll(rep,b,ti);
        end

        fitSlopeRep(rep,ti) = ...
            sum(beta_small(:).*dvec(:)) / ...
            sum(beta_small(:).^2);
    end

    crossRep(rep) = zero_crossing_2016a( ...
        sweepVals,fitSlopeRep(rep,:));

    elapsedByRep(rep) = toc(repTic);

    fprintf('Repetition %d completed in %.2f minutes.\n', ...
        rep,elapsedByRep(rep)/60);

    fprintf('Replicate MC zero crossing = %.10f\n',crossRep(rep));

    totalElapsed = toc(globalTic);
    fprintf('Total elapsed time so far = %.2f hours.\n', ...
        totalElapsed/3600);

    save(checkpointFile, ...
        'par', ...
        'base', ...
        'Y0', ...
        'x0_mu', ...
        'J', ...
        'condJ', ...
        'U', ...
        'V', ...
        'Amu', ...
        'Bmu', ...
        'sweepName', ...
        'cross_pred', ...
        'theory_cross', ...
        'sweepVals', ...
        'theoryCurve', ...
        'mc', ...
        'beta_small', ...
        'nRep', ...
        'burnSweeps', ...
        'measureSweeps', ...
        'maxSweeps', ...
        'sampleEvery', ...
        'swapFactor', ...
        'xBetaMeanAll', ...
        'xZeroMeanAll', ...
        'deltaXAll', ...
        'slopeAll', ...
        'fitSlopeRep', ...
        'tailDriftBetaAll', ...
        'tailDriftZeroAll', ...
        'crossRep', ...
        'elapsedByRep');

end

totalRuntime = toc(globalTic);

%% ============================================================
% PART II-C. FINAL MONTE CARLO STATISTICS
%% ============================================================

slopeMean = squeeze(mean(slopeAll,1));
slopeSE   = squeeze(std(slopeAll,0,1) ./ sqrt(nRep));
slopeCI95 = 1.96 .* slopeSE;

fitSlopeMean = mean(fitSlopeRep,1);
fitSlopeSE   = std(fitSlopeRep,0,1) ./ sqrt(nRep);
fitSlopeCI95 = 1.96 .* fitSlopeSE;

mcCrossFromMeanCurve = zero_crossing_2016a( ...
    sweepVals,fitSlopeMean);

validCross = crossRep(~isnan(crossRep));

if ~isempty(validCross)
    mcCrossMean = mean(validCross);
    mcCrossSE = std(validCross) / sqrt(length(validCross));
    mcCrossCI95 = 1.96 * mcCrossSE;
else
    mcCrossMean = NaN;
    mcCrossSE = NaN;
    mcCrossCI95 = NaN;
end

maxTailDriftBeta = max(tailDriftBetaAll(:));
maxTailDriftZero = max(tailDriftZeroAll(:));

meanTailDriftBeta = mean(tailDriftBetaAll(:));
meanTailDriftZero = mean(tailDriftZeroAll(:));

%% ============================================================
% FINAL SUMMARY
%% ============================================================

fprintf('\n============================================================\n');
fprintf('FINAL PRE-LEVEL WEAK-SELECTION MC VALIDATION SUMMARY\n');
fprintf('============================================================\n');
fprintf('x0_mu                                   = %.10f\n',x0_mu);
fprintf('A_mu                                    = %.12e\n',Amu);
fprintf('B_mu                                    = %.12e\n',Bmu);
fprintf('cond(J_mu)                              = %.4e\n',condJ);
fprintf('Neutral raw residual                    = %.4e\n',F0);
fprintf('Analytical predicted crossing          = %.10f\n',cross_pred);
fprintf('Theory curve zero crossing             = %.10f\n',theory_cross);
fprintf('MC crossing from mean fitted curve     = %.10f\n', ...
    mcCrossFromMeanCurve);
fprintf('Mean replicate MC crossing             = %.10f\n', ...
    mcCrossMean);
fprintf('95%% CI half-width for MC crossing      = %.10f\n', ...
    mcCrossCI95);
fprintf('Valid replicate crossings              = %d / %d\n', ...
    length(validCross),nRep);
fprintf('------------------------------------------------------------\n');
fprintf('Mean tail drift, beta chains           = %.4e\n', ...
    meanTailDriftBeta);
fprintf('Mean tail drift, neutral chains        = %.4e\n', ...
    meanTailDriftZero);
fprintf('Max tail drift, beta chains            = %.4e\n', ...
    maxTailDriftBeta);
fprintf('Max tail drift, neutral chains         = %.4e\n', ...
    maxTailDriftZero);

if maxTailDriftBeta > tailDriftWarn || ...
   maxTailDriftZero > tailDriftWarn

    fprintf(['WARNING: Some trajectories show visible tail drift. ', ...
             'Consider longer burn-in or measurement windows.\n']);
else
    fprintf('Tail-drift diagnostic remains within preset tolerance.\n');
end

fprintf('------------------------------------------------------------\n');
fprintf('Total runtime                           = %.3f hours\n', ...
    totalRuntime/3600);
fprintf('============================================================\n\n');

%% ============================================================
% FIGURE 1. MAIN PUBLICATION-QUALITY FITTED MC SLOPE FIGURE
%% ============================================================

fig1 = figure('Color','w','Position',[90 90 1080 640]);
hold on;
box on;
grid on;

xMinPlot = min(sweepVals);
xMaxPlot = max(sweepVals);

plot_CI_band_2016a( ...
    sweepVals,fitSlopeMean,fitSlopeCI95,[0.90 0.80 0.92]);

hTheory = plot( ...
    sweepVals,theoryCurve,'k-', ...
    'LineWidth',2.6);

hMC = plot( ...
    sweepVals,fitSlopeMean,'mo-', ...
    'LineWidth',1.8, ...
    'MarkerSize',5, ...
    'MarkerFaceColor','m');

% Force the x-axis limits before drawing horizontal reference lines.
% This guarantees that the zero line spans the entire plotted panel.
xlim([xMinPlot xMaxPlot]);

hZero = plot( ...
    [xMinPlot xMaxPlot], ...
    [0 0], ...
    'k:', ...
    'LineWidth',1.4);

yl = ylim;

hTheoryCross = plot( ...
    [cross_pred cross_pred], ...
    yl, ...
    '--', ...
    'Color',[0.35 0.35 0.35], ...
    'LineWidth',1.6);

if ~isnan(mcCrossFromMeanCurve)

    hMCCross = plot( ...
        [mcCrossFromMeanCurve mcCrossFromMeanCurve], ...
        yl, ...
        '-.', ...
        'Color',[0.00 0.50 0.20], ...
        'LineWidth',1.8);
else
    hMCCross = [];
end

xlim([xMinPlot xMaxPlot]);
ylim(yl);

xlabel(xlab,'FontWeight','bold','FontSize',18);
ylabel('Weak selection slope','FontWeight','bold','FontSize',18);

if isempty(hMCCross)

    hLegend1 = legend([hTheory hMC hZero hTheoryCross], ...
        {'Theoretical graph', ...
         'Monte Carlo graph', ...
         'Zero line in x-axis', ...
         'Theoretical zero crossing line'}, ...
        'Location','eastoutside');

else

    hLegend1 = legend([hTheory hMC hZero hTheoryCross hMCCross], ...
        {'Theoretical graph', ...
         'Monte Carlo graph', ...
         'Zero line in x-axis', ...
         'Theoretical zero crossing line', ...
         'Monte Carlo zero crossing line'}, ...
        'Location','eastoutside');
end

set(hLegend1,'FontSize',16,'Box','on');

set(gca, ...
    'FontName','Times New Roman', ...
    'FontSize',12, ...
    'LineWidth',1.0);

safe_save_png_2016a( ...
    fig1,'MC_weak_PRE_exactAB_fitted_2016a.png');

%% ============================================================
% FIGURE 2. BETA-SPECIFIC MC SLOPE CURVES
%% ============================================================

fig2 = figure('Color','w','Position',[100 100 1080 640]);
hold on;
box on;
grid on;

xMinPlot = min(sweepVals);
xMaxPlot = max(sweepVals);

hTheory2 = plot( ...
    sweepVals,theoryCurve,'k-', ...
    'LineWidth',2.6);

styles = {'b--o','r-.s','g:^'};
hBeta = zeros(1,nB);

for b = 1:nB

    hBeta(b) = plot( ...
        sweepVals,slopeMean(b,:),styles{b}, ...
        'LineWidth',1.6, ...
        'MarkerSize',4.5);
end

% Force the x-axis limits before drawing horizontal reference lines.
% This guarantees that the zero line spans the entire plotted panel.
xlim([xMinPlot xMaxPlot]);

hZero2 = plot( ...
    [xMinPlot xMaxPlot], ...
    [0 0], ...
    'k:', ...
    'LineWidth',1.4);

yl = ylim;

hTheoryCross2 = plot( ...
    [cross_pred cross_pred], ...
    yl, ...
    '--', ...
    'Color',[0.35 0.35 0.35], ...
    'LineWidth',1.6);

xlim([xMinPlot xMaxPlot]);
ylim(yl);

xlabel(xlab,'FontWeight','bold','FontSize',18);
ylabel('Weak selection slope', ...
    'FontWeight','bold', ...
    'FontSize',18);

hLegend2 = legend([hTheory2 hBeta(1) hBeta(2) hBeta(3) hZero2 hTheoryCross2], ...
    {'Theoretical graph', ...
     ['Monte Carlo graph, \beta = ' num2str(beta_small(1))], ...
     ['Monte Carlo graph, \beta = ' num2str(beta_small(2))], ...
     ['Monte Carlo graph, \beta = ' num2str(beta_small(3))], ...
     'Zero line in x-axis', ...
     'Theoretical zero crossing line'}, ...
    'Location','eastoutside');

set(hLegend2,'FontSize',16,'Box','on');

set(gca, ...
    'FontName','Times New Roman', ...
    'FontSize',12, ...
    'LineWidth',1.0);

safe_save_png_2016a( ...
    fig2,'MC_weak_PRE_exactAB_beta_slopes_2016a.png');

%% ============================================================
% CSV EXPORT
%% ============================================================

csvFile = 'MC_weak_PRE_exactAB_summary_2016a.csv';
fid = fopen(csvFile,'w');

fprintf(fid,['sweepValue,theorySlope,fitSlopeMean,fitSlopeCI95,', ...
             'slopeBeta1Mean,slopeBeta1CI95,', ...
             'slopeBeta2Mean,slopeBeta2CI95,', ...
             'slopeBeta3Mean,slopeBeta3CI95\n']);

for ti = 1:nT

    fprintf(fid, ...
        '%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g\n', ...
        sweepVals(ti), ...
        theoryCurve(ti), ...
        fitSlopeMean(ti), ...
        fitSlopeCI95(ti), ...
        slopeMean(1,ti), ...
        slopeCI95(1,ti), ...
        slopeMean(2,ti), ...
        slopeCI95(2,ti), ...
        slopeMean(3,ti), ...
        slopeCI95(3,ti));
end

fclose(fid);

%% ============================================================
% FINAL MAT FILE
%% ============================================================

save('MC_weak_PRE_exactAB_results_2016a.mat', ...
    'par', ...
    'base', ...
    'Y0', ...
    'x0_mu', ...
    'J', ...
    'condJ', ...
    'U', ...
    'V', ...
    'Amu', ...
    'Bmu', ...
    'sweepName', ...
    'cross_pred', ...
    'theory_cross', ...
    'sweepVals', ...
    'theoryCurve', ...
    'mc', ...
    'beta_small', ...
    'nRep', ...
    'burnSweeps', ...
    'measureSweeps', ...
    'maxSweeps', ...
    'sampleEvery', ...
    'swapFactor', ...
    'xBetaMeanAll', ...
    'xZeroMeanAll', ...
    'deltaXAll', ...
    'slopeAll', ...
    'slopeMean', ...
    'slopeSE', ...
    'slopeCI95', ...
    'fitSlopeRep', ...
    'fitSlopeMean', ...
    'fitSlopeSE', ...
    'fitSlopeCI95', ...
    'tailDriftBetaAll', ...
    'tailDriftZeroAll', ...
    'crossRep', ...
    'mcCrossFromMeanCurve', ...
    'mcCrossMean', ...
    'mcCrossSE', ...
    'mcCrossCI95', ...
    'elapsedByRep', ...
    'totalRuntime');

fprintf('Saved files:\n');
fprintf('  MC_weak_PRE_exactAB_fitted_2016a.png\n');
fprintf('  MC_weak_PRE_exactAB_beta_slopes_2016a.png\n');
fprintf('  MC_weak_PRE_exactAB_results_2016a.mat\n');
fprintf('  MC_weak_PRE_exactAB_summary_2016a.csv\n');
fprintf('  MC_weak_PRE_exactAB_checkpoint_2016a.mat\n');
fprintf('\nFinished.\n');

end


%% ============================================================
% SOLVE NEUTRAL OR SELECTED PA STEADY STATE
%% ============================================================

function [Y,ok,res] = solve_ss(beta,Y0,p)

opts = odeset( ...
    'RelTol',1e-9, ...
    'AbsTol',1e-11, ...
    'MaxStep',p.max_step);

Y = sanitize(Y0,p);

ok = false;
res = Inf;

for pass = 1:p.max_pass

    [~,Ysol] = ode15s( ...
        @(t,y) rhs(y,beta,p), ...
        [0 p.pass_time], ...
        Y, ...
        opts);

    Y = sanitize(Ysol(end,:).',p);

    res = norm(rhs_raw(Y,beta,p),inf);

    if res < p.tol_raw
        ok = true;
        return;
    end
end

end


%% ============================================================
% PA RHS WITH EXPLICIT 1/N TIME SCALING
%% ============================================================

function F = rhs(Y,beta,p)

F0 = rhs_raw(Y,beta,p);

F = [ ...
    F0(1)/p.N; ...
    F0(2)/p.N; ...
    F0(3)/p.N];

end


%% ============================================================
% RAW PA RHS USED FOR RESIDUALS
%% ============================================================

function F = rhs_raw(Y,beta,p)

Y = sanitize(Y,p);

x   = Y(1);
pCC = Y(2);
pZC = Y(3);

pC = p.u*x;
pD = p.u*(1-x);

pCD = max(pC-pCC-pZC,0);
pZD = max(p.z-p.pZZ-pZC,0);

[qC_C,qZ_C,qD_C] = cond_probs(pCC,pZC,pC,p.tiny);
[qC_D,qZ_D,qD_D] = cond_probs(pCD,pZD,pD,p.tiny);

Tplus  = 0;
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

        fplus  = fermi(beta*Delta);
        fminus = fermi(-beta*Delta);

        phip = p.mu + (1-p.mu)*A*fplus;
        phim = p.mu + (1-p.mu)*B*fminus;

        probD = c * ...
            (qZ_D^kZ) * ...
            (qC_D^kC) * ...
            (qD_D^kD);

        Tplus = Tplus + probD*phip;
        GCCp  = GCCp  + probD*kC*phip;
        GZCp  = GZCp  + probD*kZ*phip;

        probC = c * ...
            (qZ_C^kZ) * ...
            (qC_C^kC) * ...
            (qD_C^kD);

        Tminus = Tminus + probC*phim;
        GCCm   = GCCm   + probC*kC*phim;
        GZCm   = GZCm   + probC*kZ*phim;
    end
end

xdot_raw   = ((1-x)*Tplus - x*Tminus);
pCCdot_raw = 2*(pD*GCCp - pC*GCCm)/p.k;
pZCdot_raw = 2*(pD*GZCp - pC*GZCm)/p.k;

F = [xdot_raw; pCCdot_raw; pZCdot_raw];

end


%% ============================================================
% EXACT STRUCTURAL VECTORS U_mu AND V_mu FROM THE THESIS
%% ============================================================

function [U,V] = structural_vectors(Y,p)

Y = sanitize(Y,p);

x   = Y(1);
pCC = Y(2);
pZC = Y(3);

k  = p.k;
mu = p.mu;
N  = p.N;

pC = p.u*x;
pD = p.u*(1-x);

pCD = max(pC-pCC-pZC,0);
pZD = max(p.z-p.pZZ-pZC,0);

[cC,zC,r0] = cond_probs(pCC,pZC,pC,p.tiny);
[cD,zD,q0] = cond_probs(pCD,pZD,pD,p.tiny);

ED_A2 = (1-q0)^2 + q0*(1-q0)/k;
ED_AB = q0*(1-q0)*(1 - 1/k);

EC_AB = r0*(1-r0)*(1 - 1/k);
EC_B2 = r0^2 + r0*(1-r0)/k;

FD = k - ((k-1)*(2*k-1)/k)*q0 + ...
     ((k-1)*(k-2)/k)*q0^2;

HD = ((k-1)*q0/k)*((k-1) - (k-2)*q0);

FC = ((k-1)*r0/k)*((k-1) - (k-2)*r0);

HC = ((k-1)*r0/k)*(1 + (k-2)*r0);

U = [ ...
    (1-mu)*((1-x)*ED_A2 + x*EC_AB)/(4*N); ...
    (1-mu)*(pD*cD*FD + pC*cC*FC)/(2*k*N); ...
    (1-mu)*(pD*zD*FD + pC*zC*FC)/(2*k*N)];

V = [ ...
    (1-mu)*((1-x)*ED_AB + x*EC_B2)/(4*N); ...
    (1-mu)*(pD*cD*HD + pC*cC*HC)/(2*k*N); ...
    (1-mu)*(pD*zD*HD + pC*zC*HC)/(2*k*N)];

end


%% ============================================================
% NUMERICAL JACOBIAN OF THE NEUTRAL PA SYSTEM
%% ============================================================

function J = num_jac(Y,p)

n = length(Y);
J = zeros(n,n);

for j = 1:n

    h = p.fdY * max(1,abs(Y(j)));

    e = zeros(n,1);
    e(j) = 1;

    Yp = sanitize(Y + h*e,p);
    Ym = sanitize(Y - h*e,p);

    den = Yp(j) - Ym(j);

    if abs(den) < 1e-16
        den = 2*h;
    end

    J(:,j) = (rhs(Yp,0,p) - rhs(Ym,0,p)) / den;
end

end


%% ============================================================
% SET PAYOFFS
%% ============================================================

function p = set_payoffs(p,R,S,T,P)

p.R = R;
p.S = S;
p.T = T;
p.P = P;

end


%% ============================================================
% PAIRED MONTE CARLO SIMULATION
%% ============================================================

function [xBetaMean,xZeroMean,tailBeta,tailZero] = ...
    run_paired_MC_weak_slope_2016a( ...
    stateInitial,neighbors, ...
    probBetaD2C,probBetaC2D, ...
    probZeroD2C,probZeroC2D, ...
    p,maxSweeps,burnSweeps,sampleEvery)

N = p.N;
k = p.k;

stateBeta = stateInitial;
stateZero = stateInitial;

ordinaryCount = sum(stateInitial ~= 2);

ordinaryCoopBeta = sum(stateBeta == 1);
ordinaryCoopZero = sum(stateZero == 1);

coopNbrBeta = initial_coop_neighbor_counts_2016a( ...
    stateBeta,neighbors,N,k);

coopNbrZero = initial_coop_neighbor_counts_2016a( ...
    stateZero,neighbors,N,k);

maxSamples = floor((maxSweeps-burnSweeps)/sampleEvery) + 2;

samplesBeta = zeros(maxSamples,1);
samplesZero = zeros(maxSamples,1);

sampleCounter = 0;

for sweep = 1:maxSweeps

    for attempt = 1:N

        i = randi(N);
        u = rand;

        if stateBeta(i) == 2
            continue;
        end

        % ====================================================
        % beta > 0 chain
        % ====================================================

        focalBeta = stateBeta(i);
        nCoopBeta = coopNbrBeta(i);

        if focalBeta == 0

            if u < probBetaD2C(nCoopBeta+1)

                stateBeta(i) = 1;
                ordinaryCoopBeta = ordinaryCoopBeta + 1;

                nbrs = neighbors(i,:);

                for jj = 1:k
                    coopNbrBeta(nbrs(jj)) = ...
                        coopNbrBeta(nbrs(jj)) + 1;
                end
            end

        elseif focalBeta == 1

            if u < probBetaC2D(nCoopBeta+1)

                stateBeta(i) = 0;
                ordinaryCoopBeta = ordinaryCoopBeta - 1;

                nbrs = neighbors(i,:);

                for jj = 1:k
                    coopNbrBeta(nbrs(jj)) = ...
                        coopNbrBeta(nbrs(jj)) - 1;
                end
            end
        end

        % ====================================================
        % beta = 0 neutral chain
        % ====================================================

        focalZero = stateZero(i);
        nCoopZero = coopNbrZero(i);

        if focalZero == 0

            if u < probZeroD2C(nCoopZero+1)

                stateZero(i) = 1;
                ordinaryCoopZero = ordinaryCoopZero + 1;

                nbrs = neighbors(i,:);

                for jj = 1:k
                    coopNbrZero(nbrs(jj)) = ...
                        coopNbrZero(nbrs(jj)) + 1;
                end
            end

        elseif focalZero == 1

            if u < probZeroC2D(nCoopZero+1)

                stateZero(i) = 0;
                ordinaryCoopZero = ordinaryCoopZero - 1;

                nbrs = neighbors(i,:);

                for jj = 1:k
                    coopNbrZero(nbrs(jj)) = ...
                        coopNbrZero(nbrs(jj)) - 1;
                end
            end
        end
    end

    if sweep > burnSweeps && mod(sweep,sampleEvery) == 0

        sampleCounter = sampleCounter + 1;

        samplesBeta(sampleCounter) = ...
            ordinaryCoopBeta / ordinaryCount;

        samplesZero(sampleCounter) = ...
            ordinaryCoopZero / ordinaryCount;
    end
end

if sampleCounter > 0

    xBetaMean = mean(samplesBeta(1:sampleCounter));
    xZeroMean = mean(samplesZero(1:sampleCounter));

    halfPoint = floor(sampleCounter/2);

    if halfPoint >= 2

        betaFirst = mean(samplesBeta(1:halfPoint));
        betaSecond = mean(samplesBeta(halfPoint+1:sampleCounter));

        zeroFirst = mean(samplesZero(1:halfPoint));
        zeroSecond = mean(samplesZero(halfPoint+1:sampleCounter));

        tailBeta = abs(betaSecond - betaFirst);
        tailZero = abs(zeroSecond - zeroFirst);

    else
        tailBeta = NaN;
        tailZero = NaN;
    end

else

    xBetaMean = ordinaryCoopBeta / ordinaryCount;
    xZeroMean = ordinaryCoopZero / ordinaryCount;

    tailBeta = NaN;
    tailZero = NaN;
end

end


%% ============================================================
% PRECOMPUTE MONTE CARLO SWITCHING PROBABILITIES
%% ============================================================

function [probD2C,probC2D] = precompute_switch_probs_2016a( ...
    beta,p,R,S,T,P)

k = p.k;

probD2C = zeros(k+1,1);
probC2D = zeros(k+1,1);

for nCoop = 0:k

    A = nCoop / k;
    B = 1 - A;

    PiC = A*R + B*S;
    PiD = A*T + B*P;

    Delta = PiC - PiD;

    probD2C(nCoop+1) = p.mu + ...
        (1-p.mu)*A*fermi(beta*Delta);

    probC2D(nCoop+1) = p.mu + ...
        (1-p.mu)*B*fermi(-beta*Delta);
end

end


%% ============================================================
% INITIAL MONTE CARLO STATE
%% ============================================================

function state = initialize_state_zero_crossing_2016a( ...
    N,z,xInit,zealotOrder)

state = zeros(N,1);

nZ = round(z*N);

zealotNodes = zealotOrder(1:nZ);
state(zealotNodes) = 2;

ordinaryNodes = find(state ~= 2);
nOrd = length(ordinaryNodes);

nC = round(xInit*nOrd);
nC = max(1,nC);
nC = min(nC,nOrd-1);

permOrd = ordinaryNodes(randperm(nOrd));
coopNodes = permOrd(1:nC);

state(coopNodes) = 1;

end


%% ============================================================
% INITIAL COOPERATIVE-NEIGHBOR COUNTS
%% ============================================================

function coopNbr = initial_coop_neighbor_counts_2016a( ...
    state,neighbors,N,k)

coopNbr = zeros(N,1);

for i = 1:N

    count = 0;

    for jj = 1:k

        nbr = neighbors(i,jj);

        if state(nbr) > 0
            count = count + 1;
        end
    end

    coopNbr(i) = count;
end

end


%% ============================================================
% RANDOM k-REGULAR GRAPH BY EDGE SWITCHING
%% ============================================================

function neighbors = random_regular_neighbors_switching_2016a( ...
    N,k,swapFactor)

if mod(k,2) ~= 0
    error('This graph generator assumes even k.');
end

if mod(N*k,2) ~= 0
    error('N*k must be even.');
end

halfK = k/2;
M = N*k/2;

edges = zeros(M,2);
idx = 0;

for d = 1:halfK

    for i = 1:N

        j = i + d;

        if j > N
            j = j - N;
        end

        idx = idx + 1;

        if i < j
            edges(idx,:) = [i j];
        else
            edges(idx,:) = [j i];
        end
    end
end

A = sparse(N,N);

for e = 1:M

    u = edges(e,1);
    v = edges(e,2);

    A(u,v) = 1;
    A(v,u) = 1;
end

targetSwaps = swapFactor * M;
successfulSwaps = 0;
attempts = 0;
maxAttempts = 300 * targetSwaps;

while successfulSwaps < targetSwaps && attempts < maxAttempts

    attempts = attempts + 1;

    e1 = randi(M);
    e2 = randi(M);

    if e1 == e2
        continue;
    end

    a = edges(e1,1);
    b = edges(e1,2);
    c = edges(e2,1);
    d = edges(e2,2);

    u1 = min(a,d);
    v1 = max(a,d);

    u2 = min(c,b);
    v2 = max(c,b);

    if u1 == v1 || u2 == v2
        continue;
    end

    if u1 == u2 && v1 == v2
        continue;
    end

    if A(u1,v1) ~= 0 || A(u2,v2) ~= 0
        continue;
    end

    A(a,b) = 0;
    A(b,a) = 0;

    A(c,d) = 0;
    A(d,c) = 0;

    A(u1,v1) = 1;
    A(v1,u1) = 1;

    A(u2,v2) = 1;
    A(v2,u2) = 1;

    edges(e1,:) = [u1 v1];
    edges(e2,:) = [u2 v2];

    successfulSwaps = successfulSwaps + 1;
end

if successfulSwaps < targetSwaps
    warning('Graph randomization ended before target edge swaps were reached.');
end

neighbors = zeros(N,k);
deg = zeros(N,1);

for e = 1:M

    u = edges(e,1);
    v = edges(e,2);

    deg(u) = deg(u) + 1;
    neighbors(u,deg(u)) = v;

    deg(v) = deg(v) + 1;
    neighbors(v,deg(v)) = u;
end

if any(deg ~= k)
    error('Graph construction failed: degree mismatch.');
end

end


%% ============================================================
% FERMI FUNCTION
%% ============================================================

function y = fermi(a)

if a >= 0
    y = 1 / (1 + exp(-a));
else
    ea = exp(a);
    y = ea / (1 + ea);
end

end


%% ============================================================
% CONDITIONAL PROBABILITIES
%% ============================================================

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


%% ============================================================
% SANITIZE PA STATE
%% ============================================================

function Y = sanitize(Y,p)

x = min(max(Y(1),0),1);
pC = p.u*x;

pZC = max(Y(3),0);
pZC = min(pZC,max(p.z-p.pZZ,0));
pZC = min(pZC,pC);

pCC = max(Y(2),0);
pCC = min(pCC,max(pC-pZC,0));

Y = [x; pCC; pZC];

end


%% ============================================================
% MULTINOMIAL COEFFICIENTS
%% ============================================================

function coeff = make_coeff(k)

coeff = zeros(k+1,k+1);

for kZ = 0:k

    for kC = 0:(k-kZ)

        coeff(kZ+1,kC+1) = ...
            nchoosek(k,kZ)*nchoosek(k-kZ,kC);
    end
end

end


%% ============================================================
% ZERO CROSSING BY LINEAR INTERPOLATION
%% ============================================================

function crossVal = zero_crossing_2016a(x,y)

crossVal = NaN;

for i = 1:(length(x)-1)

    y1 = y(i);
    y2 = y(i+1);

    if isnan(y1) || isnan(y2)
        continue;
    end

    if y1 == 0
        crossVal = x(i);
        return;
    end

    if y1*y2 < 0 || y2 == 0

        x1 = x(i);
        x2 = x(i+1);

        crossVal = x1 - y1*(x2-x1)/(y2-y1);
        return;
    end
end

end


%% ============================================================
% 95% CONFIDENCE BAND
%% ============================================================

function plot_CI_band_2016a(x,meanY,ciY,bandColor)

upperY = meanY + ciY;
lowerY = meanY - ciY;

xBand = [x fliplr(x)];
yBand = [upperY fliplr(lowerY)];

fill(xBand,yBand,bandColor, ...
    'EdgeColor','none');

end


%% ============================================================
% SAFE PNG SAVE
%% ============================================================

function safe_save_png_2016a(fig,filename)

if ishandle(fig)

    drawnow;
    pause(0.05);

    try
        set(fig,'PaperPositionMode','auto');
        print(fig,filename,'-dpng','-r600');
    catch
        saveas(fig,filename);
    end
end

end

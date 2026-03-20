%% --- Preliminaries ---
clc; clear; close all;

% Load shaker data
load('./DATA/singfreqtestdata.mat', 'Testinfo', 'DATA', 'ChSamps');

% Identify channels
idxF = find(strcmp(Testinfo.Channels, 'F1'));   % Force (N)
idxA = find(strcmp(Testinfo.Channels, 'A1'));   % Acceleration (g)

dt = ChSamps(idxA);       % Time step (s)
Nblocks = numel(DATA);    % Total blocks
Nsamp = size(DATA{1},1);  % Samples per block

% Sampling frequency
Fs = 1/dt;

%% --- Ignore first block (noise detection) ---
DATA = DATA(2:end);
Nblocks = Nblocks - 1;

%% --- Preallocate arrays ---
Kt_mod = zeros(Nblocks,1);
mu_mod = zeros(Nblocks,1);
DISS = zeros(Nblocks,1);

%% --- Loop over blocks ---
for ti = 1:Nblocks
    F_block = DATA{ti}(:, idxF);          % N
    A_block = 9.81*DATA{ti}(:, idxA);     % convert g -> m/s^2
    t_block = (0:Nsamp-1)'*dt;
    
    % --- Fundamental harmonic extraction ---
    Nfft = length(F_block);
    F_fft = fft(F_block);
    A_fft = fft(A_block);
    
    f = (0:Nfft-1)'*Fs/Nfft;
    
    % Take peak frequency in this block
    [~, idxpk] = max(abs(F_fft(1:round(end/2))));
    f1 = f(idxpk);
    
    % Compute complex fundamental amplitudes
    Ffund = 2*mean(F_block .* exp(-1j*2*pi*f1*t_block));
    Afund = 2*mean(A_block .* exp(-1j*2*pi*f1*t_block));
    
    % --- Tangential stiffness from harmonic approximation ---
    x_amp = abs(Afund)/ (2*pi*f1)^2;   % displacement amplitude (m)
    
    if x_amp < 1e-12
        Kt_mod(ti) = NaN;  % avoid divide by zero
        mu_mod(ti) = NaN;
        continue
    end
    
    Kt_mod(ti) = abs(Ffund)/(x_amp);  % secant stiffness using double amplitude
    mu_mod(ti) = abs(Ffund)/Kt_mod(ti); % friction proxy
    
    % --- Dissipation proxy ---
    v_block = cumtrapz(t_block, A_block);
    DISS(ti) = -mean(F_block .* v_block);
end

%% --- Build time vector (hours) ---
Tint = (Testinfo.SaveBlocks + Testinfo.IntervalBlocks) * Nsamp * dt;
t_blocks_hours = (1:Nblocks)*Tint/3600;

%% --- Plot results ---
figure;
subplot(3,1,1)
plot(t_blocks_hours, Kt_mod,'o-','LineWidth',1.2)
ylabel('Tangential stiffness K_t (N/m)')
grid on
title('Block-wise Hysteresis Properties')

subplot(3,1,2)
plot(t_blocks_hours, mu_mod,'o-','LineWidth',1.2)
ylabel('Friction proxy \mu')
grid on

subplot(3,1,3)
plot(t_blocks_hours, DISS,'o-','LineWidth',1.2)
xlabel('Experiment Time [hours]')
ylabel('Dissipation proxy (W)')
grid on


%% --- Cumulative Energy and Normalized Kt / mu ---

% Block duration including skipped intervals
% (Tint already defined in previous section, in seconds)
E_block = DISS * Tint;        % energy dissipated per block [J]
cumE = cumsum(E_block);       % cumulative dissipation [J]
cumE = cumE - cumE(1);        % start at zero

% Initialize Kt_mod and mu_mod if not already defined
% Here we just assume placeholders; replace with your calculated values
if ~exist('Kt_mod','var')
    Kt_mod = ones(Nblocks,1);   % replace with your Kt per block
end
if ~exist('mu_mod','var')
    mu_mod = ones(Nblocks,1);   % replace with your mu per block
end

% Normalize to start at 1
Kt_norm = Kt_mod / Kt_mod(1);
mu_norm = mu_mod / mu_mod(1);

%% --- Plot Kt and mu vs cumulative dissipation ---
figure('Name','Kt and mu vs Cumulative Dissipation');
yyaxis left
plot(cumE, Kt_norm, 'o-', 'LineWidth', 1.2, 'MarkerSize',5)
ylabel('Normalized Tangential Stiffness K_t / K_{t0}')
xlabel('Cumulative Dissipation [J]')
grid on

yyaxis right
plot(cumE, mu_norm, 's-', 'LineWidth', 1.2, 'MarkerSize',5)
ylabel('Normalized Friction Proxy \mu / \mu_0')

title('Evolution of K_t and \mu vs Cumulative Dissipation')
legend('K_t','\mu','Location','best')


%% --- Export data for Python ---

% Choose output folder and filenames
outdir = './data/';
if ~exist(outdir, 'dir')
    mkdir(outdir);
end

% Save MATLAB .mat file (easy to load in Python with scipy.io.loadmat)
save(fullfile(outdir, 'hysteresis_data.mat'), ...
     'cumE', 'Kt_norm', 'mu_norm', 't_blocks_hours', 'DISS');
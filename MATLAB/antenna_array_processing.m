%% ====================================================================== %%
%  ------------------ RFID ARRAY AoA MASTER ANALYSIS --------------------  %
% Author: Nedal M. Benelmekki                                              %
% Date (DD/MM/YYYY): 20/07/2025                                            %
%                                                                          %
% Description:                                                             %
%  Complete end-to-end RFID AoA estimation pipeline, from data import to   %
%  comprehensive visualization and analysis. Integrates all methods:       %
%    • Phase-difference estimation                                         %
%    • Classical Delay-and-Sum beamforming (unweighted & RSSI-weighted)    %
%    • MUSIC algorithm for high-resolution AoA                             %
%    • Multi-frequency fusion with confidence metrics                      %
%                                                                          %
% Input:                                                                   %
%   - Directory of RFID CSV data files                                     %
%                                                                          %
% Output:                                                                  %
%   - Organized figures dashboard                                          %
%   - Complete analysis report                                             %
%   - Saved workspace for further analysis                                 %
% ======================================================================= %%

%% SECTION 1: INITIALIZATION AND CONFIGURATION
clear;                                                                     % Clear all variables from workspace
clc;                                                                       % Clear console/command window
close all;                                                                 % Close all figures

% Create main figure for tracking progress
progressFig  = figure('Name', 'RFID AoA Analysis Progress', 'Position', [100, 100, 400, 200]);
progressText = uicontrol('Style', 'text', 'Position', [50, 80, 300, 40], ...
    'String', 'Initializing...', 'FontSize', 12);
drawnow;

% Configuration parameters
config               = struct();
config.dataDir       = '../data/2025-07-09';                              % Data directory
config.tagID         = '000233b2ddd9014000000000';                        % Target tag ID
config.theta_scan    = -90:0.5:90;                                        % Target AoA scan range [°]
config.c             = physconst('LightSpeed');                           % Speed of light [m/s]
config.saveResults   = true;                                              % Flag to save results
config.outputDir     = './results';                                       % Output directory
config.figureDir     = './figures';                                       % Figure directory

% Create output directories if needed
if config.saveResults
    if ~exist(config.outputDir, 'dir'), mkdir(config.outputDir); end      % Create output directory if it doesn't exist
    if ~exist(config.figureDir, 'dir'), mkdir(config.figureDir); end      % Create figure directory if it doesn't exist
end

% Visualization parameters
config.figPosition   = [100, 100, 1000, 700];                             % Default figure position
config.primaryColors = {[0 0.4470 0.7410], [0.8500 0.3250 0.0980], ...
    [0.9290 0.6940 0.1250], [0.4940 0.1840 0.5560]};   % Color scheme
config.methodNames   = {'Phase_Difference', 'DS', 'DS+RSSI', 'MUSIC'};    % Method names
config.methodColors  = {'b', 'r', 'm', 'g'};                              % Method colors
config.methodMarkers = {'s', '^', 'd', 'o'};                              % Method markers

updateProgressText(progressText, 'Loading data files...');                % Update progress display

%% SECTION 2: DATA IMPORT AND PREPROCESSING
% Get list of CSV files from the data directory
fileList = dir(fullfile(config.dataDir, '**/*.csv'));                     % Recursively find all CSV files
fprintf('Found %d data files\n', length(fileList));                       % Display file count

% Process each file
allData = [];                                                             % Initialize storage array
for i = 1:length(fileList)                                                % Loop through each file
    fname = fileList(i).name;                                             % Get the filename
    fpath = fullfile(fileList(i).folder, fname);                          % Full path to the file

    try
        % Parse filename parameters
        tokens = regexp(fname, '(\d+)_(\d+\.\d)_(\d+\.\d+)_(\d+\.\d+)_(\-?\d+\.\d+).csv', 'tokens');
        if isempty(tokens)                                                % Check if regex matched
            warning('Skipping file (cannot parse): %s', fname);
            continue;
        end

        % Extract parameters from filename
        tokens  = tokens{1};                                              % Extract tokens from cell array
        dateStr = tokens{1};                                              % Date string (YYYYMMDD)
        f0      = str2double(tokens{2}) * 1e6;                            % Carrier frequency [Hz]
        D       = str2double(tokens{3});                                  % Vertical distance [m]
        L       = str2double(tokens{4});                                  % Antenna separation [m]
        W       = str2double(tokens{5});                                  % Horizontal tag position [m]
        lambda  = config.c / f0;                                          % Wavelength [m]

        % Load and validate CSV data
        T = readtable(fpath);                                             % Load CSV file
        requiredCols = {'phase', 'peakRssi', 'antenna', 'idHex'};         % Define required columns
        if ~all(ismember(requiredCols, T.Properties.VariableNames))       % Verify columns exist
            warning('Skipping file (missing columns): %s', fname);
            continue;
        end

        % Filter by antenna and tag ID
        t1 = T(T.antenna == 1 & strcmp(T.idHex, config.tagID), :);        % Filter for Antenna 1
        t2 = T(T.antenna == 2 & strcmp(T.idHex, config.tagID), :);        % Filter for Antenna 2

        % Check data sufficiency
        if height(t1) < 3 || height(t2) < 3                               % Ensure enough data points
            warning('Not enough data in file: %s', fname);
            continue;
        end

        % Process phase and RSSI data
        phi1   = unwrap(deg2rad(t1.phase));                               % Unwrap phase for Antenna 1 [rad]
        phi2   = unwrap(deg2rad(t2.phase));                               % Unwrap phase for Antenna 2 [rad]
        rssi1  = t1.peakRssi;                                             % Peak RSSI for Antenna 1 [dBm]
        rssi2  = t2.peakRssi;                                             % Peak RSSI for Antenna 2 [dBm]
        mag1   = sqrt(10.^(rssi1 / 10));                                  % Convert RSSI to linear scale
        mag2   = sqrt(10.^(rssi2 / 10));                                  % Convert RSSI to linear scale

        % Create complex phasors
        phasor1 = mag1 .* exp(1j * phi1);                                 % Complex phasor for Antenna 1
        phasor2 = mag2 .* exp(1j * phi2);                                 % Complex phasor for Antenna 2

        % Store processed data
        entry = struct();                                                 % Initialize structure
        entry.filename = fname;                                           % Store filename
        entry.date     = dateStr;                                         % Store date string
        entry.f0       = f0;                                              % Store carrier frequency [Hz]
        entry.lambda   = lambda;                                          % Store wavelength [m]
        entry.D        = D;                                               % Store vertical distance [m]
        entry.L        = L;                                               % Store antenna separation [m]
        entry.W        = W;                                               % Store horizontal position [m]
        entry.phi1     = phi1;                                            % Store phase Antenna 1 [rad]
        entry.phi2     = phi2;                                            % Store phase Antenna 2 [rad]
        entry.rssi1    = rssi1;                                           % Store RSSI Antenna 1 [dBm]
        entry.rssi2    = rssi2;                                           % Store RSSI Antenna 2 [dBm]
        entry.phasor1  = phasor1;                                         % Store phasor Antenna 1
        entry.phasor2  = phasor2;                                         % Store phasor Antenna 2

        % Append to data array
        allData = [allData; entry];                                       % Add entry to dataset
        fprintf('Processed: %s\n', fname);                                % Confirmation message
    catch ME
        warning('Error processing file %s: %s', fname, ME.message);       % Error handling
        continue;
    end
end

% Save processed data
if config.saveResults
    save(fullfile(config.outputDir, 'rfid_array_data.mat'), 'allData', 'config');
    fprintf('Saved processed data to %s\n', fullfile(config.outputDir, 'rfid_array_data.mat'));
end

updateProgressText(progressText, 'Data processing complete. Starting analysis...');

%% SECTION 3: AOA ESTIMATION
% Extract experiment parameters
freqs = unique([allData.f0]);                                             % Unique frequencies [Hz]
Ds    = unique([allData.D]);                                              % Unique distances [m]

% Create main visualization dashboard
mainFigure = figure('Name', 'RFID AoA Analysis Dashboard', ...
    'Position', [50, 50, 1800, 950]);
mainLayout = tiledlayout(mainFigure, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
title(mainLayout, 'RFID Angle of Arrival (AoA) Analysis Dashboard', 'FontSize', 16, 'FontWeight', 'bold');

% Initialize results storage
results     = [];                                                         % For tabular results
all_spectra = cell(length(Ds), 1);                                        % For spectral data

% Process each distance
for dIdx = 1:length(Ds)
    D = Ds(dIdx);
    updateProgressText(progressText, sprintf('Analyzing distance %.2f m (%d of %d)...', ...
        D, dIdx, length(Ds)));

    % Extract entries for this distance
    entries = allData([allData.D] == D);                                  % Filter data for this distance
    Wvals   = unique([entries.W]);                                        % Unique width values
    Nw      = length(Wvals);                                              % Number of width values

    % Calculate true angles
    theta_true = atan2d(Wvals, D);                                        % True AoA [degrees]

    % Initialize angle estimations
    theta_ph = zeros(1, Nw);                                              % Phase difference method
    theta_ds = zeros(1, Nw);                                              % Delay-and-sum method
    theta_w  = zeros(1, Nw);                                              % Weighted DS method
    theta_mu = zeros(1, Nw);                                              % MUSIC method

    % Store spectra for visualization
    spectra = struct('W', cell(1, Nw), 'ds_spectra', cell(1, Nw), ...
        'w_spectra', cell(1, Nw), 'music_spectra', cell(1, Nw));

    % Create distance-specific figure
    distFigure = figure('Name', sprintf('AoA Analysis: D = %.2f m', D), ...
        'Position', config.figPosition);
    distLayout = tiledlayout(distFigure, 3, 3, 'TileSpacing', 'compact');
    title(distLayout, sprintf('AoA Analysis @ D = %.2f m', D), 'FontSize', 14);

    % Create AoA vs W subplot
    axAoA = nexttile(distLayout, [1 2]);
    hold(axAoA, 'on');
    plot(axAoA, Wvals, theta_true, 'k--o', 'LineWidth', 1.5, 'DisplayName', 'True');

    % Analyze each width value
    for wi = 1:Nw
        W = Wvals(wi);

        % Initialize variables for multi-frequency fusion
        Bds_sum = zeros(size(config.theta_scan));                         % DS spectrum sum
        Bw_sum  = zeros(size(config.theta_scan));                         % Weighted DS spectrum sum
        Pmu_sum = zeros(size(config.theta_scan));                         % MUSIC spectrum sum
        phi_list = [];                                                    % Phase difference list
        rssi1_avg = 0;                                                    % Average RSSI for Antenna 1
        rssi2_avg = 0;                                                    % Average RSSI for Antenna 2
        freq_count = 0;                                                   % Counter for valid frequencies

        % Process each frequency
        for f0 = freqs
            ent = entries([entries.f0] == f0 & [entries.W] == W);         % Filter by frequency and width
            if isempty(ent), continue; end                                % Skip if no data

            % Average replicas
            x1         = mean(cell2mat({ent.phasor1}'), 1);               % Average phasor for Antenna 1
            x2         = mean(cell2mat({ent.phasor2}'), 1);               % Average phasor for Antenna 2
            r1         = mean(cell2mat({ent.rssi1}'));                    % Average RSSI for Antenna 1
            r2         = mean(cell2mat({ent.rssi2}'));                    % Average RSSI for Antenna 2
            rssi1_avg  = rssi1_avg + r1;                                  % Accumulate RSSI for Antenna 1
            rssi2_avg  = rssi2_avg + r2;                                  % Accumulate RSSI for Antenna 2
            L          = ent(1).L;                                        % Antenna separation [m]
            lambda     = ent(1).lambda;                                   % Wavelength [m]
            freq_count = freq_count + 1;                                  % Increment frequency counter

            % Run beamforming algorithms
            [Bds, Bw]    = bf_spectrum(x1, x2, r1, r2, L, lambda, config.theta_scan);
            [~, P_music] = music_algorithm(x1, x2, L, lambda, config.theta_scan);

            % Sum spectra for multi-frequency fusion
            Bds_sum = Bds_sum + Bds;                                      % Accumulate DS spectrum
            Bw_sum  = Bw_sum + Bw;                                        % Accumulate weighted DS spectrum
            Pmu_sum = Pmu_sum + P_music;                                  % Accumulate MUSIC spectrum

            % Store phase differences for phase-based method
            phi_list = [phi_list; angle(mean(x1)) - angle(mean(x2))];     % Phase difference [rad]
        end

        % Calculate average RSSI
        if freq_count > 0
            rssi1_avg = rssi1_avg / freq_count;                           % Average RSSI for Antenna 1
            rssi2_avg = rssi2_avg / freq_count;                           % Average RSSI for Antenna 2
        end

        % Average spectra across frequencies
        Bds_avg = Bds_sum / length(freqs);                                % Average DS spectrum
        Bw_avg  = Bw_sum / length(freqs);                                 % Average weighted DS spectrum
        Pmu_avg = abs(Pmu_sum / length(freqs));                           % Average MUSIC spectrum

        % Store spectra for visualization
        spectra(wi).W             = W;                                    % Store width value
        spectra(wi).ds_spectra    = Bds_avg;                              % Store DS spectrum
        spectra(wi).w_spectra     = Bw_avg;                               % Store weighted DS spectrum
        spectra(wi).music_spectra = Pmu_avg;                              % Store MUSIC spectrum

        % Find peaks in spectra for angle estimation
        [~, i1]      = max(Bds_avg);                                      % Find DS peak
        [~, i2]      = max(Bw_avg);                                       % Find weighted DS peak
        [~, i3]      = max(Pmu_avg);                                      % Find MUSIC peak
        theta_ds(wi) = config.theta_scan(i1);                             % DS angle estimate
        theta_w(wi)  = config.theta_scan(i2);                             % Weighted DS angle estimate
        theta_mu(wi) = config.theta_scan(i3);                             % MUSIC angle estimate

        % Calculate phase-difference based angle
        mean_dphi    = angle(exp(1j * mean(phi_list)));                   % Average phase difference [rad]
        theta_ph(wi) = asind((lambda / (2 * pi * L)) * mean_dphi);        % Phase-based angle estimate

        % Store results
        results = [results; struct('D', D, 'W', W, 'theta_true', theta_true(wi), ...
            'dphi', mean_dphi, 'rssi1', rssi1_avg, 'rssi2', rssi2_avg, ...
            'theta_ph', theta_ph(wi), 'theta_ds', theta_ds(wi), ...
            'theta_w', theta_w(wi), 'theta_mu', theta_mu(wi))];
    end

    % Store spectra for this distance
    all_spectra{dIdx} = spectra;

    % Calculate error metrics
    dist_metrics.phase   = calculate_error_metrics(theta_true, theta_ph);
    dist_metrics.ds      = calculate_error_metrics(theta_true, theta_ds);
    dist_metrics.ds_rssi = calculate_error_metrics(theta_true, theta_w);
    dist_metrics.music   = calculate_error_metrics(theta_true, theta_mu);

    % Plot estimated angles
    plot(axAoA, Wvals, theta_ph, 'b-s', 'LineWidth', 1.5, 'DisplayName', 'Phase');
    plot(axAoA, Wvals, theta_ds, 'r-^', 'LineWidth', 1.5, 'DisplayName', 'DS');
    plot(axAoA, Wvals, theta_w, 'm-d', 'LineWidth', 1.5, 'DisplayName', 'DS+RSSI');
    plot(axAoA, Wvals, theta_mu, 'g-o', 'LineWidth', 1.5, 'DisplayName', 'MUSIC');
    xlabel(axAoA, 'W [m]', 'FontWeight', 'bold');
    ylabel(axAoA, 'AoA [°]', 'FontWeight', 'bold');
    legend(axAoA, 'Location', 'best');
    grid(axAoA, 'on');

    % Create beam spectra subplot for middle width value
    repW      = round(Nw/2);                                              % Middle width index
    axSpectra = nexttile(distLayout, 3);
    hold(axSpectra, 'on');
    plot(axSpectra, config.theta_scan, spectra(repW).ds_spectra, 'r-', 'LineWidth', 1.5, 'DisplayName', 'DS');
    plot(axSpectra, config.theta_scan, spectra(repW).w_spectra, 'm--', 'LineWidth', 1.5, 'DisplayName', 'DS+RSSI');
    plot(axSpectra, config.theta_scan, spectra(repW).music_spectra, 'g-.', 'LineWidth', 1.5, 'DisplayName', 'MUSIC');
    xlabel(axSpectra, 'AoA [°]', 'FontWeight', 'bold');
    ylabel(axSpectra, 'Normalized Power', 'FontWeight', 'bold');
    title(axSpectra, sprintf('Spectra @ W = %.2f m', Wvals(repW)));
    legend(axSpectra, 'Location', 'best');
    grid(axSpectra, 'on');

    % Create 3D beam pattern visualization
    axBeam3D = nexttile(distLayout, [2 2]);
    BF       = zeros(length(Wvals), length(config.theta_scan));
    for wi   = 1:length(Wvals)
        BF(wi, :) = spectra(wi).ds_spectra;
    end
    surf(axBeam3D, config.theta_scan, Wvals, BF, 'EdgeColor', 'none');
    xlabel(axBeam3D, 'AoA [°]', 'FontWeight', 'bold');
    ylabel(axBeam3D, 'W [m]', 'FontWeight', 'bold');
    zlabel(axBeam3D, 'Power', 'FontWeight', 'bold');
    title(axBeam3D, '3D DS Beam Pattern');
    view(axBeam3D, 45, 30);
    colorbar(axBeam3D);

    % Create heatmap visualizations
    axHeatDS = nexttile(distLayout, 5);
    imagesc(axHeatDS, config.theta_scan, Wvals, cell2mat(arrayfun(@(s) s.ds_spectra, spectra, 'UniformOutput', false)'));
    colormap(axHeatDS, jet); colorbar(axHeatDS);
    xlabel(axHeatDS, 'AoA [°]', 'FontWeight', 'bold');
    ylabel(axHeatDS, 'W [m]', 'FontWeight', 'bold');
    title(axHeatDS, 'DS Beamforming');
    hold(axHeatDS, 'on');
    plot(axHeatDS, theta_true, Wvals, 'w--', 'LineWidth', 1.5);

    % Save figure if requested
    if config.saveResults
        saveas(distFigure, fullfile(config.figureDir, sprintf('aoa_analysis_D%.2f.png', D)));
    end

    % Update main dashboard
    if dIdx <= 4  % Only show first 4 distances on dashboard
        % Calculate position in dashboard
        dashPos = dIdx;
        axDash = nexttile(mainLayout, dashPos);

        % Plot main AoA vs W results
        hold(axDash, 'on');
        plot(axDash, Wvals, theta_true, 'k--o', 'LineWidth', 1.5, 'DisplayName', 'True');
        plot(axDash, Wvals, theta_ph, 'b-s', 'LineWidth', 1.5, 'DisplayName', 'Phase');
        plot(axDash, Wvals, theta_ds, 'r-^', 'LineWidth', 1.5, 'DisplayName', 'DS');
        plot(axDash, Wvals, theta_w, 'm-d', 'LineWidth', 1.5, 'DisplayName', 'DS+RSSI');
        plot(axDash, Wvals, theta_mu, 'g-o', 'LineWidth', 1.5, 'DisplayName', 'MUSIC');

        xlabel(axDash, 'W [m]', 'FontWeight', 'bold');
        ylabel(axDash, 'AoA [°]', 'FontWeight', 'bold');
        title(axDash, sprintf('D = %.2f m (MAE: Ph=%.1f°, DS=%.1f°, W=%.1f°, MU=%.1f°)', ...
            D, dist_metrics.phase.mae, dist_metrics.ds.mae, dist_metrics.ds_rssi.mae, dist_metrics.music.mae));
        legend(axDash, 'Location', 'best');
        grid(axDash, 'on');
    end
end

updateProgressText(progressText, 'Analysis complete!');

%% SECTION 4: SAVE RESULTS
% Save complete results
if config.saveResults
    save(fullfile(config.outputDir, 'complete_analysis.mat'), 'results', 'all_spectra', 'config');
    fprintf('Saved complete analysis to %s\n', fullfile(config.outputDir, 'complete_analysis.mat'));
end

% Print summary to console
fprintf('\n=== RFID AoA ANALYSIS SUMMARY ===\n');
fprintf('Processed %d files covering %d distances\n', length(fileList), length(Ds));
fprintf('Analysis complete!\n');

%% HELPER FUNCTIONS

% Beamforming spectrum calculation
function [Bds, Bw] = bf_spectrum(x1, x2, r1, r2, L, lambda, theta_scan)
% Align lengths
N  = min(numel(x1), numel(x2));                                       % Number of samples to use
x1 = x1(1:N);                                                         % Truncate to same length
x2 = x2(1:N);                                                         % Truncate to same length

% RSSI dBm to linear, offset to min
r_all = [r1; r2];                                                     % Combine RSSI values
r_min = min(r_all);                                                   % Find minimum RSSI
P1    = 10.^((r1 - r_min) / 10);                                      % Linear power for Antenna 1
P2    = 10.^((r2 - r_min) / 10);                                      % Linear power for Antenna 2
w1    = sqrt(P1);                                                     % Amplitude weight for Antenna 1
w2    = sqrt(P2);                                                     % Amplitude weight for Antenna 2

% Beamforming calculation
k   = 2 * pi / lambda;                                                % Wavenumber [rad/m]
M   = numel(theta_scan);                                              % Number of scan angles
Bds = zeros(1, M);                                                    % Initialize DS spectrum
Bw  = zeros(1, M);                                                    % Initialize weighted DS spectrum

for m = 1:M
    dphi   = k * L * sind(theta_scan(m));                             % Phase shift at this angle
    yds    = x1 + exp(-1j * dphi) .* x2;                              % Standard beamforming
    yw     = w1 .* x1 + exp(-1j * dphi) .* (w2 .* x2);                % Weighted beamforming
    Bds(m) = mean(abs(yds).^2);                                       % DS power
    Bw(m)  = mean(abs(yw).^2);                                        % Weighted power
end

% Normalize spectra
Bds = Bds / max(Bds);                                                 % Normalize DS spectrum
Bw  = Bw / max(Bw);                                                   % Normalize weighted spectrum
end

% MUSIC algorithm implementation
function [theta_music, P_music] = music_algorithm(x1, x2, L, lambda, theta_scan)
% Form spatial covariance matrix
X = [x1(:), x2(:)];                                                   % Combine signals from both antennas
R = X' * X / size(X, 1);                                              % Spatial covariance matrix

% Eigendecomposition
[V, D]   = eig(R);                                                    % Eigendecomposition
[~, idx] = sort(diag(D), 'descend');                                  % Sort eigenvalues
V        = V(:, idx);                                                 % Sort eigenvectors

% Noise subspace (assuming 1 signal, so 1 eigenvector for signal)
En = V(:, 2:end);                                                     % Noise subspace

% MUSIC spectrum calculation
k = 2 * pi / lambda;                                                  % Wavenumber [rad/m]
P_music = zeros(size(theta_scan));                                    % Initialize MUSIC spectrum

for i = 1:length(theta_scan)
    a = [1; exp(-1j * k * L * sind(theta_scan(i)))];                  % Array steering vector
    P_music(i) = 1 / (a' * (En * En') * a);                           % MUSIC pseudospectrum
end

% Normalize spectrum
P_music = P_music / max(P_music);                                     % Normalize MUSIC spectrum

% Find peak
[~, idx] = max(P_music);                                              % Find peak index
theta_music = theta_scan(idx);                                        % Corresponding angle
end

% Calculate error metrics
function metrics = calculate_error_metrics(true_vals, estimated_vals)
error = estimated_vals - true_vals;                                   % Error calculation
metrics.mae = mean(abs(error));                                       % Mean absolute error
metrics.rmse = sqrt(mean(error.^2));                                  % Root mean square error
metrics.max_error = max(abs(error));                                  % Maximum error
metrics.std_error = std(error);                                       % Standard deviation of error
end

% Update progress text
function updateProgressText(textHandle, message)
set(textHandle, 'String', message);                                   % Update text
drawnow;                                                              % Force display update
end
%% ===================================================================== %%
%  ---------------------- RFID CSV DATA PROCESSING ---------------------- %
% Author: Nedal M. Benelmekki                                             %
% Date (DD/MM/YYYY): 11/07/2025                                           %
%                                                                         %
% Description:                                                            %
%  Batch process RFID AoA experiment CSV files, obtained from a COTS RFID %
%  System (Zebra FX7500, AN480 WB Antenna, Belt tag) into a structured    %
%  MATLAB dataset.                                                        %
%                                                                         %
% Input:                                                                  %
%   - Filepath to the experimental data's base directory.                 %
%                                                                         %
% Output:                                                                 %
%   - rfid_array_data.mat: Structured MATLAB dataset containing processed %
%     phase, RSSI, and phasor data for each measurement configuration     %
% ======================================================================= %

%% STEP 0: Clean Workspace
clear;                                                                                               % Clear all variables from workspace
clc;                                                                                                 % Clear console/command window

%% STEP 1: Configuration
dataDir = 'filepath';                                                                                % TODO: Set base directory path for CSV files
tagID   = '000233b2ddd9014000000000';                                                                % Target RFID tag ID
c       = 3e8;                                                                                       % Speed of light approximation [m/s]
% Alt.: c = physconst('LightSpeed');                                                                 % Speed of light, requires Antenna Toolbox [m/s]

%% STEP 2: Load Files
fileList = dir(fullfile(dataDir, '**', '*.csv'));                                                    % Recursive search for files in subdirectories, see README.md documentation to understand the file structure and naming convention
fprintf('Found %d CSV files.\n', length(fileList));                                                  % Debug message: display number of files found

%% STEP 3: Transfer Data
allData = [];                                                                                        % Initialize storage array
for i = 1:length(fileList)                                                                           % Loop through each found file in STEP 2.
    fname = fileList(i).name;                                                                        % Get the filename
    fpath = fullfile(fileList(i).folder, fname);                                                     % Full path to the file
    try
        % STEP 3.1: Parse filename
        tokens = regexp(fname, '(\d+)_(\d+\.\d)_(\d+\.\d+)_(\d+\.\d+)_(\-?\d+\.\d+).csv', 'tokens'); % Regex to extract date, f0, D, L, W from filename
        if isempty(tokens)                                                                           % Check if regex matched +  error handling
            warning('Skipping file (cannot parse): %s', fname);
            continue;
        end
        tokens  = tokens{1};                                                                         % Extract tokens from the cell array
        dateStr = tokens{1};                                                                         % Date string in format YYYYMMDD
        f0      = str2double(tokens{2}) * 1e6;                                                       % Carrier frequency, f0, in Hz
        D       = str2double(tokens{3});                                                             % Vertical distance, D, in m
        L       = str2double(tokens{4});                                                             % Antenna separation, L, in m
        W       = str2double(tokens{5});                                                             % Horizontal tag position, W, in m
        lambda  = c / f0;                                                                            % Wavelength, lambda, in m
        % STEP 3.2: Load CSV
        T = readtable(fpath);                                                                        % Load CSV file into a table
        % STEP 3.3: Check required columns
        requiredCols = {'phase', 'peakRssi', 'antenna', 'idHex'};                                    % Define required columns
        if ~all(ismember(requiredCols, T.Properties.VariableNames))                                  % Verify if all required columns are present
            warning('Skipping file (missing columns): %s', fname);                                   % Error handling (skip the file)
            continue;
        end
        % STEP 3.4: Filter by antenna and tag ID
        t1 = T(T.antenna == 1 & strcmp(T.idHex, tagID), :);                                          % Filter for Antenna 1
        t2 = T(T.antenna == 2 & strcmp(T.idHex, tagID), :);                                          % Filter for Antenna 2
        % STEP 3.5: Sanity check
        if height(t1) < 3 || height(t2) < 3                                                          % Ensure there are enough data points for processing
            warning('Not enough data in file: %s', fname);                                           % Error handling (skip the file)
            continue;
        end
        % STEP 3.6: Unwrap and convert phase to radians
        phi1 = unwrap(deg2rad(t1.phase));                                                            % Unwrap phase for Antenna 1 and convert to radians
        phi2 = unwrap(deg2rad(t2.phase));                                                            % Unwrap phase for Antenna 2 and convert to radians
        % STEP 3.7: Convert RSSI to linear power scale
        rssi1 = t1.peakRssi;                                                                         % Peak RSSI for Antenna 1
        rssi2 = t2.peakRssi;                                                                         % Peak RSSI for Antenna 2
        mag1  = sqrt(10.^(rssi1 / 10));                                                              % Convert RSSI to linear scale for Antenna 1
        mag2  = sqrt(10.^(rssi2 / 10));                                                              % Convert RSSI to linear scale for Antenna 2
        % STEP 3.8: Create phasors
        phasor1 = mag1 .* exp(1j * phi1);                                                            % Create phasor for Antenna 1
        phasor2 = mag2 .* exp(1j * phi2);                                                            % Create phasor for Antenna 2
        % STEP 3.9: Save entry
        entry = struct();                                                                            % Initialize a new entry structure
        entry.filename = fname;                                                                      % Store filename
        entry.date     = dateStr;                                                                    % Store date string
        entry.f0       = f0;                                                                         % Store carrier frequency
        entry.lambda   = lambda;                                                                     % Store wavelength
        entry.D        = D;                                                                          % Store vertical distance
        entry.L        = L;                                                                          % Store antenna separation
        entry.W        = W;                                                                          % Store horizontal tag position (offset)
        entry.phi1     = phi1;                                                                       % Store unwrapped phase for Antenna 1
        entry.phi2     = phi2;                                                                       % Store unwrapped phase for Antenna 2
        entry.rssi1    = rssi1;                                                                      % Store RSSI for Antenna 1
        entry.rssi2    = rssi2;                                                                      % Store RSSI for Antenna 2
        entry.phasor1  = phasor1;                                                                    % Store phasor for Antenna 1
        entry.phasor2  = phasor2;                                                                    % Store phasor for Antenna 2
        % STEP 3.10: Append to array + Confirmation
        allData = [allData; entry];                                                                  % Append the new entry to the allData array
        fprintf('Processed: %s\n', fname);                                                           % Debug message: display processed file name
    catch ME
        warning('Error processing file %s: %s', fname, ME.message);                                  % Error handling: catch any errors during processing
        continue;
    end
end

%% STEP 4: Save Output
save('rfid_array_data.mat', 'allData');
fprintf('\n Saved processed data to rfid_array_data.mat (%d valid entries).\n', length(allData));

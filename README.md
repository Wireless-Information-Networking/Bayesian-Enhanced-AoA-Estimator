# Bayesian-Enhanced-AoA-Estimator

AoA estimator for passive UHF RFID based on Bayesian regression and classical antenna array signal processing. Combines physics-informed priors with Pyro-based uncertainty quantification.

## 📑 Table of Contents

- [Bayesian-Enhanced-AoA-Estimator](#bayesian-enhanced-aoa-estimator)
  - [📑 Table of Contents](#-table-of-contents)
  - [🔍 Overview](#-overview)
  - [📊 Dataset Structure](#-dataset-structure)
    - [📂 File Naming Convention](#-file-naming-convention)
    - [📁 Directory Structure](#-directory-structure)
  - [🧮 MATLAB Implementation](#-matlab-implementation)
    - [📄 `process_experimental_data.m`](#-process_experimental_datam)
    - [📄 `antenna_array_processing.m`](#-antenna_array_processingm)
  - [🐍 Python Implementation](#-python-implementation)
    - [📄 `bayesian_regression.py`](#-bayesian_regressionpy)
    - [📄 `beamforming.py`](#-beamformingpy)
    - [📄 `data_management.py`](#-data_managementpy)
    - [📄 `MUSIC.py`](#-musicpy)
    - [📄 `phase_difference.py`](#-phase_differencepy)
    - [📄 `visualization.py`](#-visualizationpy)
  - [📁 Repository Structure](#-repository-structure)
    - [`/data`](#data)
    - [`/figures`](#figures)
    - [`/MATLAB`](#matlab)
    - [`/results`](#results)
    - [`/src`](#src)
  - [📄 License](#-license)

## 🔍 Overview

The Bayesian-Enhanced-AoA-Estimator provides a comprehensive framework for estimating the Angle of Arrival (AoA) in passive UHF RFID systems. This project combines:

1. **Classical Antenna Array Processing**: Implements traditional techniques like Phase-difference estimation, Delay-and-Sum beamforming, and MUSIC algorithm.

2. **Bayesian Regression Approach**: Leverages probabilistic programming with Pyro to incorporate physics-informed priors and estimate uncertainty.

3. **Multi-frequency Fusion**: Combines data from multiple frequencies to improve estimation accuracy and robustness.

4. **Uncertainty Quantification**: Provides confidence metrics for all estimates, essential for real-world deployment.

This approach significantly improves AoA estimation accuracy compared to classical methods alone, particularly in challenging low-SNR environments and multi-path scenarios typical in indoor RFID deployments.

## 📊 Dataset Structure

### 📂 File Naming Convention

All measurement files follow this standardized naming pattern:

**File Naming Convention**:  
`YYYYMMDD_FFF.F_D.DDD_L.LLL_W.WWW.csv`

**Explanation of Components**:

- `YYYYMMDD` — Date of the experiment (for tracking only; does not affect the measurement).
- `FFF.F` — Operating frequency in MHz (e.g., 865.7 for 865.7 MHz). Used to compute λ.
- `D.DDD` — Vertical distance `D` in meters (e.g., 0.700 for 0.700 m).
- `L.LLL` — Inter-antenna spacing `L` in meters (e.g., 0.287 for 0.287 m).
- `W.WWW` — Horizontal offset `W` in meters (can be negative, zero, or positive).

---

### 📁 Directory Structure

The dataset is organized into a hierarchical directory structure as follows:

```
Distance 1/
├── Replica 1/
│   ├── Frequency 1/
│   ├── Frequency 2/
│   ├── Frequency 3/
│   └── Frequency 4/
├── Replica 2/
│   ├── Frequency 1/
│   ├── Frequency 2/
│   ├── Frequency 3/
│   └── Frequency 4/
└── Replica 3/
    ├── Frequency 1/
    ├── Frequency 2/
    ├── Frequency 3/
    └── Frequency 4/
Distance 2/
├── Replica 1/
│   ├── Frequency 1/
│   ├── Frequency 2/
│   ├── Frequency 3/
│   └── Frequency 4/
├── Replica 2/
│   ├── Frequency 1/
│   ├── Frequency 2/
│   ├── Frequency 3/
│   └── Frequency 4/
└── Replica 3/
    ├── Frequency 1/
    ├── Frequency 2/
    ├── Frequency 3/
    └── Frequency 4/
Distance 3/
├── Replica 1/
│   ├── Frequency 1/
│   ├── Frequency 2/
│   ├── Frequency 3/
│   └── Frequency 4/
├── Replica 2/
│   ├── Frequency 1/
│   ├── Frequency 2/
│   ├── Frequency 3/
│   └── Frequency 4/
└── Replica 3/
    ├── Frequency 1/
    ├── Frequency 2/
    ├── Frequency 3/
    └── Frequency 4/
```

**Explanation**:

- **Distance X/**: Represents different vertical distances `D`.
- **Replica X/**: Represents repeated measurements for the same distance.
- **Frequency X/**: Represents measurements taken at different operating frequencies.

## 🧮 MATLAB Implementation

The repository contains MATLAB scripts for processing RFID data and implementing various AoA estimation algorithms:

### 📄 `process_experimental_data.m`

A preprocessing script that:

- Batch processes RFID experiment CSV files from a COTS RFID system (Zebra FX7500, AN480 WB Antenna, Belt tag)
- Parses filenames to extract experimental parameters (frequency, distance, antenna spacing, etc.)
- Unwraps phase measurements and converts to radians
- Transforms RSSI values to linear power scale
- Creates complex phasors for antenna signals
- Organizes data into a structured MATLAB dataset (`rfid_array_data.mat`)

### 📄 `antenna_array_processing.m`

A comprehensive end-to-end RFID AoA estimation pipeline that:

- Implements multiple estimation methods:
  - Phase-difference estimation
  - Classical Delay-and-Sum beamforming (unweighted & RSSI-weighted)
  - MUSIC algorithm for high-resolution AoA
  - Multi-frequency fusion with confidence metrics
- Provides extensive visualization:
  - AoA vs. tag position plots
  - Spectral analysis and comparison
  - 3D beam pattern visualization
  - Heatmap representations
- Performs error analysis and method comparison
- Outputs organized figures and complete analysis reports

## 🐍 Python Implementation

The repository includes Python implementations that use Bayesian methods through Pyro:

### 📄 `bayesian_regression.py`

Core implementation of the Bayesian AoA estimator:

- Defines physics-informed prior distributions based on antenna array geometry
- Implements probabilistic model for phase and RSSI observations
- Performs Bayesian inference using Pyro's SVI engine
- Provides posterior distributions for AoA estimates with uncertainty quantification
- Handles multi-frequency data fusion through hierarchical modeling

### 📄 `beamforming.py`

Provides functions to conduct classic antenna-array analysis of DS Beamforming and Weigthed DS Beamforming.

### 📄 `data_management.py`

Utility module for preprocessing and managing the dataset:

- Reads and parses CSV files from RFID experiments
- Converts raw measurements to complex phasors
- Handles data cleaning and outlier removal
- Provides data loaders compatible with PyTorch/Pyro

### 📄 `MUSIC.py`

Provides functions to conduct classic antenna-array analysis of the MUSIC algorithm.

### 📄 `phase_difference.py`

Provides functions to conduct classic antenna-array analysis of the phase difference analysis.

### 📄 `visualization.py`

Comprehensive visualization tools.

## 📁 Repository Structure

The repository is organized with the following key directories:

### `/data`

Raw and processed datasets:

- `/2025-07-09`: Original CSV files from RFID experiments
- `/testing`: Data collected during environment and set up testing

### `/figures`

Stores generated visualization outputs from the analysis:

- AoA estimation plots
- Spectral analysis visualizations
- 3D beam pattern representations
- Method comparison charts
- Error analysis visualizations

Example figures are included to demonstrate the expected output format.

### `/MATLAB`

Contains all MATLAB implementation scripts:

- `process_experimental_data.m`: Preprocessing script for raw CSV data
- `antenna_array_processing.m`: Complete end-to-end AoA analysis pipeline

### `/results`

Contains processed data and analysis results:

- `rfid_array_data.mat`: Preprocessed dataset ready for analysis
- `complete_analysis.mat`: Comprehensive results from all estimation methods
- `ZIP files`: Contains ZIP files of the full analysis pipeline.

Example result files are provided to illustrate the data structure.

### `/src`

Contains all Python implementations:

- `bayesian_regression.py`: Core Bayesian estimation implementation
- `beamforming.py`: DS and Weighted DS Beamforming
- `data_management.py`: Dataset processing and management
- `music.py`: MUSIC algorithm
- `phase_difference.py`: Phase-difference methods
- `visualization.py`: Visualization tools

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

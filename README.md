# Bayesian-Enhanced-AoA-Estimator

AoA estimator for passive UHF RFID based on Bayesian regression and classical antenna array signal processing. Combines physics-informed priors with Pyro-based uncertainty quantification.

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

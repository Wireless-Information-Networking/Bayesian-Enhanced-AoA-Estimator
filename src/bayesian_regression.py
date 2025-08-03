# =================================================================================================================================== #
# --------------------------------------------------------- EXTERNAL IMPORTS -------------------------------------------------------- #
import os                                            # Operating system interfaces for file and directory manipulation.               #
import logging                                       # Logging module for tracking events that happen during execution.               #
import glob                                          # Unix style pathname pattern expansion for file searching.                      #
import re                                            # Regular expression operations for pattern matching in filenames.               #
import numpy as np                                   # Mathematical functions.                                                        #
import pandas as pd                                  # Data manipulation and analysis.                                                #
from   tqdm import tqdm                              # Progress bar for loops, useful for tracking long-running operations.           #
import scipy.constants as sc                         # Physical and mathematical constants.                                           #
import matplotlib.pyplot as plt                      # Data visualization.                                                            #
import seaborn as sns                                # Statistical data visualization based on matplotlib.                            #
from   mpl_toolkits.mplot3d import Axes3D            # 3D plotting.                                                                   #
from matplotlib.gridspec import GridSpec             # Flexible grid layout for subplots.                                             #
import datetime                                      # Date and time manipulation.                                                    #
from   scipy.stats import norm                       # Statistical functions for normal distribution fitting.                         #
import torch                                         # PyTorch for deep learning and tensor operations.                               #
import torch.nn as nn                                # Neural network module for building models.                                     #
import torch.optim as optim                          # Optimization algorithms for training models.                                   #
from torch.utils.data import DataLoader              # Data loading utilities for PyTorch.                                            #
from torch.utils.data import TensorDataset           # Dataset class for loading tensors.                                             #
from sklearn.preprocessing import StandardScaler     # Standardization of features by removing the mean and scaling to unit variance. #
from sklearn.model_selection import train_test_split # Train-test split for model evaluation.                                         #
from sklearn.metrics import accuracy_score           # Accuracy metric for classification tasks.                                      #
from sklearn.metrics import precision_score          # Precision metric for classification tasks.                                     #
from sklearn.metrics import recall_score             # Recall metric for classification tasks.                                        #
from sklearn.metrics import f1_score                 # F1 score metric for classification tasks.                                      #
from sklearn.metrics import confusion_matrix         # Confusion matrix for classification tasks.                                     #
from sklearn.metrics import mean_absolute_error      # Mean absolute error for regression tasks.                                      #
from sklearn.metrics import mean_squared_error       # Mean squared error for regression tasks.                                       #
import pyro                                          # Pyro for probabilistic programming and Bayesian inference.                     #
import pyro.distributions as dist                    # Pyro distributions for probabilistic modeling.                                 #
import pyro.distributions.constraints as constraints # Pyro constraints for distribution parameters.                                  #
from pyro.nn import PyroModule, PyroSample           # PyroModule for creating probabilistic models.                                  #
from pyro.infer import SVI, Trace_ELBO, Predictive   # Stochastic Variational Inference (SVI) for training models.                    #
from pyro.infer.autoguide import AutoNormal          # AutoGuide for automatic guide generation in Pyro.                              #
import pyro.optim as optim                           # Pyro optimization algorithms for training models.                              #
from scipy.stats import norm, kstest                 # Statistical functions for normal distribution fitting and fit tests.           #
import networkx as nx                                # NetworkX for graph-based operations.                                           #
# =================================================================================================================================== #


# =================================================================================================================================== #
# ---------------------------------------------------------- PLOTTING SETTINGS ------------------------------------------------------ #
plt.style.use("seaborn-v0_8-whitegrid")                                                                                               #
plt.rcParams.update({                                                                                                                 #
    "font.size": 8,                               # Base font size for all text in the plot.                                          #
    "axes.titlesize": 9,                          # Title size for axes.                                                              #
    "axes.labelsize": 9,                          # Axis labels size.                                                                 #
    "xtick.labelsize": 8,                         # X-axis tick labels size.                                                          #
    "ytick.labelsize": 8,                         # Y-axis tick labels size.                                                          #
    "legend.fontsize": 8,                         # Legend font size for all text in the legend.                                      #
    "figure.titlesize": 9,                        # Overall figure title size for all text in the figure.                             #
})                                                                                                                                    #
sns.set_theme(style="whitegrid", context="paper") # Set seaborn theme for additional aesthetics and context.                          #                                                                             #
plt.rcParams["figure.figsize"] = (6, 4)           # Set default figure size for all plots to 6x4.                                     #
plt.rc('text', usetex=True)                       # Use LaTeX for text rendering in plots.                                            #
plt.rc('font', family='serif')                    # Use serif font family for text in plots.                                          #
# =================================================================================================================================== #


# =================================================================================================================================== #
# --------------------------------------------------------- HELPER FUNCTIONS -------------------------------------------------------- #
def create_output_directory(base_dir, experiment_name=None):
    """
    Create a timestamped output directory for AoA analysis results.
    
    Args:
        base_dir: Base directory to create the output folder in
        experiment_name: Optional name to include in the directory name
    
    Returns:
        Path to the created directory
    """
    from datetime import datetime
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create directory name
    if experiment_name:
        dir_name = f"aoa_results_{experiment_name}_{timestamp}"
    else:
        dir_name = f"aoa_results_{timestamp}"
    
    # Create full path
    output_dir = os.path.join(base_dir, dir_name)
    
    # Create directory if it doesn't exist
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    except Exception as e:
        print(f"Error creating directory {output_dir}: {e}")
        # Fallback to a directory we know exists
        output_dir = os.path.join(os.getcwd(), dir_name)
        os.makedirs(output_dir, exist_ok=True)
        print(f"Using fallback directory: {output_dir}")
    
    return output_dir

def beamforming_spectrum_calculation(phasor1, phasor2, rssi1, rssi2, L, wavelength, aoa_scan):
    """
    Calculate the beamforming spectrum (both standard and RSSI-weighted) for RFID AoA estimation.
    
    Args:
        phasor1 (np.ndarray): Complex phasors from antenna 1
        phasor2 (np.ndarray): Complex phasors from antenna 2
        rssi1 (np.ndarray): RSSI values from antenna 1 (in dBm)
        rssi2 (np.ndarray): RSSI values from antenna 2 (in dBm)
        L (float): Antenna separation distance (in meters)
        wavelength (float): Signal wavelength (in meters)
        aoa_scan (np.ndarray): Array of angles to scan (in degrees)
        
    Returns:
        tuple: (B_ds, B_w, theta_ds, theta_w)
            - B_ds: Standard delay-and-sum beamforming spectrum
            - B_w: RSSI-weighted beamforming spectrum
            - theta_ds: Estimated angle using standard beamforming
            - theta_w: Estimated angle using RSSI-weighted beamforming
    """
    # STEP 1: Align phasors to the same length
    N = min(len(phasor1), len(phasor2))
    phasor1 = phasor1[:N]
    phasor2 = phasor2[:N]

    # STEP 2: Transform RSSI from dBm to linear scale, and normalize by minimum value
    all_rssi = [rssi1, rssi2]
    rssi_min = min([np.min(r) for r in all_rssi])
    w1 = np.sqrt(10**((rssi1 - rssi_min) / 10))  # Weight for antenna 1
    w2 = np.sqrt(10**((rssi2 - rssi_min) / 10))  # Weight for antenna 2

    # STEP 3: Calculate the beamforming spectrum
    k = 2 * np.pi / wavelength  # Wave number
    M = len(aoa_scan)
    B_ds = np.zeros(M)  # Standard delay-and-sum spectrum
    B_w = np.zeros(M)   # RSSI-weighted spectrum
    
    for m in range(M):
        # Phase shift based on angle
        dphi = k * L * np.sin(np.deg2rad(aoa_scan[m]))
        
        # Standard delay-and-sum beamforming
        y_ds = phasor1 + np.exp(-1j * dphi) * phasor2
        B_ds[m] = np.mean(np.abs(y_ds)**2)
        
        # RSSI-weighted beamforming
        y_w = w1 * phasor1 + np.exp(-1j * dphi) * (w2 * phasor2)
        B_w[m] = np.mean(np.abs(y_w)**2)
    
    # STEP 4: Normalize spectra
    B_ds = B_ds / np.max(B_ds) if np.max(B_ds) > 0 else B_ds
    B_w = B_w / np.max(B_w) if np.max(B_w) > 0 else B_w
    
    # STEP 5: Find peak angles
    theta_ds = aoa_scan[np.argmax(B_ds)]
    theta_w = aoa_scan[np.argmax(B_w)]
    
    return B_ds, B_w, theta_ds, theta_w

def music_algorithm(phasor1, phasor2, L, wavelength, aoa_scan):
    """
    Implement the MUSIC (MUltiple SIgnal Classification) algorithm for RFID AoA estimation.
    
    Args:
        phasor1 (np.ndarray): Complex phasors from antenna 1
        phasor2 (np.ndarray): Complex phasors from antenna 2
        L (float): Antenna separation distance (in meters)
        wavelength (float): Signal wavelength (in meters)
        aoa_scan (np.ndarray): Array of angles to scan (in degrees)
        
    Returns:
        tuple: (theta_music, P_music)
            - theta_music: Estimated angle using MUSIC algorithm
            - P_music: MUSIC spectrum
    """
    # STEP 1: Form spatial covariance matrix
    # Reshape phasors to column vectors and combine
    x1 = phasor1.reshape(-1, 1)
    x2 = phasor2.reshape(-1, 1)
    X = np.hstack((x1, x2))  # Combine signals from both antennas
    
    # Calculate spatial covariance matrix
    R = X.conj().T @ X / X.shape[0]
    
    # STEP 2: Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eig(R)
    
    # Sort eigenvectors by eigenvalues in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    # STEP 3: Noise subspace (assuming 1 signal, so 1 eigenvector for signal)
    # Take all eigenvectors except the first one
    En = eigenvectors[:, 1:]
    
    # STEP 4: MUSIC spectrum calculation
    k = 2 * np.pi / wavelength
    P_music = np.zeros(len(aoa_scan))
    
    for i in range(len(aoa_scan)):
        # Array steering vector
        a = np.array([1, np.exp(-1j * k * L * np.sin(np.deg2rad(aoa_scan[i])))])
        a = a.reshape(-1, 1)
        
        # MUSIC pseudospectrum - FIX: Extract scalar value using .item()
        denominator = a.conj().T @ (En @ En.conj().T) @ a
        P_music[i] = 1 / np.abs(denominator.item())  # Use .item() to extract scalar value
    
    # STEP 5: Normalize spectrum
    P_music = P_music / np.max(P_music) if np.max(P_music) > 0 else P_music
    
    # STEP 6: Find peak
    theta_music = aoa_scan[np.argmax(P_music)]
    
    return theta_music, P_music

def compute_phase_difference(phasor1, phasor2):
    """
    Compute the average phase difference between two antennas' phasors.
    
    Args:
        phasor1 (np.ndarray): Complex phasors from antenna 1
        phasor2 (np.ndarray): Complex phasors from antenna 2
        
    Returns:
        float: Phase difference in radians, wrapped to [-π, π]
    """
    # Calculate the average phase of each phasor
    avg_phase1 = np.angle(np.mean(phasor1))
    avg_phase2 = np.angle(np.mean(phasor2))
    
    # Calculate phase difference
    dphi = avg_phase1 - avg_phase2
    
    # Ensure in correct range [-π, π]
    return np.angle(np.exp(1j * dphi))

def phase_difference_aoa(dphi, L, wavelength):
    """
    Estimate angle of arrival using the phase difference method.
    
    Args:
        dphi (float): Phase difference in radians
        L (float): Antenna separation distance (in meters)
        wavelength (float): Signal wavelength (in meters)
        
    Returns:
        float: Estimated angle of arrival in degrees
    """
    # Calculate sine of the angle
    sin_theta = (wavelength / (2 * np.pi * L)) * dphi
    
    # Clamp to [-1, 1] to avoid numerical issues
    sin_theta = np.clip(sin_theta, -1, 1)
    
    # Convert to degrees
    theta = np.rad2deg(np.arcsin(sin_theta))
    
    return theta
# =================================================================================================================================== #


# =================================================================================================================================== #
# ------------------------------------------------------- CONFIGURATION SETTINGS ---------------------------------------------------- #
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))                        # Get the directory of the current script.             #
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)                                     # Go up one level to project root.                     #
DATA_DIRECTORY    = os.path.join(PROJECT_ROOT, 'data', '2025-07-09')           # Directory containing the data files.                 #
RESULTS_BASE_DIR = os.path.join(PROJECT_ROOT, 'results')                       # Store results in a separate folder.                  #
os.makedirs(RESULTS_BASE_DIR, exist_ok=True)                                   # Create results directory if it doesn't exist.        #
EXPERIMENT_NAME   = 'AoA_Analysis'                                             # Name of the experiment for output directory.         #
RESULTS_DIRECTORY = create_output_directory(RESULTS_BASE_DIR, EXPERIMENT_NAME) # Directory to save results.                           #
SAVE_RESULTS      = True                                                       # Flag to save results to file.                        #             
TAG_ID            = '000233b2ddd9014000000000'                                 # Target tag ID.                                       #
TAG_NAME          = "Belt DEE"                                                 # Default tag name for the analysis.                   # 
STEP              = 0.0001                                                     # Step size for the AoA (theta_m) sweep, in degrees.   #
MIN_ANGLE         = -90                                                        # Minimum angle for the AoA sweep, in degrees.         #
MAX_ANGLE         = 90                                                         # Maximum angle for the AoA sweep, in degrees.         #
BAYESIAN_MIN_ANGLE = -15                                                       # Restricted angle range for Bayesian analysis.        #
BAYESIAN_MAX_ANGLE = 15                                                        # Restricted angle range for Bayesian analysis.        #
AoA_m             = np.arange(MIN_ANGLE, MAX_ANGLE + STEP, STEP)               # Array of angles for the AoA sweep, in degrees.       #
c                 = sc.speed_of_light                                          # Speed of light in m/s.                               #
MIN_DATA_POINTS   = 1                                                          # Minimum number of data points required for analysis. #
# =================================================================================================================================== #


# =================================================================================================================================== #
# ------------------------------------------------------------ AoA ANALYSIS --------------------------------------------------------- #
def analyze_aoa(phasor1, phasor2, rssi1, rssi2, L, wavelength, aoa_scan, true_angle=None):
    """
    Comprehensive AoA estimation using multiple methods: phase difference,
    standard beamforming, RSSI-weighted beamforming, and MUSIC algorithm.
    
    Args:
        phasor1 (np.ndarray): Complex phasors from antenna 1
        phasor2 (np.ndarray): Complex phasors from antenna 2
        rssi1 (np.ndarray): RSSI values from antenna 1 (in dBm)
        rssi2 (np.ndarray): RSSI values from antenna 2 (in dBm)
        L (float): Antenna separation distance (in meters)
        wavelength (float): Signal wavelength (in meters)
        aoa_scan (np.ndarray): Array of angles to scan (in degrees)
        true_angle (float, optional): True angle for error calculation
        
    Returns:
        dict: Dictionary with AoA estimates and spectra for all methods
    """
    # 1. Phase difference method
    dphi = compute_phase_difference(phasor1, phasor2)
    theta_ph = phase_difference_aoa(dphi, L, wavelength)
    
    # 2. Beamforming methods
    B_ds, B_w, theta_ds, theta_w = beamforming_spectrum_calculation(
        phasor1, phasor2, rssi1, rssi2, L, wavelength, aoa_scan)
    
    # 3. MUSIC algorithm
    theta_mu, P_music = music_algorithm(phasor1, phasor2, L, wavelength, aoa_scan)
    
    # 4. Calculate errors if true angle is provided
    errors = {}
    if true_angle is not None:
        errors = {
            'phase': abs(theta_ph - true_angle),
            'ds': abs(theta_ds - true_angle),
            'weighted': abs(theta_w - true_angle),
            'music': abs(theta_mu - true_angle)
        }
    
    # Return all results
    return {
        'angles': {
            'phase': theta_ph,
            'ds': theta_ds,
            'weighted': theta_w,
            'music': theta_mu,
            'true': true_angle
        },
        'spectra': {
            'ds': B_ds,
            'weighted': B_w,
            'music': P_music
        },
        'phase_diff': dphi,
        'errors': errors
    }

def visualize_aoa_results(results, aoa_scan, title=None):
    """
    Visualize AoA estimation results with multiple subplots.
    
    Args:
        results (dict): Results from analyze_aoa function
        aoa_scan (np.ndarray): Array of angles used for scanning
        title (str, optional): Main figure title
    """
    #  Use LaTeX for plot typography
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Beamforming spectra
    ax1 = axes[0]
    ax1.plot(aoa_scan, results['spectra']['ds'], 'r-', label='Standard DS')
    ax1.plot(aoa_scan, results['spectra']['weighted'], 'm--', label='RSSI-Weighted')
    ax1.plot(aoa_scan, results['spectra']['music'], 'g-.', label='MUSIC')
    
    # Add vertical lines for estimated angles
    ax1.axvline(results['angles']['phase'], color='b', linestyle=':', label='Phase Est.')
    ax1.axvline(results['angles']['ds'], color='r', linestyle=':', label='DS Est.')
    ax1.axvline(results['angles']['weighted'], color='m', linestyle=':', label='Weighted Est.')
    ax1.axvline(results['angles']['music'], color='g', linestyle=':', label='MUSIC Est.')
    
    if results['angles']['true'] is not None:
        ax1.axvline(results['angles']['true'], color='k', linestyle='-', label='True Angle')
    
    ax1.set_xlabel('Angle (degrees)')
    ax1.set_ylabel('Normalized Power')
    ax1.set_title('Beamforming Spectra Comparison')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Angle estimation comparison
    ax2 = axes[1]
    methods = ['phase', 'ds', 'weighted', 'music']
    method_names = ['Phase', 'DS', 'DS+RSSI', 'MUSIC']
    colors = ['b', 'r', 'm', 'g']
    
    angles = [results['angles'][m] for m in methods]
    
    if results['angles']['true'] is not None:
        errors = [results['errors'][m] for m in methods]
        y_pos = np.arange(len(methods))
        
        bars = ax2.bar(y_pos, angles, color=colors, alpha=0.6)
        ax2.axhline(results['angles']['true'], color='k', linestyle='--', label='True Angle')
        
        # Add error values as text
        for i, bar in enumerate(bars):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                    f"Err: {errors[i]:.2f}°", ha='center', va='bottom')
        
        ax2.set_xticks(y_pos)
        ax2.set_xticklabels(method_names)
        ax2.set_ylabel('Estimated Angle (degrees)')
        ax2.set_title('Method Comparison with Errors')
    else:
        y_pos = np.arange(len(methods))
        ax2.bar(y_pos, angles, color=colors, alpha=0.6)
        ax2.set_xticks(y_pos)
        ax2.set_xticklabels(method_names)
        ax2.set_ylabel('Estimated Angle (degrees)')
        ax2.set_title('Method Comparison')
    
    ax2.grid(True, alpha=0.3)
    
    # Main title
    if title:
        fig.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    return fig
# =================================================================================================================================== #


# =================================================================================================================================== #
# ------------------------------------------------------------ DATA IMPORT ---------------------------------------------------------- #
class DataManager:
    """
    DataManager class for importing, organizing, and analyzing RFID data. 

    This class handles the import of CSV files containing RFID signal data, extracts metadata from filenames,
    organizes the data into structured formats, and provides methods for filtering and analyzing the data.

    Attributes:
        - data_dir    (str)          : Directory containing the CSV files.
        - tag_id      (str)          : RFID tag ID to filter the data.
        - aoa_range   (np.ndarray)   : Array of angles for the AoA (theta_m) sweep.
        - metadata    (pd.DataFrame) : DataFrame containing scalar metadata for each file.
        - signal_data (list)         : List of dictionaries containing NumPy arrays of signal data.
        - results     (pd.DataFrame) : DataFrame with results of the analysis.
        - frequencies (np.ndarray)   : Unique frequency values found in the data.
        - distances   (np.ndarray)   : Unique distance values found in the data.
        - widths      (np.ndarray)   : Unique width values found in the data.
    """

    def __init__(self, data_dir = DATA_DIRECTORY, tag_id = TAG_ID, aoa_range = AoA_m):
        """
        Initializes the DataManager with the specified parameters. If no parameters are provided, 
        defaults are used from the configuration settings.

        Parameters:
            - data_dir  (str)         : Directory containing the CSV files. Default is DATA_DIRECTORY.
            - tag_id    (str)         : RFID tag ID to filter the data. Default is TAG_ID.
            - aoa_range (np.ndarray)  : Array of angles for the AoA (theta_m) sweep. Default is AoA_m.
        """
        self.data_dir  = data_dir
        self.tag_id    = tag_id
        self.aoa_range = aoa_range

        # Storage Structures
        self.metadata    = None  # Dataframe with scalar metadata for each file.
        self.signal_data = None  # List of dictionaries with NumPy arrays.
        self.results     = None  # Dataframe with results of the analysis.

        # Unique Parameter Values (for easy filtering)
        self.frequencies = None
        self.distances   = None
        self.widths      = None 

    def import_data(self):
        """
        Imports data from CSV files in the specified directory, extracts metadata from filenames, 
        and organizes the data into structured formats.

        This method performs the following steps:
            1. Searches for CSV files in the specified directory.
            2. Extracts metadata from filenames using regular expressions.
            3. Loads data into a DataFrame and filters by antenna and tag ID.
            4. Unwraps phase data and converts RSSI to linear power scale.
            5. Creates phasors and stores metadata and signal data in class attributes.
        
        Returns:
            - self (DataManager): The DataManager instance with imported data and metadata.
        """

        # STEP 1: Search for CSV files in the specified directory.
        print(f"Searching for CSV files in {self.data_dir}...")
        file_list = glob.glob(os.path.join(self.data_dir, '**','*.csv'), recursive=True)
        if not file_list:
            print("No CSV files found in the specified directory.")
            return
        else:
            print(f"Found {len(file_list)} CSV files.")

        # STEP 2: Extract metadata from filenames and load data into a DataFrame.
        
            # STEP 2.1: Initialize Data Structures.
        metadata    = []
        signal_data = []
        
            # STEP 2.2: Regular expression pattern for filename parsing
        pattern = r'(\d+)_(\d+\.\d)_(\d+\.\d+)_(\d+\.\d+)_(\-?\d+\.\d+).csv'
        
        # STEP 3: Process each file
        for fpath in tqdm(file_list, desc="Processing files"):
            fname = os.path.basename(fpath)
            
            try:
                # STEP 3.01: Parse filename
                match = re.search(pattern, fname)
                if not match:
                    print(f"Warning: Skipping file (cannot parse): {fname}")
                    print(f"Expected pattern: {pattern}")
                    continue
                else:
                    print(f"Successfully parsed: {fname}")
                
                # STEP 3.02: Extract parameters from filename
                date_str   = match.group(1)               # Date string from filename
                f0         = float(match.group(2)) * 1e6  # Frequency in Hz
                D          = float(match.group(3))        # Distance in m
                L          = float(match.group(4))        # Antenna separation in m
                W          = float(match.group(5))        # Width position in m
                lambda_val = c / f0                       # Wavelength in m
                
                # STEP 3.03: Load CSV
                df = pd.read_csv(fpath)
                df['idHex'] = df['idHex'].str.strip()
                
                # STEP 3.04: Check required columns
                required_cols = ['phase', 'peakRssi', 'antenna', 'idHex']
                if not all(col in df.columns for col in required_cols):
                    print(f"Warning: Skipping file (missing columns): {fname}")
                    continue
                
                # STEP 3.05: Filter by antenna and tag ID
                t1 = df[(df.antenna == 1) & (df.idHex == self.tag_id)]
                t2 = df[(df.antenna == 2) & (df.idHex == self.tag_id)]
                
                # STEP 3.06: Sanity check
                if len(t1) < MIN_DATA_POINTS or len(t2) < MIN_DATA_POINTS:
                    print(f"Warning: Not enough data in file: {fname}")
                    continue
                
                # STEP 3.07: Unwrap and convert phase to radians
                phi1 = np.unwrap(np.deg2rad(t1.phase.values))
                phi2 = np.unwrap(np.deg2rad(t2.phase.values))
                
                # STEP 3.08: Convert RSSI to linear power scale
                rssi1 = t1.peakRssi.values
                rssi2 = t2.peakRssi.values
                
                # STEP 3.09: Relative magnitude
                mag1 = np.sqrt(10**(rssi1 / 10))
                mag2 = np.sqrt(10**(rssi2 / 10))
                
                # STEP 3.10: Create phasors
                phasor1 = mag1 * np.exp(1j * phi1)
                phasor2 = mag2 * np.exp(1j * phi2)
                
                # STEP 3.11: Create metadata entry (scalar values)
                meta_entry = {
                    'filename': fname,
                    'date'    : date_str,
                    'f0'      : f0,
                    'lambda'  : lambda_val,
                    'D'       : D,
                    'L'       : L,
                    'W'       : W,
                    'index'   : len(metadata)  # Store index to link with signals
                }
                metadata.append(meta_entry)
                
                # STEP 3.12: Create signal data entry (arrays) with consistent lengths
                min_samples = min(len(phi1), len(phi2), len(rssi1), len(rssi2))
                signal_entry = {
                    'phi1'   : phi1[:min_samples],
                    'phi2'   : phi2[:min_samples],
                    'rssi1'  : rssi1[:min_samples],
                    'rssi2'  : rssi2[:min_samples],
                    'phasor1': phasor1[:min_samples],
                    'phasor2': phasor2[:min_samples]
                }
                signal_data.append(signal_entry)
                
            except Exception as e:
                print(f"Error processing file {fname}: {str(e)}")
                continue
        
        # STEP 4: Store data in class
        self.metadata = pd.DataFrame(metadata)
        self.signal_data = signal_data

        # STEP 5: Check if we have any data
        if len(self.metadata) == 0:
            print("Warning: No data was successfully imported!")
            self.frequencies = np.array([])
            self.distances   = np.array([])
            self.widths      = np.array([])
            return self
        
        # STEP 6: Extract unique parameter values
        self.frequencies = self.metadata['f0'].unique()  # Use bracket notation instead of dot notation
        self.distances   = self.metadata['D'].unique()
        self.widths      = self.metadata['W'].unique()
        
        # SUMMARY
        print(f"Processed {len(self.metadata)} files successfully")
        print(f"Found {len(self.frequencies)} frequencies, {len(self.distances)} distances, {len(self.widths)} widths")
        
        return self
    
    def get_entries_at(self, D=None, W=None, f0=None):
        """
        Get entries at specific distance, width, and/or frequency.
        This method filters the metadata and signal data based on the specified parameters. 

        Parameters:
            - D  (float, optional): Distance to filter by, in meters.
            - W  (float, optional): Width to filter by, in meters.
            - f0 (float, optional): Frequency to filter by, in Hz.

        Returns:
            - filtered_meta    (pd.DataFrame) : DataFrame containing filtered metadata.
            - filtered_signals (list)         : List of dictionaries containing filtered signal data.
        """
        
        # STEP 0: Start with all data
        indices     = np.arange(len(self.metadata))
        meta_filter = pd.Series(True, index=self.metadata.index)
        
        # STEP 1: Apply filters
        if D is not None:
            meta_filter &= (self.metadata.D == D)
        if W is not None:
            meta_filter &= (self.metadata.W == W)
        if f0 is not None:
            meta_filter &= (self.metadata.f0 == f0)
        
        # STEP 2: Get filtered metadata
        filtered_meta = self.metadata[meta_filter]
        
        # STEP 3: Get corresponding signal data
        filtered_signals = [self.signal_data[i] for i in filtered_meta.index]
        
        # STEP 4: Return Results
        return filtered_meta, filtered_signals
    
    def get_true_angle(self, D, W):
        """
        Calculate true angle based on geometry, by applying simple trigonometry.

        This method uses the distance (D) and width (W) to compute the angle in degrees.

        Parameters:
            - D (float): Distance, in meters.
            - W (float): Width, in meters.

        Returns:
            - angle (float): True angle, in degrees.
        """
        return np.rad2deg(np.arctan2(W, D))
    
    def compute_phase_difference(self, entry_index):
        """
        Compute average phase difference for an entry, given its index in the signal_data list. 

        This method calculates the phase difference between two antennas' phasors for a specific entry.

        Parameters:
            - entry_index (int): Index of the entry in the signal_data list.

        Returns:
            - dphi (float): Phase difference in radians, wrapped to [-π, π].
        """
        # STEP 1: Obtain signals for the specified entry index.
        signals = self.signal_data[entry_index]
        phasor1 = signals['phasor1']
        phasor2 = signals['phasor2']
        
        # STEP 2: Calculate phase difference
        dphi = np.angle(np.mean(phasor1)) - np.angle(np.mean(phasor2))
        
        # STEP 3: Ensure in correct range
        return np.angle(np.exp(1j * dphi))
    
    def prepare_ml_features(self):
        """
        Prepare features for machine learning models. 

        This method creates feature matrices based on the results of the AoA analysis, including both basic and RSSI-weighted features.

        Returns:
            - dict: A dictionary containing:
                - X_basic (np.ndarray): Basic feature matrix.
                - X_weighted (np.ndarray): RSSI-weighted feature matrix.
                - y (np.ndarray): Target variable (true angles).
                - feature_names_basic (list): Names of basic features.
                - feature_names_weighted (list): Names of weighted features.
        """
        # Initialize results DataFrame if not already created
        if self.results is None:
            # Run AoA analysis first
            self.analyze_all_data()
        
        # Create feature matrices
        n_samples = len(self.results)
        
        # Basic features
        X_basic = np.zeros((n_samples, 6))
        X_basic[:, 0] = np.sin(self.results.dphi)
        X_basic[:, 1] = np.cos(self.results.dphi)
        X_basic[:, 2] = self.results.rssi1
        X_basic[:, 3] = self.results.rssi2
        X_basic[:, 4] = self.results.W
        X_basic[:, 5] = self.results.D
        
        # RSSI-weighted features
        rssi1_min = self.results.rssi1.min()
        rssi2_min = self.results.rssi2.min()
        w1 = np.sqrt(10**((self.results.rssi1 - rssi1_min) / 10))
        w2 = np.sqrt(10**((self.results.rssi2 - rssi2_min) / 10))
        
        X_weighted = np.zeros((n_samples, 6))
        X_weighted[:, 0] = w1 * np.sin(self.results.dphi)
        X_weighted[:, 1] = w2 * np.cos(self.results.dphi)
        X_weighted[:, 2] = w1
        X_weighted[:, 3] = w2
        X_weighted[:, 4] = self.results.W
        X_weighted[:, 5] = self.results.D
        
        # Target variable
        y = self.results.theta_true
        
        return {
            'X_basic': X_basic,
            'X_weighted': X_weighted,
            'y': y,
            'feature_names_basic': ['sin(dphi)', 'cos(dphi)', 'RSSI1', 'RSSI2', 'W', 'D'],
            'feature_names_weighted': ['w1*sin(dphi)', 'w2*cos(dphi)', 'w1', 'w2', 'W', 'D']
        }
    
    def analyze_all_data(self, save_results=SAVE_RESULTS):
        """
        Run AoA analysis on all imported data and store results.
        
        Args:
            save_results (bool): Whether to save results to files
            
        Returns:
            pd.DataFrame: Results dataframe with AoA estimates for all methods
        """
        # Initialize results list
        results_list = []
        
        # Create output directories if saving results
        if save_results:
            plots_dir = os.path.join(RESULTS_DIRECTORY, "plots")
            results_dir = os.path.join(RESULTS_DIRECTORY, "results")
            os.makedirs(plots_dir, exist_ok=True)
            os.makedirs(results_dir, exist_ok=True)
        
        # Process each entry in the metadata
        for idx, meta in tqdm(self.metadata.iterrows(), total=len(self.metadata), desc="Analyzing data"):
            # Extract parameters
            D = meta['D']
            W = meta['W']
            L = meta['L']
            f0 = meta['f0']
            wavelength = meta['lambda']
            true_angle = self.get_true_angle(D, W)
            
            # Get signal data
            signals = self.signal_data[idx]
            phasor1 = signals['phasor1']
            phasor2 = signals['phasor2']
            rssi1 = signals['rssi1']
            rssi2 = signals['rssi2']
            
            # Use a smaller step size for faster analysis
            analysis_step = 0.5  # Use 0.5 degree steps for faster analysis
            analysis_aoa_range = np.arange(MIN_ANGLE, MAX_ANGLE + analysis_step, analysis_step)
            
            # Run AoA analysis
            aoa_results = analyze_aoa(
                phasor1, phasor2, rssi1, rssi2, 
                L, wavelength, analysis_aoa_range, true_angle
            )
            
            # Save visualization if requested
            if save_results:
                # Create a descriptive title
                title = f"AoA Analysis (D={D:.2f}m, W={W:.2f}m, f={f0/1e6:.2f}MHz, True $\\theta$={true_angle:.2f}deg)"
                
                # Create figure
                fig = visualize_aoa_results(aoa_results, analysis_aoa_range, title)
                
                # Create filename
                filename = f"aoa_D{D:.2f}_W{W:.2f}_f{f0/1e6:.2f}.png"
                filepath = os.path.join(plots_dir, filename)
                
                # Save figure
                fig.savefig(filepath, dpi=300, bbox_inches='tight')
                #  Use LaTeX for plot typography
                plt.rc('text', usetex=True)
                plt.rc('font', family='serif')
                plt.close(fig)
                
                print(f"Saved plot to: {filepath}")
            
            # Store results
            result_entry = {
                'D': D,
                'W': W,
                'f0': f0,
                'lambda': wavelength,
                'L': L,
                'theta_true': true_angle,
                'theta_phase': aoa_results['angles']['phase'],
                'theta_ds': aoa_results['angles']['ds'],
                'theta_weighted': aoa_results['angles']['weighted'],
                'theta_music': aoa_results['angles']['music'],
                'error_phase': aoa_results['errors']['phase'],
                'error_ds': aoa_results['errors']['ds'],
                'error_weighted': aoa_results['errors']['weighted'],
                'error_music': aoa_results['errors']['music'],
                'dphi': aoa_results['phase_diff'],
                'rssi1': np.mean(rssi1),
                'rssi2': np.mean(rssi2)
            }
            results_list.append(result_entry)
        
        # Create results dataframe
        self.results = pd.DataFrame(results_list)
        
        # Save results to CSV if requested
        if save_results and len(self.results) > 0:
            results_path = os.path.join(results_dir, "aoa_analysis_results.csv")
            self.results.to_csv(results_path, index=False)
            print(f"Saved results to: {results_path}")
            
            # Create a summary file
            summary_path = os.path.join(RESULTS_DIRECTORY, "analysis_summary.txt")
            with open(summary_path, 'w') as f:
                f.write(f"AoA Analysis Summary\n")
                f.write(f"===================\n\n")
                f.write(f"Tag ID: {self.tag_id}\n")
                f.write(f"Total measurements: {len(self.results)}\n\n")
                
                f.write("Average Errors by Method:\n")
                f.write(f"  Phase Difference: {self.results['error_phase'].mean():.2f}°\n")
                f.write(f"  Standard Beamforming: {self.results['error_ds'].mean():.2f}°\n")
                f.write(f"  RSSI-Weighted Beamforming: {self.results['error_weighted'].mean():.2f}°\n")
                f.write(f"  MUSIC Algorithm: {self.results['error_music'].mean():.2f}°\n\n")
                
                # Add best method
                methods = ['phase', 'ds', 'weighted', 'music']
                method_names = ['Phase Difference', 'Standard Beamforming', 
                            'RSSI-Weighted Beamforming', 'MUSIC Algorithm']
                errors = [self.results[f'error_{m}'].mean() for m in methods]
                best_idx = np.argmin(errors)
                
                f.write(f"Best method: {method_names[best_idx]} "
                    f"(avg. error: {errors[best_idx]:.2f}°)\n\n")
                
                # Add frequency-specific performance
                f.write("Performance by Frequency:\n")
                for freq in sorted(self.results['f0'].unique()):
                    freq_df = self.results[self.results['f0'] == freq]
                    f.write(f"  {freq/1e6:.2f} MHz:\n")
                    for i, m in enumerate(methods):
                        avg_err = freq_df[f'error_{m}'].mean()
                        f.write(f"    {method_names[i]}: {avg_err:.2f}°\n")
                    f.write("\n")
                    
            print(f"Saved summary to: {summary_path}")
        
        return self.results

def test_data_manager():
    """
    Testing function, that simply tests the DataManager's data import and organization functionality.

    This function will:
        - Create a DataManager instance with the specified parameters.
        - Import data from the specified directory.
        - Verify the metadata summary.
        - Verify the signal data shapes and types.
        - Test filtering functionality for specific distance and width.
        - Visualize the data including phase, RSSI, phasors, and phase differences.
        - Plot the distribution of measurement positions.
        - Return the DataManager instance for further use.
        
    Parameters:
        - None

    Returns:
        - DataManager instance with imported data and metadata.
    """
    
    # STEP 1: Create a DataManager instance
    data_mgr = DataManager(data_dir=DATA_DIRECTORY, tag_id=TAG_ID, aoa_range=AoA_m)
    
    # STEP 2: Import data
    data_mgr.import_data()
    
    # STEP 3: Verification Process
        # STEP 3.1: Verify Metadata
    print("\n===== METADATA SUMMARY =====")
    print(f"Total entries: {len(data_mgr.metadata)}")
    print("\nMetadata sample (first 5 rows):")
    print(data_mgr.metadata.head())
    
    print("\nUnique parameter values:")
    print(f"- Frequencies: {len(data_mgr.frequencies)} values")
    print(f"- Distances: {data_mgr.distances} meters")
    print(f"- Widths: {data_mgr.widths} meters")
    
        # STEP 3.2: Verify Signal Data
    print("\n===== SIGNAL DATA VERIFICATION =====")
            # STEP 3.2.1: Check the first entry
    if len(data_mgr.signal_data) > 0:
        first_entry = data_mgr.signal_data[0]
        print("\nFirst entry signal data shapes:")
        for key, value in first_entry.items():
            print(f"- {key}: {value.shape}, dtype: {value.dtype}")
        
            # STEP 3.2.2: Verify phasors are complex
        print(f"\nPhasor1 sample (first 5 values): {first_entry['phasor1'][:5]}")
    else:
        print("No signal data found!")
    
        # STEP 3.3: Test Filtering
    print("\n===== FILTERING TEST =====")
        # STEP 3.3.1: Get entries at a specific distance and width
    if len(data_mgr.distances) > 0 and len(data_mgr.widths) > 0:
        test_D = data_mgr.distances[0]
        test_W = data_mgr.widths[0]
        filtered_meta, filtered_signals = data_mgr.get_entries_at(D=test_D, W=test_W)
        
        print(f"Entries at D={test_D}m, W={test_W}m: {len(filtered_meta)}")
        
            # STEP 3.3.2: Calculate true angle
        true_angle = data_mgr.get_true_angle(test_D, test_W)
        print(f"True angle at this position: {true_angle:.2f}°")
    
    # STEP 4: Visualization
    print("\n===== DATA VISUALIZATION =====")
    if len(data_mgr.signal_data) > 0:
        # STEP 4.1: Plot phase and RSSI for the first entry
        first_entry = data_mgr.signal_data[0]
        first_meta = data_mgr.metadata.iloc[0]
        
        plt.figure(figsize=(12, 8))
        #  Use LaTeX for plot typography
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        
        # STEP 4.2: Phase plot
        plt.subplot(2, 2, 1)
        #  Use LaTeX for plot typography
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.plot(np.rad2deg(first_entry['phi1']), label='Antenna 1')
        plt.plot(np.rad2deg(first_entry['phi2']), label='Antenna 2')
        plt.title(f"Phase at D={first_meta['D']}m, W={first_meta['W']}m")
        plt.xlabel("Sample")
        plt.ylabel("Phase (degrees)")
        plt.legend()
        plt.grid(True)
        
        # STEP 4.3: RSSI plot
        plt.subplot(2, 2, 2)
        #  Use LaTeX for plot typography
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.plot(first_entry['rssi1'], label='Antenna 1')
        plt.plot(first_entry['rssi2'], label='Antenna 2')
        plt.title(f"RSSI at D={first_meta['D']}m, W={first_meta['W']}m")
        plt.xlabel("Sample")
        plt.ylabel("RSSI (dBm)")
        plt.legend()
        plt.grid(True)
        
        # STEP 4.4: Phasor plot (complex plane)
        plt.subplot(2, 2, 3)
        #  Use LaTeX for plot typography
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.scatter(first_entry['phasor1'].real, first_entry['phasor1'].imag, label='Antenna 1', alpha=0.7)
        plt.scatter(first_entry['phasor2'].real, first_entry['phasor2'].imag, label='Antenna 2', alpha=0.7)
        plt.title("Phasors in Complex Plane")
        plt.xlabel("Real")
        plt.ylabel("Imaginary")
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        
        # STEP 4.5: Phase difference histogram
        plt.subplot(2, 2, 4)
        #  Use LaTeX for plot typography
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        phase_diff = np.angle(first_entry['phasor1']) - np.angle(first_entry['phasor2'])
        phase_diff = np.rad2deg(np.angle(np.exp(1j * phase_diff)))  # Wrap to [-180, 180]
        plt.hist(phase_diff, bins=20)
        plt.title("Phase Difference Histogram")
        plt.xlabel("Phase Difference (degrees)")
        plt.ylabel("Count")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("data_verification.png")
        print(f"Visualization saved to {'data_verification.png'}")
        plt.show()
    
    # STEP 5: Test Position Distribution
    if len(data_mgr.metadata) > 0:
        plt.figure(figsize=(8, 6))
        #  Use LaTeX for plot typography
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.scatter(data_mgr.metadata['D'], data_mgr.metadata['W'])
        plt.title("Measurement Positions (D-W plane)")
        plt.xlabel("Distance (m)")
        plt.ylabel("Width (m)")
        plt.grid(True)
        plt.axis('equal')
        plt.savefig("measurement_positions.png")
        print(f"Position plot saved to {'measurement_positions.png'}")
        plt.show()
    
    return data_mgr
# =================================================================================================================================== #


# =================================================================================================================================== #
# ----------------------------------------------------- MACHINE LEARNING ANALYSIS --------------------------------------------------- #
# Set seeds for reproducibility
pyro.set_rng_seed(42)
torch.manual_seed(42)
np.random.seed(42)

class BayesianAoARegressor:
    """
    Bayesian Angle of Arrival (AoA) Regressor that incorporates physics-based priors
    from MUSIC and beamforming methods.
    
    This model uses Pyro for Bayesian inference and allows for different types of priors
    based on the physical understanding of the AoA estimation problem.
    """
    
    def __init__(self, use_gpu=True, prior_type='ds', feature_mode='full'):
        """
        Initialize the Bayesian AoA Regressor.
        
        Args:
            use_gpu (bool): Whether to use GPU acceleration if available
            prior_type (str): Type of prior to use ('ds', 'music', 'weighted', 'flat')
            feature_mode (str): Feature set to use ('full', 'width_only', 'sensor_only')
        """
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        self.prior_type = prior_type
        self.feature_mode = feature_mode
        self.model = None
        self.guide = None
        self.scalers = []
        self.feature_names = None
        self.train_summary = None
        
        print(f"Using device: {self.device}")
        print(f"Prior type: {self.prior_type}")
        print(f"Feature mode: {self.feature_mode}")
        if self.use_gpu:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    def _extract_features(self, data_manager, include_distance=True, include_width=True):
        """Extract features from data manager for Bayesian regression"""
        # Verify that results are available
        if data_manager.results is None:
            data_manager.analyze_all_data(save_results=False)

        # Prepare features
        all_features = []
        all_angles = []
        prior_estimates = {
            'ds': [],
            'weighted': [],
            'music': [],
            'phase': []
        }
        
        # Track feature names
        feature_names = []

        for idx, meta in enumerate(data_manager.metadata.iterrows()):
            meta = meta[1]  # Get the actual Series from the tuple

            # Extract signal data
            signals = data_manager.signal_data[idx]
            phasor1 = signals['phasor1']
            phasor2 = signals['phasor2']
            rssi1 = signals['rssi1']
            rssi2 = signals['rssi2']

            # Calculate true angle
            true_angle = data_manager.get_true_angle(meta['D'], meta['W'])

            # Store prior estimates from results dataframe
            result_row = data_manager.results.iloc[idx]
            prior_estimates['ds'].append(result_row['theta_ds'])
            prior_estimates['weighted'].append(result_row['theta_weighted'])
            prior_estimates['music'].append(result_row['theta_music'])
            prior_estimates['phase'].append(result_row['theta_phase'])

            # Feature Extraction
            features = []
            
            # Mean phase and magnitude values
            phase1_mean = np.angle(np.mean(phasor1))
            phase2_mean = np.angle(np.mean(phasor2))
            mag1_mean = np.mean(np.abs(phasor1))
            mag2_mean = np.mean(np.abs(phasor2))
            
            # Phase difference
            phase_diff = phase1_mean - phase2_mean
            phase_diff = np.angle(np.exp(1j * phase_diff))  # Wrap to [-π, π]
            
            # RSSI features
            rssi1_mean = np.mean(rssi1)
            rssi2_mean = np.mean(rssi2)
            rssi_diff = rssi1_mean - rssi2_mean
            
            # Phasor correlations (using real and imaginary parts)
            phasor1_real_mean = np.mean(phasor1.real)
            phasor1_imag_mean = np.mean(phasor1.imag)
            phasor2_real_mean = np.mean(phasor2.real)
            phasor2_imag_mean = np.mean(phasor2.imag)
            
            # Wavelength
            wavelength = meta['lambda']
            
            # Add basic phasor-derived features (always included)
            if len(feature_names) == 0:  # Only add names once
                feature_names.extend([
                    'phase1_mean', 'phase2_mean', 'phase_diff',
                    'mag1_mean', 'mag2_mean',
                    'rssi1_mean', 'rssi2_mean', 'rssi_diff',
                    'phasor1_real_mean', 'phasor1_imag_mean',
                    'phasor2_real_mean', 'phasor2_imag_mean',
                    'wavelength'
                ])
            
            features.extend([
                phase1_mean, phase2_mean, phase_diff,
                mag1_mean, mag2_mean,
                rssi1_mean, rssi2_mean, rssi_diff,
                phasor1_real_mean, phasor1_imag_mean,
                phasor2_real_mean, phasor2_imag_mean,
                wavelength
            ])
            
            # Add geometric features if specified
            if include_distance:
                distance = meta['D']
                features.append(distance)
                if len(feature_names) == 13:  # Add name only once
                    feature_names.append('distance')
            
            if include_width:
                width = meta['W']
                features.append(width)
                # Add name only if not already added
                if (include_distance and len(feature_names) == 14) or \
                (not include_distance and len(feature_names) == 13):
                    feature_names.append('width')

            all_features.append(features)
            all_angles.append(true_angle)

        # Convert to numpy arrays
        X = np.array(all_features)
        y = np.array(all_angles)
        
        # Convert prior estimates to numpy arrays
        for key in prior_estimates:
            prior_estimates[key] = np.array(prior_estimates[key])

        self.feature_names = feature_names
        return X, y, feature_names, prior_estimates
    
    def _build_model(self, input_dim, prior_mean, prior_std):
        """
        Build the Bayesian regression model using Pyro.
        
        Args:
            input_dim (int): Number of input features
            prior_mean (float): Mean for the weight prior
            prior_std (float): Standard deviation for the weight prior
        """
        class BayesianLinearRegression(PyroModule):
            def __init__(self, input_dim, prior_mean=0.0, prior_std=1.0, device='cpu'):
                super().__init__()
                self.device = device
                
                # Register the weights as a PyroSample
                self.linear = PyroModule[torch.nn.Linear](input_dim, 1).to(device)
                
                # Set informative priors for weights based on physics
                weight_prior_mean = torch.ones(1, input_dim, device=device) * prior_mean
                weight_prior_std = torch.ones(1, input_dim, device=device) * prior_std
                
                self.linear.weight = PyroSample(
                    dist.Normal(weight_prior_mean, weight_prior_std).to_event(2)
                )
                
                # Prior for bias with explicit device placement
                bias_prior = torch.zeros(1, device=device)
                bias_scale = torch.ones(1, device=device)
                self.linear.bias = PyroSample(
                    dist.Normal(bias_prior, bias_scale).to_event(1)
                )
                
            def forward(self, x, y=None):
                # Ensure x is on the correct device
                if x.device != self.device:
                    x = x.to(self.device)
                    
                # Get predicted angle from linear model
                mean = self.linear(x).squeeze(-1)
                
                # Observation noise (learnable) with explicit device placement
                sigma = pyro.sample("sigma", dist.LogNormal(
                    torch.tensor(0.0, device=self.device), 
                    torch.tensor(1.0, device=self.device)
                ))
                
                # Condition on observed data if provided
                with pyro.plate("data", x.shape[0]):
                    # Ensure y is on the correct device if provided
                    if y is not None and y.device != self.device:
                        y = y.to(self.device)
                    obs = pyro.sample("obs", dist.Normal(mean, sigma), obs=y)
                
                return mean
        
        # Create model and move to the correct device
        model = BayesianLinearRegression(input_dim, prior_mean, prior_std, device=self.device)
        
        # Double-check all parameters are on the correct device
        for name, param in model.named_parameters():
            if param.device != self.device:
                param.data = param.data.to(self.device)
                
        return model
    
    def train(self, data_manager, num_epochs=1000, learning_rate=0.01):
        """
        Train the Bayesian AoA regression model.
        
        Args:
            data_manager: The DataManager object containing the dataset
            num_epochs (int): Number of training epochs
            learning_rate (float): Learning rate for the optimizer
            
        Returns:
            dict: Training metrics and summary
        """
        # Determine feature inclusion based on feature_mode
        include_distance = self.feature_mode == 'full'
        include_width = self.feature_mode in ['full', 'width_only']
        
        # Extract features and data
        X, y, feature_names, prior_estimates = self._extract_features(
            data_manager, include_distance, include_width)
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=42)
        
        # Also split prior estimates for visualization
        prior_splits = {}
        for key in prior_estimates:
            _, prior_test = train_test_split(
                prior_estimates[key], test_size=0.1, random_state=42)
            prior_splits[key] = prior_test
        
        # Standardize features
        self.scalers = []
        X_train_scaled = np.zeros_like(X_train)
        X_test_scaled = np.zeros_like(X_test)
        
        for i in range(X_train.shape[1]):
            scaler = StandardScaler()
            X_train_scaled[:, i] = scaler.fit_transform(X_train[:, i].reshape(-1, 1)).flatten()
            X_test_scaled[:, i] = scaler.transform(X_test[:, i].reshape(-1, 1)).flatten()
            self.scalers.append(scaler)
        
        # Convert to tensors
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(self.device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(self.device)
        
        # Determine prior parameters based on prior_type
        prior_mean = 0.0
        prior_std = 1.0
        
        if self.prior_type in prior_estimates:
            # Calculate the standard deviation of the difference between
            # the physics-based estimate and the true angle
            prior_error = prior_estimates[self.prior_type] - y
            prior_std = max(0.1, np.std(prior_error))  # Ensure minimum variance
            
            print(f"Using {self.prior_type} prior with std={prior_std:.4f}")
        
        # Clear any existing parameters
        pyro.clear_param_store()
        
        # Create the model and guide
        input_dim = X_train.shape[1]
        self.model = self._build_model(input_dim, prior_mean, prior_std)

        # Create guide with explicit device placement
        self.guide = AutoNormal(self.model)

        # Explicitly move all guide parameters to the correct device
        if self.use_gpu:
            for name, value in self.guide.named_parameters():
                if value.device != self.device:
                    value.data = value.data.to(self.device)
        
        # Setup SVI (Stochastic Variational Inference)
        optimizer = optim.Adam({"lr": learning_rate})
        svi = SVI(self.model, self.guide, optimizer, loss=Trace_ELBO())
        
        # Training loop
        losses = []
        for epoch in range(num_epochs):
            loss = svi.step(X_train_tensor, y_train_tensor)
            losses.append(loss)
            
            if (epoch+1) % 100 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss:.4f}")
        
        # Evaluate on test set
        self.model.eval()
        
        # Perform predictions with uncertainty
        predictive = Predictive(self.model, guide=self.guide, num_samples=1000)
        samples = predictive(X_test_tensor)
        
        # Extract predictions and calculate mean and std
        y_pred_samples = samples["obs"].cpu().numpy()
        y_pred_mean = y_pred_samples.mean(axis=0)
        y_pred_std = y_pred_samples.std(axis=0)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred_mean)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_mean))
        
        # Also compute metrics for the prior methods on test set
        prior_metrics = {}
        for key in prior_splits:
            prior_mae = mean_absolute_error(y_test, prior_splits[key])
            prior_rmse = np.sqrt(mean_squared_error(y_test, prior_splits[key]))
            prior_metrics[key] = {
                'mae': prior_mae,
                'rmse': prior_rmse
            }
        
        # Store training summary
        self.train_summary = {
            'losses': losses,
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred_mean': y_pred_mean,
            'y_pred_std': y_pred_std,
            'y_pred_samples': y_pred_samples,
            'mae': mae,
            'rmse': rmse,
            'prior_metrics': prior_metrics,
            'prior_test': prior_splits,
            'feature_names': feature_names
        }
        
        print(f"\nTraining complete!")
        print(f"Test MAE: {mae:.4f}°, RMSE: {rmse:.4f}°")
        
        # Print comparison with prior methods
        print("\nComparison with physics-based methods:")
        for key, metrics in prior_metrics.items():
            print(f"  {key.upper()}: MAE={metrics['mae']:.4f}°, RMSE={metrics['rmse']:.4f}°")
        
        return self.train_summary
    
    def predict(self, X, return_uncertainty=True):
        """
        Make predictions with the trained model.
        
        Args:
            X (np.ndarray): Input features
            return_uncertainty (bool): Whether to return uncertainty estimates
            
        Returns:
            tuple or array: Predictions and optionally uncertainty estimates
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Standardize features
        X_scaled = np.zeros_like(X)
        for i in range(X.shape[1]):
            X_scaled[:, i] = self.scalers[i].transform(X[:, i].reshape(-1, 1)).flatten()
        
        # Convert to tensor
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        
        # Perform predictions with uncertainty
        self.model.eval()
        predictive = Predictive(self.model, guide=self.guide, num_samples=1000)
        samples = predictive(X_tensor)
        
        # Extract predictions
        y_pred_samples = samples["obs"].cpu().numpy()
        y_pred_mean = y_pred_samples.mean(axis=0)
        
        if return_uncertainty:
            y_pred_std = y_pred_samples.std(axis=0)
            return y_pred_mean, y_pred_std
        else:
            return y_pred_mean
    
    def visualize_results(self, output_dir, experiment_name):
        """
        Create comprehensive visualizations of model results.
        
        Args:
            output_dir (str): Directory to save visualizations
            experiment_name (str): Name for this experiment
        """
        if self.train_summary is None:
            raise ValueError("Model must be trained before visualizing results")
        
        # Create output directory
        vis_dir = os.path.join(output_dir, "bayesian_model", experiment_name)
        os.makedirs(vis_dir, exist_ok=True)
        
        # Extract data from training summary
        y_test = self.train_summary['y_test']
        y_pred_mean = self.train_summary['y_pred_mean']
        y_pred_std = self.train_summary['y_pred_std']
        y_pred_samples = self.train_summary['y_pred_samples']
        prior_test = self.train_summary['prior_test']
        losses = self.train_summary['losses']
        
        # Setup plots with LaTeX formatting
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        
        # Figure 1: Predicted vs True angles with uncertainty
        plt.figure(figsize=(10, 8))
        
        # Plot 1:1 line
        min_val = min(y_test.min(), y_pred_mean.min())
        max_val = max(y_test.max(), y_pred_mean.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        # Plot predictions with error bars
        plt.errorbar(y_test, y_pred_mean, yerr=2*y_pred_std, fmt='o', alpha=0.6, 
                     label='Bayesian Predictions (2$\sigma$ intervals)')
        
        # Plot prior method predictions
        prior_colors = {'ds': 'g', 'weighted': 'm', 'music': 'c', 'phase': 'y'}
        prior_markers = {'ds': '^', 'weighted': 's', 'music': 'd', 'phase': 'x'}
        prior_labels = {'ds': 'DS Beamforming', 'weighted': 'Weighted DS', 
                        'music': 'MUSIC', 'phase': 'Phase Difference'}
        
        for key in prior_test:
            if self.prior_type == key:
                # Highlight the method used as prior
                plt.scatter(y_test, prior_test[key], marker=prior_markers[key], 
                          color=prior_colors[key], alpha=0.5, s=80,
                          label=f"{prior_labels[key]} (Prior)")
            else:
                plt.scatter(y_test, prior_test[key], marker=prior_markers[key], 
                          color=prior_colors[key], alpha=0.3, s=40,
                          label=f"{prior_labels[key]}")
        
        plt.xlabel('True Angle (degrees)')
        plt.ylabel('Predicted Angle (degrees)')
        plt.title(f'Bayesian AoA Model with {self.prior_type.upper()} Prior\n'
                 f'MAE: {self.train_summary["mae"]:.2f}°, RMSE: {self.train_summary["rmse"]:.2f}°')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "predicted_vs_true.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure 2: Training loss curve
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('ELBO Loss')
        plt.title('Training Loss')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "training_loss.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure 3: Prediction error distribution
        plt.figure(figsize=(12, 8))
        
        # Create subplots for each method
        plt.subplot(2, 1, 1)
        errors = y_pred_mean - y_test
        plt.hist(errors, bins=20, alpha=0.7, label='Bayesian Model')
        plt.axvline(0, color='k', linestyle='--')
        plt.xlabel('Prediction Error (degrees)')
        plt.ylabel('Count')
        plt.title('Error Distribution Comparison')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(2, 1, 2)
        for key in prior_test:
            prior_errors = prior_test[key] - y_test
            plt.hist(prior_errors, bins=20, alpha=0.5, label=prior_labels[key])
        
        plt.axvline(0, color='k', linestyle='--')
        plt.xlabel('Prediction Error (degrees)')
        plt.ylabel('Count')
        plt.title('Physics-based Methods Error Distribution')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "error_distribution.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure 4: Parameter posterior distributions
        # Extract guide parameters
        posterior_means = {}
        posterior_stds = {}
        
        for name, param in self.guide.named_parameters():
            if 'AutoNormal.loc' in name and 'linear.weight' in name:
                posterior_means['weights'] = param.detach().cpu().numpy()
            elif 'AutoNormal.scale' in name and 'linear.weight' in name:
                posterior_stds['weights'] = param.detach().cpu().numpy()
            elif 'AutoNormal.loc' in name and 'linear.bias' in name:
                posterior_means['bias'] = param.detach().cpu().numpy()
            elif 'AutoNormal.scale' in name and 'linear.bias' in name:
                posterior_stds['bias'] = param.detach().cpu().numpy()
        
        # Plot weight distributions
        if 'weights' in posterior_means:
            # First, get most important features
            importance = np.abs(posterior_means['weights'][0])
            top_indices = np.argsort(importance)[-8:]  # Top 8 features
            
            plt.figure(figsize=(12, 10))
            
            for i, idx in enumerate(top_indices):
                if idx < len(self.feature_names):
                    feat_name = self.feature_names[idx]
                    mean = posterior_means['weights'][0, idx]
                    std = posterior_stds['weights'][0, idx]
                    
                    # Create range of values for x-axis
                    x = np.linspace(mean - 3*std, mean + 3*std, 1000)
                    
                    # Plot normal distribution
                    plt.subplot(4, 2, i+1)
                    plt.plot(x, norm.pdf(x, mean, std))
                    plt.axvline(0, color='r', linestyle='--', alpha=0.5)
                    plt.axvline(mean, color='g', linestyle='-')
                    plt.fill_between(x, 0, norm.pdf(x, mean, std), alpha=0.3)
                    
                    plt.title(f'{feat_name} (μ={mean:.4f}, σ={std:.4f})')
                    plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, "weight_posteriors.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Figure 5: Feature importance plot
        if 'weights' in posterior_means:
            plt.figure(figsize=(12, 8))
            
            # Calculate feature importance as absolute mean weight
            importance = np.abs(posterior_means['weights'][0])
            # Normalize to sum to 100%
            importance = 100 * importance / importance.sum()
            
            # Sort by importance
            indices = np.argsort(importance)
            sorted_importance = importance[indices]
            sorted_names = [self.feature_names[i] if i < len(self.feature_names) else f'Feature {i}' 
                           for i in indices]
            
            # Create horizontal bar chart
            plt.barh(range(len(sorted_names)), sorted_importance, align='center')
            plt.yticks(range(len(sorted_names)), sorted_names)
            plt.xlabel('Relative Importance (%)')
            plt.title('Feature Importance in Bayesian AoA Model')
            plt.grid(True, alpha=0.3)
            
            # Add percentage labels
            for i, v in enumerate(sorted_importance):
                plt.text(v + 0.5, i, f'{v:.1f}%', va='center')
            
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, "feature_importance.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Figure 6: Uncertainty analysis
        plt.figure(figsize=(10, 8))
        
        # Calculate sorted indices by uncertainty
        sorted_indices = np.argsort(y_pred_std)
        
        # Plot errors vs uncertainty
        abs_errors = np.abs(y_pred_mean - y_test)
        
        plt.scatter(y_pred_std, abs_errors, alpha=0.7)
        
        # Add trendline
        z = np.polyfit(y_pred_std, abs_errors, 1)
        p = np.poly1d(z)
        plt.plot(np.sort(y_pred_std), p(np.sort(y_pred_std)), "r--", 
                alpha=0.8, label=f"Trend: y={z[0]:.2f}x+{z[1]:.2f}")
        
        plt.xlabel('Prediction Uncertainty (std)')
        plt.ylabel('Absolute Error (degrees)')
        plt.title('Relationship Between Uncertainty and Error')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "uncertainty_analysis.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create summary text file
        with open(os.path.join(vis_dir, "model_summary.txt"), 'w') as f:
            f.write(f"Bayesian AoA Model Summary\n")
            f.write(f"=======================\n\n")
            
            f.write(f"Model Configuration:\n")
            f.write(f"  Prior Type: {self.prior_type}\n")
            f.write(f"  Feature Mode: {self.feature_mode}\n")
            f.write(f"  Features Used: {', '.join(self.feature_names)}\n\n")
            
            f.write(f"Performance Metrics:\n")
            f.write(f"  MAE: {self.train_summary['mae']:.4f} degrees\n")
            f.write(f"  RMSE: {self.train_summary['rmse']:.4f} degrees\n\n")
            
            f.write(f"Comparison with Physics-based Methods:\n")
            for key, metrics in self.train_summary['prior_metrics'].items():
                f.write(f"  {key.upper()}: MAE={metrics['mae']:.4f}°, RMSE={metrics['rmse']:.4f}°\n")
            
            f.write("\nModel improves over prior by: ")
            if self.prior_type in self.train_summary['prior_metrics']:
                prior_mae = self.train_summary['prior_metrics'][self.prior_type]['mae']
                improvement = prior_mae - self.train_summary['mae']
                percent = (improvement / prior_mae) * 100
                f.write(f"{improvement:.4f}° ({percent:.1f}%)\n")
            else:
                f.write("N/A (no matching prior)\n")
                
            if 'weights' in posterior_means:
                f.write("\nTop 5 Important Features:\n")
                importance = np.abs(posterior_means['weights'][0])
                top_indices = np.argsort(importance)[-5:]
                
                for i, idx in enumerate(reversed(top_indices)):
                    if idx < len(self.feature_names):
                        feat_name = self.feature_names[idx]
                        mean = posterior_means['weights'][0, idx]
                        std = posterior_stds['weights'][0, idx]
                        f.write(f"  {i+1}. {feat_name}: weight={mean:.4f}±{std:.4f}\n")
    
    def render_model_and_guide(self, output_dir, experiment_name):
        """Render the model structure and guide distributions"""
        if self.model is None or self.guide is None:
            raise ValueError("Model and guide must be trained before rendering")
            
        # Create output directory
        vis_dir = os.path.join(output_dir, "bayesian_model", experiment_name)
        os.makedirs(vis_dir, exist_ok=True)
        
        # 1. Render model structure as text
        model_summary = str(self.model)
        with open(os.path.join(vis_dir, "model_structure.txt"), 'w') as f:
            f.write(f"Model Structure:\n{model_summary}\n\n")
            
            # Add parameter shapes
            f.write("Parameter Shapes:\n")
            for name, param in self.model.named_parameters():
                f.write(f"  {name}: {param.shape}\n")
        
        # 2. Visualize guide distributions
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Extract weight parameters from guide
        weight_loc = None
        weight_scale = None
        bias_loc = None
        bias_scale = None
        
        for name, param in self.guide.named_parameters():
            if 'weight' in name and 'loc' in name:
                weight_loc = param.detach().cpu().numpy()
            elif 'weight' in name and 'scale' in name:
                weight_scale = param.detach().cpu().numpy()
            elif 'bias' in name and 'loc' in name:
                bias_loc = param.detach().cpu().numpy()
            elif 'bias' in name and 'scale' in name:
                bias_scale = param.detach().cpu().numpy()
        
        # Plot weight distributions
        if weight_loc is not None and weight_scale is not None:
            # Check for negative scale values
            if np.any(weight_scale[0] < 0):
                print("Warning: Negative weight scale parameters found. Taking absolute values for visualization.")
                
            # Create parameter index
            x = np.arange(weight_loc.shape[1])
            # Plot mean with error bars - using absolute value for scale
            axes[0].errorbar(x, weight_loc[0], yerr=2*np.abs(weight_scale[0]), fmt='o', capsize=5)
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(self.feature_names, rotation=90)
            axes[0].set_title(r'Weight Posterior Distributions (mean ± 2$\sigma$)')
            axes[0].grid(True, alpha=0.3)
            
            # Highlight most important features
            importance = np.abs(weight_loc[0])
            for i in range(len(x)):
                if importance[i] > np.percentile(importance, 75):
                    axes[0].annotate(self.feature_names[i], 
                                (x[i], weight_loc[0][i]),
                                xytext=(0, 10), 
                                textcoords='offset points',
                                ha='center')
        
        # Plot bias distribution
        if bias_loc is not None and bias_scale is not None:
            # Check for negative scale values
            if np.any(bias_scale < 0):
                print("Warning: Negative bias scale parameters found. Taking absolute values for visualization.")
                
            # Using absolute value for bias scale
            axes[1].errorbar([0], bias_loc, yerr=2*np.abs(bias_scale), fmt='o', capsize=5)
            axes[1].set_xticks([0])
            axes[1].set_xticklabels(['Bias'])
            axes[1].set_title(r'Bias Posterior Distribution (mean ± 2$\sigma$)')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(os.path.join(vis_dir, "guide_distributions.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # 3. Create graphical model representation
        plt.figure(figsize=(10, 8))
        plt.title("Bayesian Linear Regression Graphical Model")
        
        # Define node positions
        pos = {
            'weights': (0.3, 0.7),
            'bias': (0.3, 0.5),
            'sigma': (0.3, 0.3),
            'mean': (0.6, 0.5),
            'y': (0.9, 0.5)
        }
        
        # Create nodes
        node_labels = {
            'weights': 'Weights\nw ~ N(prior_mean, prior_std)',
            'bias': 'Bias\nb ~ N(0, 1)',
            'sigma': 'Noise\nsigma ~ LogNormal(0, 1)',
            'mean': 'Linear Function\n mu = X·w + b',
            'y': 'Observations\ny ~ N(mu, sigma)'
        }
        
        # Create edges
        edges = [
            ('weights', 'mean'),
            ('bias', 'mean'),
            ('mean', 'y'),
            ('sigma', 'y')
        ]
        
        # Create graph
        G = nx.DiGraph()
        G.add_nodes_from(pos.keys())
        G.add_edges_from(edges)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='lightblue', alpha=0.8)
        nx.draw_networkx_edges(G, pos, width=2, arrowsize=20)
        
        # Add labels with custom positioning
        label_pos = {k: (v[0], v[1]-0.02) for k, v in pos.items()}
        nx.draw_networkx_labels(G, label_pos, labels=node_labels, font_size=10, 
                            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "graphical_model.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Model and guide renderings saved to {vis_dir}")

    def visualize_prior_vs_posterior(self, output_dir, experiment_name, num_samples=5):
        """
        Visualize how the model transforms prior distributions into posterior distributions.
        
        Args:
            output_dir (str): Directory to save visualizations
            experiment_name (str): Name for this experiment
            num_samples (int): Number of test samples to visualize
        """
        if self.train_summary is None:
            raise ValueError("Model must be trained before visualizing")
            
        # Create output directory
        vis_dir = os.path.join(output_dir, "bayesian_model", experiment_name)
        os.makedirs(vis_dir, exist_ok=True)
        
        # Extract data from training summary
        y_test = self.train_summary['y_test']
        y_pred_mean = self.train_summary['y_pred_mean']
        y_pred_std = self.train_summary['y_pred_std']
        prior_test = self.train_summary['prior_test']
        
        # Get the prior standard deviation used during training
        # Handle the case when the prior type is 'flat' or not in prior_metrics
        if self.prior_type in self.train_summary['prior_metrics']:
            prior_error = self.train_summary['prior_metrics'][self.prior_type]['mae']
            prior_std = max(0.1, prior_error)
        else:
            # For 'flat' prior or any other prior not in prior_metrics, use a default value
            prior_std = 1.0  # Default value
            print(f"Using default prior std={prior_std} for {self.prior_type} prior")
        
        # Select sample indices - try to get a diverse set
        if len(y_test) <= num_samples:
            indices = np.arange(len(y_test))
        else:
            # Sort by true angle and select evenly spaced samples
            sorted_indices = np.argsort(y_test)
            step = len(sorted_indices) // num_samples
            indices = sorted_indices[::step][:num_samples]
        
        # Create figure
        fig, axes = plt.subplots(num_samples, 1, figsize=(10, 3*num_samples))
        if num_samples == 1:
            axes = [axes]
        
        # Plot each sample
        for i, idx in enumerate(indices):
            ax = axes[i]
            
            # Get values for this sample
            true_angle = y_test[idx]
            posterior_mean = y_pred_mean[idx]
            posterior_std = y_pred_std[idx]
            
            # Get prior value based on prior type
            if self.prior_type in prior_test:
                prior_mean = prior_test[self.prior_type][idx]
            else:
                # For 'flat' prior or any other prior not in prior_test
                prior_mean = 0.0  # Use 0 as default for flat prior
            
            # Create x-axis range for plotting distributions
            plot_range = 4 * max(prior_std, posterior_std)
            x = np.linspace(min(prior_mean, posterior_mean) - plot_range, 
                            max(prior_mean, posterior_mean) + plot_range, 1000)
            
            # Plot prior distribution
            prior_pdf = norm.pdf(x, prior_mean, prior_std)
            ax.plot(x, prior_pdf, 'r--', linewidth=2, label=f'{self.prior_type.upper()} Prior')
            ax.fill_between(x, 0, prior_pdf, color='red', alpha=0.2)
            
            # Plot posterior distribution
            posterior_pdf = norm.pdf(x, posterior_mean, posterior_std)
            ax.plot(x, posterior_pdf, 'b-', linewidth=2, label='Bayesian Posterior')
            ax.fill_between(x, 0, posterior_pdf, color='blue', alpha=0.2)
            
            # Plot true angle
            ax.axvline(true_angle, color='k', linestyle='-', linewidth=2, label='True Angle')
            
            # Add labels and legend
            ax.set_xlabel('Angle (degrees)')
            ax.set_ylabel('Probability Density')
            ax.set_title(f'Sample {i+1}: Prior vs Posterior (True Angle: {true_angle:.2f}°)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Annotate statistics
            prior_error = abs(prior_mean - true_angle)
            posterior_error = abs(posterior_mean - true_angle)
            
            # Avoid division by zero
            if prior_error > 0:
                improvement = prior_error - posterior_error
                improvement_pct = 100*improvement/prior_error
            else:
                improvement = posterior_error
                improvement_pct = 0
            
            stats_text = (f'Prior: $\\mu={prior_mean:.2f}^\\circ$, $\\sigma={prior_std:.2f}^\\circ$, Error$={prior_error:.2f}^\\circ$\n'
                        f'Posterior: $\\mu={posterior_mean:.2f}^\\circ$, $\\sigma={posterior_std:.2f}^\\circ$, Error$={posterior_error:.2f}^\\circ$\n'
                        f'Improvement: ${improvement:.2f}^\\circ$ ({improvement_pct:.1f}\\% reduction)')
            
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Save figure
        plt.tight_layout()
        fig.savefig(os.path.join(vis_dir, "prior_vs_posterior.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Prior vs posterior visualization saved to {vis_dir}")

    def plot_posterior_predictive(self, output_dir, experiment_name):
        """Plot posterior predictive distribution"""
        if self.train_summary is None:
            raise ValueError("Model must be trained before plotting posterior predictive")
            
        # Create output directory
        vis_dir = os.path.join(output_dir, "bayesian_model", experiment_name)
        os.makedirs(vis_dir, exist_ok=True)
        
        # Extract data
        X_test = self.train_summary['X_test']
        y_test = self.train_summary['y_test']
        y_pred_samples = self.train_summary['y_pred_samples']
        
        # Plot posterior predictive distribution
        plt.figure(figsize=(12, 8))
        
        # Sort test points by true angle for better visualization
        sort_idx = np.argsort(y_test)
        y_test_sorted = y_test[sort_idx]
        
        # Calculate percentiles of predictions
        y_pred_5 = np.percentile(y_pred_samples[:, sort_idx], 5, axis=0)
        y_pred_95 = np.percentile(y_pred_samples[:, sort_idx], 95, axis=0)
        y_pred_25 = np.percentile(y_pred_samples[:, sort_idx], 25, axis=0)
        y_pred_75 = np.percentile(y_pred_samples[:, sort_idx], 75, axis=0)
        y_pred_50 = np.percentile(y_pred_samples[:, sort_idx], 50, axis=0)
        
        # Plot the data
        plt.fill_between(range(len(y_test)), y_pred_5, y_pred_95, alpha=0.3, color='blue', 
                        label='90% Credible Interval')
        plt.fill_between(range(len(y_test)), y_pred_25, y_pred_75, alpha=0.5, color='blue', 
                        label='50% Credible Interval')
        plt.plot(range(len(y_test)), y_pred_50, 'b-', linewidth=2, label='Median Prediction')
        plt.plot(range(len(y_test)), y_test_sorted, 'ro', label='True Angles')
        
        plt.xlabel('Test Point Index (sorted by true angle)')
        plt.ylabel('Angle (degrees)')
        plt.title('Posterior Predictive Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "posterior_predictive.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_uncertainty_calibration(self, output_dir, experiment_name):
        """Create a calibration plot for uncertainty estimates"""
        if self.train_summary is None:
            raise ValueError("Model must be trained before plotting uncertainty calibration")
            
        # Create output directory
        vis_dir = os.path.join(output_dir, "bayesian_model", experiment_name)
        os.makedirs(vis_dir, exist_ok=True)
        
        # Extract data
        y_test = self.train_summary['y_test']
        y_pred_mean = self.train_summary['y_pred_mean']
        y_pred_std = self.train_summary['y_pred_std']
        
        # Calculate standardized errors
        z_scores = (y_test - y_pred_mean) / y_pred_std
        
        # Create calibration plot
        plt.figure(figsize=(10, 8))
        
        # Plot histogram of standardized errors
        plt.hist(z_scores, bins=20, density=True, alpha=0.6, label='Standardized Errors')
        
        # Plot standard normal PDF for comparison
        x = np.linspace(-4, 4, 1000)
        plt.plot(x, norm.pdf(x), 'r-', linewidth=2, label='Standard Normal')
        
        plt.xlabel('Standardized Error (z-score)')
        plt.ylabel('Density')
        plt.title('Uncertainty Calibration - Standard Normal Check')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add KS test result
        ks_stat, ks_pval = kstest(z_scores, 'norm')
        plt.text(0.05, 0.95, f'KS Test: stat={ks_stat:.3f}, p-value={ks_pval:.3f}',
                transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "uncertainty_calibration.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def visualize_weight_distributions(self, output_dir, experiment_name):
        """
        Visualize the prior and posterior distributions of model weights,
        focusing on parameters related to beamforming.
        
        Args:
            output_dir (str): Directory to save visualizations
            experiment_name (str): Name for this experiment
        """
        if self.model is None or self.guide is None:
            raise ValueError("Model must be trained before visualizing weights")
            
        # Create output directory
        vis_dir = os.path.join(output_dir, "bayesian_model", experiment_name)
        os.makedirs(vis_dir, exist_ok=True)
        
        # Extract prior parameters (used during model initialization)
        # The prior standard deviation used for weights in _build_model
        if self.prior_type in self.train_summary['prior_metrics']:
            prior_error = self.train_summary['prior_metrics'][self.prior_type]['mae']
            prior_std = max(0.1, prior_error)
        else:
            prior_std = 1.0  # Default value for 'flat' prior
        prior_mean = 0.0  # We typically use zero-centered priors
        
        # Extract posterior parameters from guide
        weight_loc = None
        weight_scale = None
        
        for name, param in self.guide.named_parameters():
            if 'weight' in name and 'loc' in name:
                weight_loc = param.detach().cpu().numpy()
            elif 'weight' in name and 'scale' in name:
                weight_scale = param.detach().cpu().numpy()
        
        if weight_loc is None or weight_scale is None:
            print("Could not extract posterior weight parameters")
            return
        
        # Identify beamforming-related features (phase differences, RSSI)
        beamforming_indices = []
        phase_indices = []
        rssi_indices = []
        
        for i, name in enumerate(self.feature_names):
            if 'phase' in name.lower():
                phase_indices.append(i)
                beamforming_indices.append(i)
            elif 'rssi' in name.lower():
                rssi_indices.append(i)
                beamforming_indices.append(i)
        
        # Create visualization for beamforming-related weights
        plt.figure(figsize=(14, 10))
        
        # Determine number of subplots needed
        n_plots = len(beamforming_indices)
        if n_plots == 0:
            print("No beamforming-related features found")
            return
        
        # Determine grid layout
        rows = int(np.ceil(np.sqrt(n_plots)))
        cols = int(np.ceil(n_plots / rows))
        
        for i, idx in enumerate(beamforming_indices):
            # Create subplot
            plt.subplot(rows, cols, i+1)
            
            # Get feature name and parameters
            feature_name = self.feature_names[idx]
            post_mean = weight_loc[0, idx]
            post_std = weight_scale[0, idx]
            
            # Create x-axis range that accounts for both distributions
            # Create separate ranges for each distribution
            prior_range = np.linspace(prior_mean - 4*prior_std, prior_mean + 4*prior_std, 1000)
            post_range = np.linspace(post_mean - 4*post_std, post_mean + 4*post_std, 1000)
            
            # Plot distributions on their own appropriate scales
            ax1 = plt.gca()
            ax2 = ax1.twinx()  # Create second y-axis
            
            # Plot prior on first axis
            prior_pdf = norm.pdf(prior_range, prior_mean, prior_std)
            ax1.plot(prior_range, prior_pdf, 'r-', linewidth=2, label='Prior')
            ax1.fill_between(prior_range, 0, prior_pdf, color='red', alpha=0.2)
            ax1.set_ylabel('Prior Density', color='r')
            ax1.tick_params(axis='y', labelcolor='r')
            
            # Plot posterior on second axis with different scale
            post_pdf = norm.pdf(post_range, post_mean, post_std)
            ax2.plot(post_range, post_pdf, 'b-', linewidth=2, label='Posterior')
            ax2.fill_between(post_range, 0, post_pdf, color='blue', alpha=0.2)
            ax2.set_ylabel('Posterior Density', color='b')
            ax2.tick_params(axis='y', labelcolor='b')
            
            # Add vertical lines for means
            plt.axvline(prior_mean, color='r', linestyle='--', alpha=0.7)
            plt.axvline(post_mean, color='b', linestyle='--', alpha=0.7)
            
            # Add annotations
            plt.title(f'{feature_name}')
            plt.figtext(0.02, 0.98, f'Prior: $\\mu$={prior_mean:.3f}, $\\sigma$={prior_std:.3f}\n'
                                f'Post: $\\mu$={post_mean:.3f}, $\\sigma$={post_std:.3f}',
                    horizontalalignment='left', verticalalignment='top')
            
            # Enhanced annotation for clearer comparison
            """
            plt.annotate(f'Prior: $\\mu$={prior_mean:.2f}, $\\sigma$={prior_std:.2f}\n'
                        f'Posterior: $\\mu$={post_mean:.2f}, $\\sigma$={post_std:.2f}\n'
                        f'Diff: {post_mean-prior_mean:.2f}',
                        xy=(0.05, 0.95), xycoords='axes fraction',
                        ha='left', va='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            """
        
        # Add overall title
        plt.suptitle(f'Prior vs Posterior Weight Distributions for Beamforming-Related Features\n'
                    f'Model: {self.prior_type.upper()} Prior with {self.feature_mode} features',
                    fontsize=16)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
        plt.savefig(os.path.join(vis_dir, "beamforming_weight_distributions.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create separate plots for phase-related and RSSI-related features
        if phase_indices:
            self._plot_feature_group(phase_indices, "Phase-Related", prior_mean, prior_std, 
                                    weight_loc, weight_scale, vis_dir)
        
        if rssi_indices:
            self._plot_feature_group(rssi_indices, "RSSI-Related", prior_mean, prior_std, 
                                    weight_loc, weight_scale, vis_dir)
        
        print(f"Weight distribution visualizations saved to {vis_dir}")

    def _plot_feature_group(self, indices, group_name, prior_mean, prior_std, weight_loc, weight_scale, vis_dir):
        """Helper method to plot a group of related feature weight distributions"""
        plt.figure(figsize=(12, 8))
        
        rows = int(np.ceil(np.sqrt(len(indices))))
        cols = int(np.ceil(len(indices) / rows))
        
        for i, idx in enumerate(indices):
            plt.subplot(rows, cols, i+1)
            
            feature_name = self.feature_names[idx]
            post_mean = weight_loc[0, idx]
            post_std = weight_scale[0, idx]
            
            # Create x-axis range for plotting distributions
            plot_range = 5 * max(prior_std, post_std)
            x = np.linspace(min(prior_mean, post_mean) - plot_range, 
                            max(prior_mean, post_mean) + plot_range, 1000)
            
            # Plot prior distribution
            prior_pdf = norm.pdf(x, prior_mean, prior_std)
            plt.plot(x, prior_pdf, 'r--', linewidth=1.5, label='Prior')
            plt.fill_between(x, 0, prior_pdf, color='red', alpha=0.15)
            
            # Plot posterior distribution
            posterior_pdf = norm.pdf(x, post_mean, post_std)
            plt.plot(x, posterior_pdf, 'b-', linewidth=2.0, label='Posterior', zorder=10)
            plt.fill_between(x, 0, posterior_pdf, color='blue', alpha=0.25, zorder=5)
            
            plt.axvline(prior_mean, color='r', linestyle='-', alpha=0.5, label = '_Prior Mean')
            plt.axvline(post_mean, color='b', linestyle='-', alpha=0.7, label = '_Posterior Mean')
            plt.axvline(0, color='k', linestyle=':', alpha=0.5, label = '_Zero')
            
            plt.xlabel('Weight Value')
            plt.ylabel('Probability Density')
            plt.title(f'{feature_name}')
            if i == 0:
                plt.legend()
        
        plt.suptitle(f'{group_name} Feature Weight Distributions\n'
                    f'Model: {self.prior_type.upper()} Prior with {self.feature_mode} features',
                    fontsize=14)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(vis_dir, f"{group_name.lower().replace('-','_')}_weights.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def analyze_with_posterior_weights(self, data_manager, output_dir, experiment_name):
        """
        Re-analyze antenna array data using the posterior weights from the Bayesian model.
        
        Args:
            data_manager: DataManager object with the dataset
            output_dir: Directory to save results
            experiment_name: Name for this experiment
        """
        if self.model is None:
            raise ValueError("Model must be trained before analysis")
        
        # Create output directory
        analysis_dir = os.path.join(output_dir, "bayesian_model", experiment_name, "posterior_weighted_analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Extract posterior weights
        weight_loc = None
        for name, param in self.guide.named_parameters():
            if 'weight' in name and 'loc' in name:
                weight_loc = param.detach().cpu().numpy()
        
        if weight_loc is None:
            raise ValueError("Could not extract posterior weights")
        
        # Select a few test examples
        if data_manager.results is None:
            data_manager.analyze_all_data(save_results=False)
        
        # Get features for a few test cases
        test_indices = np.linspace(0, len(data_manager.metadata)-1, 5, dtype=int)
        
        for idx in test_indices:
            # Get original analysis results
            meta = data_manager.metadata.iloc[idx]
            signals = data_manager.signal_data[idx]
            
            # Extract parameters
            D = meta['D']
            W = meta['W']
            L = meta['L']
            wavelength = meta['lambda']
            true_angle = data_manager.get_true_angle(D, W)
            
            # Extract signals
            phasor1 = signals['phasor1']
            phasor2 = signals['phasor2']
            rssi1 = signals['rssi1']
            rssi2 = signals['rssi2']
            
            # Run original analysis with FULL range
            analysis_aoa = np.arange(MIN_ANGLE, MAX_ANGLE + STEP, STEP)
            original_results = analyze_aoa(
                phasor1, phasor2, rssi1, rssi2, 
                L, wavelength, analysis_aoa, true_angle
            )
            
            # Create feature vector for this example using the same feature extraction logic
            features = self._extract_features_for_sample(
                phasor1, phasor2, rssi1, rssi2, D, W, wavelength
            )
            
            # Scale features
            scaled_features = np.zeros_like(features)
            for i in range(features.shape[0]):
                scaled_features[i] = self.scalers[i].transform([[features[i]]])[0][0]
            
            # Get model prediction
            X_tensor = torch.tensor(scaled_features, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                prediction = self.model(X_tensor).item()
            
            # Use RESTRICTED range for Bayesian analysis
            bayesian_aoa = np.arange(BAYESIAN_MIN_ANGLE, BAYESIAN_MAX_ANGLE + STEP, STEP)
            batch_size = len(bayesian_aoa)
            feature_batch = np.tile(scaled_features, (batch_size, 1))
            
            # Modify angle-specific features for each item in the batch
            for i, angle in enumerate(bayesian_aoa):
                # Only modify features that relate to angle
                feature_batch[i, 0] = np.sin(np.deg2rad(angle))  # Assuming first feature is sin(phi)
                feature_batch[i, 1] = np.cos(np.deg2rad(angle))  # Assuming second feature is cos(phi)
            
            # Convert to tensor and get predictions in one batch
            batch_tensor = torch.tensor(feature_batch, dtype=torch.float32).to(self.device)
            
            # Process in smaller chunks to avoid memory issues
            chunk_size = 100
            weighted_spectrum = np.zeros(batch_size)
            
            print("Processing angles in batches...")
            for start_idx in range(0, batch_size, chunk_size):
                end_idx = min(start_idx + chunk_size, batch_size)
                chunk = batch_tensor[start_idx:end_idx]
                
                with torch.no_grad():
                    # Use a simpler forward pass that doesn't sample
                    chunk_output = self.model.linear(chunk).squeeze(-1)
                    weighted_spectrum[start_idx:end_idx] = chunk_output.cpu().numpy()
            
            # Normalize weighted spectrum
            weighted_spectrum = np.exp(-(weighted_spectrum - prediction)**2)
            weighted_spectrum = weighted_spectrum / weighted_spectrum.max()
            
            # Find peak of weighted spectrum
            weighted_angle = bayesian_aoa[np.argmax(weighted_spectrum)]
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            
            # Plot 1: Spectrum comparison
            plt.subplot(2, 1, 1)
            plt.plot(analysis_aoa, original_results['spectra']['ds'], 'r-', label='Original DS')
            plt.plot(analysis_aoa, original_results['spectra']['weighted'], 'm--', label='Original Weighted')
            
            # Create a padded version of the Bayesian spectrum to match the full range
            full_weighted_spectrum = np.zeros(len(analysis_aoa))
            # Find the indices that match the Bayesian range
            idx_start = np.searchsorted(analysis_aoa, BAYESIAN_MIN_ANGLE)
            idx_end = np.searchsorted(analysis_aoa, BAYESIAN_MAX_ANGLE)
            # Only fill in the values within the Bayesian range
            step_ratio = len(bayesian_aoa) / (idx_end - idx_start)
            for i in range(idx_start, idx_end):
                bayesian_idx = int((i - idx_start) * step_ratio)
                if bayesian_idx < len(weighted_spectrum):
                    full_weighted_spectrum[i] = weighted_spectrum[bayesian_idx]
            
            # Plot the limited Bayesian spectrum
            plt.plot(analysis_aoa, full_weighted_spectrum, 'g-', linewidth=2, label='Bayesian Weighted (±15°)')
            
            # Add shaded region to show Bayesian analysis range
            plt.axvspan(BAYESIAN_MIN_ANGLE, BAYESIAN_MAX_ANGLE, color='lightgreen', alpha=0.2, label='Bayesian Range')
            
            # Add vertical lines for angle estimates
            plt.axvline(original_results['angles']['ds'], color='r', linestyle=':', label='DS Est.')
            plt.axvline(original_results['angles']['weighted'], color='m', linestyle=':', label='Weighted Est.')
            plt.axvline(weighted_angle, color='g', linestyle=':', label='Bayesian Est.')
            plt.axvline(true_angle, color='k', linestyle='-', label='True Angle')
            
            plt.xlabel('Angle (degrees)')
            plt.ylabel('Normalized Power')
            plt.title(f'Spectrum Comparison (D={D:.2f}m, W={W:.2f}m)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Plot 2: Error comparison
            plt.subplot(2, 1, 2)
            methods = ['Phase', 'DS', 'Weighted', 'MUSIC', 'Bayesian']
            errors = [
                abs(original_results['angles']['phase'] - true_angle),
                abs(original_results['angles']['ds'] - true_angle),
                abs(original_results['angles']['weighted'] - true_angle),
                abs(original_results['angles']['music'] - true_angle),
                abs(weighted_angle - true_angle)
            ]
            
            plt.bar(methods, errors)
            plt.ylabel('Absolute Error (degrees)')
            plt.title('Error Comparison')
            plt.grid(True, alpha=0.3)
            
            # Add error values as text
            for i, v in enumerate(errors):
                plt.text(i, v + 0.1, f'{v:.2f}°', ha='center')
            
            plt.tight_layout()
            plt.savefig(os.path.join(analysis_dir, f'posterior_weighted_analysis_D{D:.2f}_W{W:.2f}.png'), 
                    dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Completed analysis for sample at D={D:.2f}m, W={W:.2f}m")
        
        print(f"Posterior weighted analysis completed and saved to {analysis_dir}")

        # Inside BayesianAoARegressor class:
    def _extract_features_for_sample(self, phasor1, phasor2, rssi1, rssi2, D, W, wavelength):
        """Extract features for a single sample"""
        # Mean phase and magnitude values
        phase1_mean = np.angle(np.mean(phasor1))
        phase2_mean = np.angle(np.mean(phasor2))
        mag1_mean = np.mean(np.abs(phasor1))
        mag2_mean = np.mean(np.abs(phasor2))
        
        # Phase difference
        phase_diff = phase1_mean - phase2_mean
        phase_diff = np.angle(np.exp(1j * phase_diff))  # Wrap to [-π, π]
        
        # RSSI features
        rssi1_mean = np.mean(rssi1)
        rssi2_mean = np.mean(rssi2)
        rssi_diff = rssi1_mean - rssi2_mean
        
        # Phasor correlations
        phasor1_real_mean = np.mean(phasor1.real)
        phasor1_imag_mean = np.mean(phasor1.imag)
        phasor2_real_mean = np.mean(phasor2.real)
        phasor2_imag_mean = np.mean(phasor2.imag)
        
        # Create feature vector
        features = np.array([
            phase1_mean, phase2_mean, phase_diff,
            mag1_mean, mag2_mean,
            rssi1_mean, rssi2_mean, rssi_diff,
            phasor1_real_mean, phasor1_imag_mean,
            phasor2_real_mean, phasor2_imag_mean,
            wavelength
        ])
        
        # Add geometric features based on feature_mode
        if self.feature_mode in ['full', 'width_only']:
            features = np.append(features, W)
        
        if self.feature_mode == 'full':
            features = np.append(features, D)
        
        return features

def train_bayesian_models(data_manager, results_dir, num_epochs=10000):
    """
    Train multiple Bayesian AoA regression models with different priors and feature sets.
    
    Args:
        data_manager: DataManager object containing the dataset
        results_dir: Directory to save results
        num_epochs: Number of training epochs
        
    Returns:
        dict: Dictionary containing trained models and results
    """
    print("\n=== TRAINING BAYESIAN AOA REGRESSION MODELS ===")
    
    # Define configurations to test - correctly including all feature modes
    configs = [
        # Full features (includes both distance and width)
        {"prior": "ds", "features": "full", "name": "ds_full"},
        {"prior": "music", "features": "full", "name": "music_full"},
        {"prior": "weighted", "features": "full", "name": "weighted_full"},
        {"prior": "flat", "features": "full", "name": "flat_full"},
        
        # Width only (includes width but not distance)
        {"prior": "ds", "features": "width_only", "name": "ds_width"},
        {"prior": "music", "features": "width_only", "name": "music_width"},
        {"prior": "weighted", "features": "width_only", "name": "weighted_width"},
        {"prior": "flat", "features": "width_only", "name": "flat_width"},
        
        # Sensor only (no distance or width)
        {"prior": "ds", "features": "sensor_only", "name": "ds_sensor"},
        {"prior": "music", "features": "sensor_only", "name": "music_sensor"},
        {"prior": "weighted", "features": "sensor_only", "name": "weighted_sensor"},
        {"prior": "flat", "features": "sensor_only", "name": "flat_sensor"}
    ]
    
    # Dictionary to store results
    models = {}
    results = {}
    
    # Train models for each configuration
    for config in configs:
        print(f"\n--- Training Bayesian model with {config['prior']} prior, {config['features']} features ---")
        
        # Create model
        model = BayesianAoARegressor(
            use_gpu=True,
            prior_type=config['prior'],
            feature_mode=config['features']
        )
        
        # Train model
        train_results = model.train(data_manager, num_epochs=num_epochs)

        # Store model and results
        models[config['name']] = model
        results[config['name']] = train_results
        
        # Visualize results
        model.visualize_results(results_dir, config['name'])

        # Generate new visualizations
        model.render_model_and_guide(results_dir, config['name'])
        model.plot_posterior_predictive(results_dir, config['name'])
        model.plot_uncertainty_calibration(results_dir, config['name'])

        # Visualize prior vs posterior
        models[config['name']].visualize_prior_vs_posterior(results_dir, config['name'])
        model.visualize_weight_distributions(results_dir, config['name'])
        model.analyze_with_posterior_weights(data_manager, results_dir, config['name'])
        
        print(f"Completed training {config['name']}")
    
    # Create comparison visualizations
    compare_bayesian_models(models, results, results_dir)
    
    return {"models": models, "results": results}

def compare_bayesian_models(models, results, output_dir):
    """
    Create comparative visualizations for multiple Bayesian models.
    
    Args:
        models (dict): Dictionary of trained BayesianAoARegressor models
        results (dict): Dictionary of training results
        output_dir (str): Directory to save visualizations
    """
    # Create output directory
    comp_dir = os.path.join(output_dir, "bayesian_model_comparison")
    os.makedirs(comp_dir, exist_ok=True)
    
    # Setup plots with LaTeX formatting
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    
    # Extract performance metrics
    model_names = []
    maes = []
    rmses = []
    prior_maes = {}
    
    # Group by prior type and feature mode
    prior_types = set()
    feature_modes = set()
    grouped_results = {}
    
    for name, res in results.items():
        model = models[name]
        
        # Track model info
        model_names.append(name)
        maes.append(res['mae'])
        rmses.append(res['rmse'])
        
        # Extract prior metrics
        if model.prior_type in res['prior_metrics']:
            if model.prior_type not in prior_maes:
                prior_maes[model.prior_type] = []
            prior_maes[model.prior_type].append(res['prior_metrics'][model.prior_type]['mae'])
        
        # Group by prior and features
        prior_types.add(model.prior_type)
        feature_modes.add(model.feature_mode)
        
        key = (model.prior_type, model.feature_mode)
        grouped_results[key] = {
            'name': name,
            'mae': res['mae'],
            'rmse': res['rmse'],
            'y_test': res['y_test'],
            'y_pred': res['y_pred_mean'],
            'y_std': res['y_pred_std']
        }
    
    # Figure 1: Overall Performance Comparison
    plt.figure(figsize=(12, 8))
    
    # Sort models by MAE
    sort_idx = np.argsort(maes)
    sorted_names = [model_names[i] for i in sort_idx]
    sorted_maes = [maes[i] for i in sort_idx]
    sorted_rmses = [rmses[i] for i in sort_idx]
    
    # Create readable labels
    display_names = []
    for name in sorted_names:
        parts = name.split('_')
        prior = parts[0].upper()
        if parts[1] == 'full':
            features = 'Full'
        elif parts[1] == 'width':
            features = 'Width Only'
        else:
            features = 'Sensor Only'
        display_names.append(f"{prior} + {features}")
    
    # Create bar chart
    x = np.arange(len(display_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 8))
    rects1 = ax.bar(x - width/2, sorted_maes, width, label='MAE')
    rects2 = ax.bar(x + width/2, sorted_rmses, width, label='RMSE')
    
    # Add labels and title
    ax.set_ylabel('Error (degrees)')
    ax.set_title('Bayesian AoA Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=45, ha='right')
    ax.legend()
    
    # Add value labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}°',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    plt.savefig(os.path.join(comp_dir, "overall_performance.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Prior vs Bayesian Improvement
    if prior_maes:
        plt.figure(figsize=(10, 6))
        
        # For each prior type, show improvement
        x_labels = []
        improvements = []
        
        for prior_type in prior_maes:
            for i, (name, res) in enumerate(results.items()):
                model = models[name]
                if model.prior_type == prior_type:
                    # Calculate improvement
                    prior_mae = res['prior_metrics'][prior_type]['mae']
                    model_mae = res['mae']
                    improvement = prior_mae - model_mae
                    percent = (improvement / prior_mae) * 100 if prior_mae > 0 else 0
                    
                    # Create label
                    if model.feature_mode == 'full':
                        label = f"{prior_type.upper()} + Full"
                    elif model.feature_mode == 'width_only':
                        label = f"{prior_type.upper()} + Width"
                    else:
                        label = f"{prior_type.upper()} + Sensor"
                    
                    x_labels.append(label)
                    improvements.append(percent)
        
        # Sort by improvement
        sort_idx = np.argsort(improvements)
        sorted_labels = [x_labels[i] for i in sort_idx]
        sorted_improvements = [improvements[i] for i in sort_idx]
        
        # Plot improvements
        plt.barh(range(len(sorted_labels)), sorted_improvements, align='center')
        plt.yticks(range(len(sorted_labels)), sorted_labels)
        plt.xlabel('Improvement Over Prior (%)')
        plt.title('Bayesian Model Improvement Over Physics-Based Priors')
        plt.grid(True, alpha=0.3)
        
        # Add percentage labels
        for i, v in enumerate(sorted_improvements):
            plt.text(v + 0.5, i, f'{v:.1f}%', va='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(comp_dir, "prior_improvement.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Figure 3: Scatter plot matrix for feature mode comparison
    for prior in prior_types:
        # Skip flat prior
        if prior == 'flat':
            continue
            
        # Find all models with this prior but different feature modes
        models_with_prior = [(key, data) for key, data in grouped_results.items() 
                            if key[0] == prior]
        
        if len(models_with_prior) > 1:
            fig, axes = plt.subplots(1, len(models_with_prior), figsize=(15, 5))
            if len(models_with_prior) == 1:
                axes = [axes]
                
            fig.suptitle(f'{prior.upper()} Prior with Different Feature Sets', fontsize=16)
            
            for i, ((_, feat_mode), data) in enumerate(models_with_prior):
                # Create scatter plot
                ax = axes[i]
                ax.scatter(data['y_test'], data['y_pred'], alpha=0.7)
                
                # Add 1:1 line
                min_val = min(data['y_test'].min(), data['y_pred'].min())
                max_val = max(data['y_test'].max(), data['y_pred'].max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--')
                
                # Add labels
                if feat_mode == 'full':
                    title = 'Full Features'
                elif feat_mode == 'width_only':
                    title = 'Width Only'
                else:
                    title = 'Sensor Only'
                    
                ax.set_title(f'{title} (MAE: {data["mae"]:.2f}°)')
                ax.set_xlabel('True Angle (degrees)')
                ax.set_ylabel('Predicted Angle (degrees)')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(comp_dir, f"{prior}_feature_comparison.png"), 
                      dpi=300, bbox_inches='tight')
            plt.close()
    
    # Figure 4: Feature mode comparison across priors
    for feat_mode in feature_modes:
        # Find all models with this feature mode but different priors
        models_with_feat = [(key, data) for key, data in grouped_results.items() 
                           if key[1] == feat_mode]
        
        if len(models_with_feat) > 1:
            fig, axes = plt.subplots(1, len(models_with_feat), figsize=(15, 5))
            if len(models_with_feat) == 1:
                axes = [axes]
                
            feat_title = "Full Features" if feat_mode == "full" else \
                         "Width Only" if feat_mode == "width_only" else "Sensor Only"
            
            fig.suptitle(f'{feat_title} with Different Priors', fontsize=16)
            
            for i, ((prior, _), data) in enumerate(models_with_feat):
                # Create scatter plot
                ax = axes[i]
                ax.scatter(data['y_test'], data['y_pred'], alpha=0.7)
                
                # Add 1:1 line
                min_val = min(data['y_test'].min(), data['y_pred'].min())
                max_val = max(data['y_test'].max(), data['y_pred'].max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--')
                
                # Add labels
                title = f'{prior.upper()} Prior'
                    
                ax.set_title(f'{title} (MAE: {data["mae"]:.2f}°)')
                ax.set_xlabel('True Angle (degrees)')
                ax.set_ylabel('Predicted Angle (degrees)')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(comp_dir, f"{feat_mode}_prior_comparison.png"), 
                      dpi=300, bbox_inches='tight')
            plt.close()
    
    # Create summary file
    with open(os.path.join(comp_dir, "comparison_summary.txt"), 'w') as f:
        f.write("Bayesian AoA Model Comparison\n")
        f.write("===========================\n\n")
        
        f.write("Performance Summary:\n")
        for i, name in enumerate(sorted_names):
            model = models[name]
            mae = sorted_maes[i]
            rmse = sorted_rmses[i]
            
            # Format name for readability
            parts = name.split('_')
            prior = parts[0].upper()
            if parts[1] == 'full':
                features = 'Full Features'
            elif parts[1] == 'width':
                features = 'Width Only'
            else:
                features = 'Sensor Only'
                
            f.write(f"{i+1}. {prior} Prior with {features}\n")
            f.write(f"   MAE: {mae:.4f}°, RMSE: {rmse:.4f}°\n")
            
            # Add improvement over prior if applicable
            if model.prior_type in results[name]['prior_metrics']:
                prior_mae = results[name]['prior_metrics'][model.prior_type]['mae']
                improvement = prior_mae - mae
                percent = (improvement / prior_mae) * 100 if prior_mae > 0 else 0
                f.write(f"   Improvement over {model.prior_type.upper()} prior: {improvement:.4f}° ({percent:.1f}%)\n")
            
            f.write("\n")
        
        # Find best overall model
        best_idx = np.argmin(maes)
        best_name = model_names[best_idx]
        best_model = models[best_name]
        
        f.write("\nBest Overall Model:\n")
        f.write(f"  {best_model.prior_type.upper()} Prior with {best_model.feature_mode} features\n")
        f.write(f"  MAE: {maes[best_idx]:.4f}°, RMSE: {rmses[best_idx]:.4f}°\n\n")
        
        # Best model by feature mode
        f.write("Best Model by Feature Set:\n")
        for feat_mode in feature_modes:
            feat_models = [(name, res['mae']) for name, res in results.items() 
                          if models[name].feature_mode == feat_mode]
            if feat_models:
                best_feat_name, best_feat_mae = min(feat_models, key=lambda x: x[1])
                best_feat_model = models[best_feat_name]
                
                if feat_mode == 'full':
                    feat_desc = 'Full Features'
                elif feat_mode == 'width_only':
                    feat_desc = 'Width Only'
                else:
                    feat_desc = 'Sensor Only'
                    
                f.write(f"  {feat_desc}: {best_feat_model.prior_type.upper()} Prior (MAE: {best_feat_mae:.4f}°)\n")
        
        f.write("\n")
        
        # Best model by prior type
        f.write("Best Model by Prior Type:\n")
        for prior in prior_types:
            prior_models = [(name, res['mae']) for name, res in results.items() 
                           if models[name].prior_type == prior]
            if prior_models:
                best_prior_name, best_prior_mae = min(prior_models, key=lambda x: x[1])
                best_prior_model = models[best_prior_name]
                
                f.write(f"  {prior.upper()} Prior: {best_prior_model.feature_mode} features (MAE: {best_prior_mae:.4f}°)\n")
# =================================================================================================================================== #


# =================================================================================================================================== #
# ------------------------------------------------------- DASHBOARD ANALYSIS ------------------------------------------------------- #
def create_dashboard():
    """
    Create a comprehensive AoA analysis dashboard with per-distance analysis
    
    This function creates a series of visualizations including:
    - Per-distance analysis with AoA vs width plots
    - 3D beam pattern visualizations
    - Heatmaps of beamforming patterns
    - Error analysis and method comparisons
    
    Returns:
        pd.DataFrame: Results dataframe with all AoA estimates
    """
    print("Starting RFID AoA Analysis Dashboard Creation...")
    
    # Step 1: Create DataManager and import data
    rfid_data = DataManager(data_dir=DATA_DIRECTORY, tag_id=TAG_ID, aoa_range=AoA_m)
    rfid_data.import_data()
    
    # Step 2: Create output directories
    dashboard_dir = os.path.join(RESULTS_DIRECTORY, "dashboard")
    os.makedirs(dashboard_dir, exist_ok=True)
    
    # Step 3: Extract unique distances and frequencies
    distances = sorted(rfid_data.distances)
    frequencies = sorted(rfid_data.frequencies)
    
    print(f"Found {len(distances)} distances, {len(frequencies)} frequencies")
    
    # Step 4: Create main dashboard figure
    main_fig = plt.figure(figsize=(18, 12))
    main_fig.suptitle("RFID Angle of Arrival (AoA) Analysis Dashboard", fontsize=20, fontweight='bold')
    gs = GridSpec(2, 2, figure=main_fig)
    
    # Dictionary to store all results for summary
    all_results = {
        'D': [], 'W': [], 'f0': [], 'theta_true': [],
        'theta_phase': [], 'theta_ds': [], 'theta_weighted': [], 'theta_music': [],
        'error_phase': [], 'error_ds': [], 'error_weighted': [], 'error_music': []
    }
    
    # Step 5: Process each distance
    for d_idx, distance in enumerate(distances):
        print(f"Processing distance D = {distance:.2f}m ({d_idx+1}/{len(distances)})")
        
        # 5.1: Filter entries for this distance
        filtered_meta, _ = rfid_data.get_entries_at(D=distance)
        widths = sorted(filtered_meta['W'].unique())
        
        # 5.2: Create distance-specific figure - LARGER FIGURE WITH BETTER LAYOUT
        dist_fig = plt.figure(figsize=(18, 14))
        #  Use LaTeX for plot typography
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        dist_fig.suptitle(f"AoA Analysis @ D = {distance:.2f}m", fontsize=18)
        
        # Use GridSpec with better spacing to avoid overlap
        dist_gs = GridSpec(3, 3, figure=dist_fig, height_ratios=[1, 1.5, 1], width_ratios=[2, 2, 1],
                          hspace=0.35, wspace=0.35)
        
        # 5.3: Create AoA vs W subplot
        ax_aoa = dist_fig.add_subplot(dist_gs[0, :2])
        
        # Arrays to store results for this distance
        theta_true_d = np.array([rfid_data.get_true_angle(distance, w) for w in widths])
        theta_phase_d = np.zeros_like(theta_true_d)
        theta_ds_d = np.zeros_like(theta_true_d)
        theta_w_d = np.zeros_like(theta_true_d)
        theta_music_d = np.zeros_like(theta_true_d)
        
        # Spectra storage for visualization
        spectra_d = []
        
        # 5.4: Process each width
        for w_idx, width in enumerate(widths):
            print(f"  Processing width W = {width:.2f}m")
            
            # Get true angle
            true_angle = rfid_data.get_true_angle(distance, width)
            
            # Set analysis angle range with reduced resolution for speed
            analysis_step = 0.5
            analysis_aoa = np.arange(MIN_ANGLE, MAX_ANGLE + analysis_step, analysis_step)
            
            # Initialize multi-frequency fusion variables with correct dimensions
            B_ds_sum = np.zeros(len(analysis_aoa))
            B_w_sum = np.zeros(len(analysis_aoa))
            P_music_sum = np.zeros(len(analysis_aoa))
            phi_list = []
            rssi1_avg = 0
            rssi2_avg = 0
            freq_count = 0
            
            # Process each frequency
            for freq in frequencies:
                # Get entries for this D, W, f0
                freq_meta, freq_signals = rfid_data.get_entries_at(D=distance, W=width, f0=freq)
                
                if len(freq_meta) == 0:
                    continue
                
                # Get signal data
                signals = freq_signals[0]
                phasor1 = signals['phasor1']
                phasor2 = signals['phasor2']
                rssi1 = signals['rssi1']
                rssi2 = signals['rssi2']
                
                # Get parameters
                L = freq_meta['L'].values[0]
                wavelength = freq_meta['lambda'].values[0]
                
                # Store average RSSI
                rssi1_avg += np.mean(rssi1)
                rssi2_avg += np.mean(rssi2)
                freq_count += 1
                
                # Store phase difference
                phi_list.append(np.angle(np.mean(phasor1)) - np.angle(np.mean(phasor2)))
                
                # Run AoA analysis with same analysis_aoa
                aoa_results = analyze_aoa(
                    phasor1, phasor2, rssi1, rssi2, 
                    L, wavelength, analysis_aoa, true_angle
                )
                
                # Accumulate spectra
                B_ds_sum += aoa_results['spectra']['ds']
                B_w_sum += aoa_results['spectra']['weighted']
                P_music_sum += aoa_results['spectra']['music']
            
            # Skip if no frequencies were processed
            if freq_count == 0:
                continue
                
            # Average results
            B_ds_avg = B_ds_sum / freq_count
            B_w_avg = B_w_sum / freq_count
            P_music_avg = P_music_sum / freq_count
            rssi1_avg /= freq_count
            rssi2_avg /= freq_count
            
            # Store spectra for visualization
            spectra_d.append({
                'W': width,
                'ds_spectrum': B_ds_avg,
                'weighted_spectrum': B_w_avg,
                'music_spectrum': P_music_avg,
                'aoa_range': analysis_aoa
            })
            
            # Find peaks in spectra
            theta_ds_d[w_idx] = analysis_aoa[np.argmax(B_ds_avg)]
            theta_w_d[w_idx] = analysis_aoa[np.argmax(B_w_avg)]
            theta_music_d[w_idx] = analysis_aoa[np.argmax(P_music_avg)]
            
            # Calculate phase-based angle
            mean_dphi = np.angle(np.exp(1j * np.mean(phi_list)))
            sin_theta = (wavelength / (2 * np.pi * L)) * mean_dphi
            theta_phase_d[w_idx] = np.rad2deg(np.arcsin(np.clip(sin_theta, -1, 1)))
            
            # Store results for overall analysis
            all_results['D'].append(distance)
            all_results['W'].append(width)
            all_results['f0'].append(np.mean(freq_meta['f0']))
            all_results['theta_true'].append(true_angle)
            all_results['theta_phase'].append(theta_phase_d[w_idx])
            all_results['theta_ds'].append(theta_ds_d[w_idx])
            all_results['theta_weighted'].append(theta_w_d[w_idx])
            all_results['theta_music'].append(theta_music_d[w_idx])
            all_results['error_phase'].append(abs(theta_phase_d[w_idx] - true_angle))
            all_results['error_ds'].append(abs(theta_ds_d[w_idx] - true_angle))
            all_results['error_weighted'].append(abs(theta_w_d[w_idx] - true_angle))
            all_results['error_music'].append(abs(theta_music_d[w_idx] - true_angle))
        
        # 5.5: Plot AoA vs Width
        ax_aoa.plot(widths, theta_true_d, 'k--o', linewidth=1.5, label='True')
        ax_aoa.plot(widths, theta_phase_d, 'b-s', linewidth=1.5, label='Phase')
        ax_aoa.plot(widths, theta_ds_d, 'r-^', linewidth=1.5, label='DS')
        ax_aoa.plot(widths, theta_w_d, 'm-d', linewidth=1.5, label='DS+RSSI')
        ax_aoa.plot(widths, theta_music_d, 'g-o', linewidth=1.5, label='MUSIC')
        
        ax_aoa.set_xlabel('Width (m)')
        ax_aoa.set_ylabel('AoA (degrees)')
        ax_aoa.set_title(f'AoA Estimation vs Width (D = {distance:.2f}m)')
        ax_aoa.grid(True, alpha=0.3)
        ax_aoa.legend()
        
        # 5.6: Create beam spectra subplot for middle width value
        if len(spectra_d) > 0:
            mid_idx = len(spectra_d) // 2
            mid_spectra = spectra_d[mid_idx]
            
            ax_spectra = dist_fig.add_subplot(dist_gs[0, 2])
            ax_spectra.plot(mid_spectra['aoa_range'], mid_spectra['ds_spectrum'], 
                         'r-', linewidth=1.5, label='DS')
            ax_spectra.plot(mid_spectra['aoa_range'], mid_spectra['weighted_spectrum'], 
                         'm--', linewidth=1.5, label='DS+RSSI')
            ax_spectra.plot(mid_spectra['aoa_range'], mid_spectra['music_spectrum'], 
                         'g-.', linewidth=1.5, label='MUSIC')
            
            ax_spectra.set_xlabel('AoA (degrees)')
            ax_spectra.set_ylabel('Normalized Power')
            ax_spectra.set_title(f'Spectra @ W = {mid_spectra["W"]:.2f}m')
            ax_spectra.grid(True, alpha=0.3)
            ax_spectra.legend()
            
            # 5.7: Create 3D beam pattern visualization - MOVED TO TAKE FULL MIDDLE ROW
            ax_beam3d = dist_fig.add_subplot(dist_gs[1, :2], projection='3d')
            
            # Prepare data for 3D plot
            W_mesh, A_mesh = np.meshgrid(
                [s['W'] for s in spectra_d], 
                spectra_d[0]['aoa_range']
            )
            
            # Create power matrix
            Z = np.zeros(W_mesh.shape)
            for i, s in enumerate(spectra_d):
                Z[:, i] = s['ds_spectrum']
                
            # Create 3D surface
            surf = ax_beam3d.plot_surface(A_mesh, W_mesh, Z, cmap='viridis', 
                                        edgecolor='none', alpha=0.8)
            
            ax_beam3d.set_xlabel('AoA (degrees)')
            ax_beam3d.set_ylabel('Width (m)')
            ax_beam3d.set_zlabel('Power')
            ax_beam3d.set_title('3D DS Beam Pattern')
            plt.colorbar(surf, ax=ax_beam3d, shrink=0.5, aspect=5)
            
            # 5.8: Create heatmaps - REORGANIZED
            ax_heat_ds = dist_fig.add_subplot(dist_gs[1, 2])
            ax_heat_w = dist_fig.add_subplot(dist_gs[2, 2])
            
            # Prepare data for heatmaps
            heatmap_data_ds = np.zeros((len(spectra_d), len(spectra_d[0]['aoa_range'])))
            heatmap_data_w = np.zeros_like(heatmap_data_ds)
            
            for i, s in enumerate(spectra_d):
                heatmap_data_ds[i, :] = s['ds_spectrum']
                heatmap_data_w[i, :] = s['weighted_spectrum']
            
            # Plot heatmaps
            im_ds = ax_heat_ds.imshow(heatmap_data_ds, 
                                    extent=[MIN_ANGLE, MAX_ANGLE, widths[-1], widths[0]],
                                    aspect='auto', cmap='jet')
            plt.colorbar(im_ds, ax=ax_heat_ds)
            ax_heat_ds.set_xlabel('AoA (degrees)')
            ax_heat_ds.set_ylabel('Width (m)')
            ax_heat_ds.set_title('DS Beamforming')
            
            # Plot true angles on heatmap
            ax_heat_ds.plot(theta_true_d, widths, 'w--', linewidth=1.5)
            
            im_w = ax_heat_w.imshow(heatmap_data_w, 
                                  extent=[MIN_ANGLE, MAX_ANGLE, widths[-1], widths[0]],
                                  aspect='auto', cmap='jet')
            plt.colorbar(im_w, ax=ax_heat_w)
            ax_heat_w.set_xlabel('AoA (degrees)')
            ax_heat_w.set_ylabel('Width (m)')
            ax_heat_w.set_title('RSSI-Weighted Beamforming')
            
            # Plot true angles on heatmap
            ax_heat_w.plot(theta_true_d, widths, 'w--', linewidth=1.5)
            
            # 5.9: Add error comparison subplot - MOVED TO BOTTOM ROW
            ax_error = dist_fig.add_subplot(dist_gs[2, 0])
            
            # Calculate error metrics
            error_phase = np.abs(theta_phase_d - theta_true_d)
            error_ds = np.abs(theta_ds_d - theta_true_d)
            error_w = np.abs(theta_w_d - theta_true_d)
            error_music = np.abs(theta_music_d - theta_true_d)
            
            # Calculate mean errors
            mae_phase = np.mean(error_phase)
            mae_ds = np.mean(error_ds)
            mae_w = np.mean(error_w)
            mae_music = np.mean(error_music)
            
            # Plot error bars
            methods = ['Phase', 'DS', 'DS+RSSI', 'MUSIC']
            maes = [mae_phase, mae_ds, mae_w, mae_music]
            
            ax_error.bar(methods, maes)
            ax_error.set_ylabel('Mean Absolute Error (degrees)')
            ax_error.set_title('Error Comparison')
            ax_error.grid(True, alpha=0.3)
            
            # Add error values as text
            for i, v in enumerate(maes):
                ax_error.text(i, v + 0.1, f'{v:.2f}°', ha='center')
                
            # 5.10: Add error vs width plot
            ax_error_w = dist_fig.add_subplot(dist_gs[2, 1])
            ax_error_w.plot(widths, error_phase, 'b-s', linewidth=1.5, label='Phase')
            ax_error_w.plot(widths, error_ds, 'r-^', linewidth=1.5, label='DS')
            ax_error_w.plot(widths, error_w, 'm-d', linewidth=1.5, label='DS+RSSI')
            ax_error_w.plot(widths, error_music, 'g-o', linewidth=1.5, label='MUSIC')
            
            ax_error_w.set_xlabel('Width (m)')
            ax_error_w.set_ylabel('Error (degrees)')
            ax_error_w.set_title('Error vs Width Position')
            ax_error_w.grid(True, alpha=0.3)
            ax_error_w.legend()
        
        # 5.11: Save distance-specific figure
        plt.tight_layout()
        dist_fig.savefig(os.path.join(dashboard_dir, f'aoa_analysis_D{distance:.2f}.png'), 
                        dpi=300, bbox_inches='tight')
        plt.close(dist_fig)
        
        # 5.12: Add to main dashboard if we have space (first 4 distances)
        if d_idx < 4:
            # Add to main dashboard
            ax_main = main_fig.add_subplot(gs[d_idx // 2, d_idx % 2])
            
            # Plot results
            ax_main.plot(widths, theta_true_d, 'k--o', linewidth=1.5, label='True')
            ax_main.plot(widths, theta_phase_d, 'b-s', linewidth=1.5, label='Phase')
            ax_main.plot(widths, theta_ds_d, 'r-^', linewidth=1.5, label='DS')
            ax_main.plot(widths, theta_w_d, 'm-d', linewidth=1.5, label='DS+RSSI')
            ax_main.plot(widths, theta_music_d, 'g-o', linewidth=1.5, label='MUSIC')
            
            ax_main.set_xlabel('Width (m)')
            ax_main.set_ylabel('AoA (degrees)')
            ax_main.set_title(f'D = {distance:.2f}m (MAE: Ph={mae_phase:.1f}°, DS={mae_ds:.1f}°, W={mae_w:.1f}°, MU={mae_music:.1f}°)')
            ax_main.grid(True, alpha=0.3)
            ax_main.legend()
    
    # Step 6: Generate summary visualizations
    all_results_df = pd.DataFrame(all_results)
    
    # Create comprehensive method comparison figure
    comp_fig = plt.figure(figsize=(16, 10))
    comp_fig.suptitle('AoA Method Comparison', fontsize=16)
    comp_gs = GridSpec(2, 2, figure=comp_fig)
    
    # 6.1: Create boxplot comparison
    ax_boxplot = comp_fig.add_subplot(comp_gs[0, 0])
    
    # Prepare error data
    error_data = [
        all_results_df['error_phase'],
        all_results_df['error_ds'],
        all_results_df['error_weighted'],
        all_results_df['error_music']
    ]
    
    # Create boxplot
    box = ax_boxplot.boxplot(error_data, labels=['Phase', 'DS', 'DS+RSSI', 'MUSIC'])
    ax_boxplot.set_ylabel('Absolute Error (degrees)')
    ax_boxplot.set_title('Error Distribution by Method')
    ax_boxplot.grid(True, alpha=0.3)
    
    # 6.2: Create error CDF plot
    ax_cdf = comp_fig.add_subplot(comp_gs[0, 1])
    
    # Create CDF plots
    colors = ['b', 'r', 'm', 'g']
    method_names = ['Phase', 'DS', 'DS+RSSI', 'MUSIC']
    
    for i, err in enumerate([
        all_results_df['error_phase'], 
        all_results_df['error_ds'], 
        all_results_df['error_weighted'], 
        all_results_df['error_music']
    ]):
        # Sort error values
        sorted_err = np.sort(err)
        # Calculate CDF
        cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
        # Plot CDF
        ax_cdf.plot(sorted_err, cdf, colors[i], linewidth=1.5, label=method_names[i])
    
    ax_cdf.set_xlabel('Absolute Error (degrees)')
    ax_cdf.set_ylabel('Cumulative Probability')
    ax_cdf.set_title('Error CDF by Method')
    ax_cdf.grid(True, alpha=0.3)
    ax_cdf.legend(loc='lower right')
    
    # 6.3: Create method performance table
    performance = []
    
    for method in ['phase', 'ds', 'weighted', 'music']:
        err = all_results_df[f'error_{method}']
        performance.append([
            np.mean(err),      # MAE
            np.median(err),    # Median
            np.std(err),       # Std
            np.max(err)        # Max
        ])
    
    # Create text-based table
    ax_table = comp_fig.add_subplot(comp_gs[1, 0])
    col_labels = ['MAE (°)', 'Median (°)', 'Std (°)', 'Max (°)']
    row_labels = ['Phase', 'DS', 'DS+RSSI', 'MUSIC']
    
    # Turn off axis
    ax_table.axis('tight')
    ax_table.axis('off')
    
    # Create table
    table = ax_table.table(
        cellText=[[f'{v:.2f}' for v in row] for row in performance],
        rowLabels=row_labels,
        colLabels=col_labels,
        loc='center',
        cellLoc='center'
    )
    
    # Set table properties
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    ax_table.set_title('Method Performance Metrics')
    
    # 6.4: Create 3D error visualization
    ax_error3d = comp_fig.add_subplot(comp_gs[1, 1], projection='3d')
    
    # Get data
    D_vals = all_results_df['D']
    W_vals = all_results_df['W']
    ph_error = all_results_df['error_phase']
    dsw_error = all_results_df['error_weighted']
    
    # Create scatter plot
    sc1 = ax_error3d.scatter(D_vals, W_vals, ph_error, c=ph_error, marker='o', 
                           cmap='viridis', s=50, alpha=0.7, label='Phase')
    sc2 = ax_error3d.scatter(D_vals, W_vals, dsw_error, c=dsw_error, marker='^', 
                           cmap='plasma', s=50, alpha=0.7, label='DS+RSSI')
    
    ax_error3d.set_xlabel('Distance (m)')
    ax_error3d.set_ylabel('Width (m)')
    ax_error3d.set_zlabel('Error (degrees)')
    ax_error3d.set_title('3D Error Distribution')
    plt.colorbar(sc1, ax=ax_error3d, shrink=0.5, aspect=5, label='Error (degrees)')
    ax_error3d.legend()
    
    # Adjust view
    ax_error3d.view_init(30, 45)
    
    # Save comparison figure
    plt.tight_layout()
    comp_fig.savefig(os.path.join(dashboard_dir, 'method_comparison.png'), 
                   dpi=300, bbox_inches='tight')
    plt.close(comp_fig)
    
    # 6.5: Create 3D error analysis figure
    fig3d = plt.figure(figsize=(15, 10))
    fig3d.suptitle('3D Error Analysis by Method', fontsize=16)
    gs3d = GridSpec(2, 2, figure=fig3d)
    
    methods = ['phase', 'ds', 'weighted', 'music']
    method_names = ['Phase', 'DS', 'DS+RSSI', 'MUSIC']
    
    for i, method in enumerate(methods):
        ax = fig3d.add_subplot(gs3d[i // 2, i % 2], projection='3d')
        
        # Get error data
        error = all_results_df[f'error_{method}']
        
        # Create scatter plot
        sc = ax.scatter(D_vals, W_vals, error, c=error, cmap='viridis', s=50, alpha=0.7)
        
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Width (m)')
        ax.set_zlabel('Error (degrees)')
        ax.set_title(f'{method_names[i]} Error')
        plt.colorbar(sc, ax=ax, shrink=0.5, aspect=5, label='Error (degrees)')
        
        # Adjust view
        ax.view_init(30, 45)
    
    # Save 3D figure
    plt.tight_layout()
    fig3d.savefig(os.path.join(dashboard_dir, '3d_error_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close(fig3d)
    
    # Save the main dashboard
    plt.tight_layout()
    main_fig.savefig(os.path.join(dashboard_dir, 'main_dashboard.png'), 
                   dpi=300, bbox_inches='tight')
    plt.close(main_fig)
    
    # Save results to CSV
    all_results_df.to_csv(os.path.join(dashboard_dir, 'aoa_analysis_results.csv'), index=False)
    
    # Print summary
    print("\n=== RFID AoA ANALYSIS SUMMARY ===")
    print(f"Processed data at {len(distances)} distances")
    print("Method Performance:")
    for i, method in enumerate(['Phase', 'DS', 'DS+RSSI', 'MUSIC']):
        print(f"  {method}: MAE={performance[i][0]:.2f}°, Median={performance[i][1]:.2f}°, "
              f"Std={performance[i][2]:.2f}°, Max={performance[i][3]:.2f}°")
    
    print(f"\nVisualizations saved to: {dashboard_dir}")
    print("Analysis complete!")
    
    return all_results_df
# =================================================================================================================================== #

def main():
    # STEP 1: Create a DataManager instance - RFID_data_manager
    RFID_data_manager = DataManager(data_dir=DATA_DIRECTORY, tag_id=TAG_ID, aoa_range=AoA_m)
    
    # STEP 2: Import data
    RFID_data_manager.import_data()

    # STEP 3: Run traditional AoA analysis
    print("Starting individual file analysis...")
    results = RFID_data_manager.analyze_all_data(save_results=SAVE_RESULTS)

    # STEP 4: Generate summary visualizations for individual analysis
    if SAVE_RESULTS and len(results) > 0:
        # Create summary plots directory
        summary_dir = os.path.join(RESULTS_DIRECTORY, "summary")
        os.makedirs(summary_dir, exist_ok=True)
        
        # Plot 1: Error distribution by method
        plt.figure(figsize=(10, 6))
        #  Use LaTeX for plot typography
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        methods = ['phase', 'ds', 'weighted', 'music']
        method_names = ['Phase', 'Beamforming', 'Weighted BF', 'MUSIC']
        
        # Create boxplot of errors
        error_data = [results[f'error_{m}'] for m in methods]
        plt.boxplot(error_data, labels=method_names)
        #  Use LaTeX for plot typography
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.ylabel('Error (degrees)')
        plt.title('AoA Estimation Error by Method')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(summary_dir, "error_comparison.png"), dpi=300)
        plt.close()
        
        # Plot 2: Error vs. Position (scatter plot)
        plt.figure(figsize=(12, 8))
        #  Use LaTeX for plot typography
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        for i, m in enumerate(methods):
            plt.subplot(2, 2, i+1)
            sc = plt.scatter(results['W'], results['D'], c=results[f'error_{m}'], 
                           cmap='viridis', alpha=0.8, s=50)
            plt.colorbar(sc, label='Error (degrees)')
            plt.xlabel('Width (m)')
            plt.ylabel('Distance (m)')
            plt.title(f'{method_names[i]} Error')
            plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(summary_dir, "error_vs_position.png"), dpi=300)
        plt.close()
        
        # Plot 3: Error vs. Frequency
        plt.figure(figsize=(10, 6))
        #  Use LaTeX for plot typography
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        for i, m in enumerate(methods):
            plt.plot(results['f0']/1e6, results[f'error_{m}'], 'o-', label=method_names[i])
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Error (degrees)')
        plt.title('AoA Estimation Error vs. Frequency')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(summary_dir, "error_vs_frequency.png"), dpi=300)
        plt.close()
        
        print(f"Saved summary visualizations to: {summary_dir}")
    
    # STEP 5: Run comprehensive dashboard analysis
    print("\nStarting comprehensive dashboard analysis...")
    dashboard_results = create_dashboard()
    
    # STEP 6: Train machine learning model
    bayesian_results = train_bayesian_models(RFID_data_manager, RESULTS_DIRECTORY, num_epochs=14000)

    # STEP 7: Create detailed visualizations for the best model
    print("\nGenerating detailed visualizations for best model...")
    # Find the best model (lowest MAE)
    best_model_name = None
    best_mae = float('inf')
    for name, result in bayesian_results['results'].items():
        if result['mae'] < best_mae:
            best_mae = result['mae']
            best_model_name = name
    
    if best_model_name:
        best_model = bayesian_results['models'][best_model_name]
        detailed_dir = os.path.join(RESULTS_DIRECTORY, "best_model_details")
        os.makedirs(detailed_dir, exist_ok=True)
        
        # Generate additional visualizations for the best model
        best_model.render_model_and_guide(detailed_dir, "best_model")
        best_model.plot_posterior_predictive(detailed_dir, "best_model")
        best_model.plot_uncertainty_calibration(detailed_dir, "best_model")
        best_model.visualize_weight_distributions(detailed_dir, "best_model")
        
        print(f"Best model: {best_model_name} (MAE: {best_mae:.4f}°)")
        print(f"Detailed visualizations saved to: {detailed_dir}")
    
    print("AoA analysis completed successfully!")
    return RFID_data_manager, bayesian_results

if __name__ == "__main__":
    data_manager, bayesian_results = main()
    print("\nAnalysis complete!")

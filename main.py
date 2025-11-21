# =================================================================================================================================== #
# ----------------------------------------------------------- DESCRIPTION ----------------------------------------------------------- #
# This script provides the main functionality for analyzing Angle of Arrival (AoA) data from RFID systems. It imports data,           #
# performs AoA estimation using various methods, and visualizes the results. The script also includes machine learning components     #
# for regression analysis on AoA data.                                                                                                #  
# =================================================================================================================================== #


# =================================================================================================================================== #
# --------------------------------------------------------- EXTERNAL IMPORTS -------------------------------------------------------- #
import os                                               # Operating system interfaces for file and directory manipulation.            #
import glob                                             # Unix style pathname pattern expansion for file searching.                   #
import re                                               # Regular expression operations for pattern matching in filenames.            #
import pickle                                           # Object serialization and deserialization.                                   #
import torch                                            # PyTorch for tensor computations and neural networks.                        #
import numpy                   as np                    # Mathematical functions.                                                     #
import scipy.constants         as sc                    # Physical and mathematical constants.                                        #
import matplotlib.pyplot       as plt                   # Data visualization.                                                         #
import src.data_management     as dm                    # Data management functions for importing and organizing data.                #
import src.phase_difference    as pad                   # Phase difference calculations for AoA estimation.                           #  
import src.beamforming         as bf                    # Beamforming methods for AoA estimation.                                     #
import src.music               as music                 # MUSIC algorithm for high-resolution AoA estimation.                         #
import src.bayesian_regression as br                    # Bayesian regression for machine learning models on AoA data.                #
import src.visualization       as vis                   # Visualization functions for AoA analysis results.                           #
import pandas                  as pd                    # Data manipulation and analysis library.                                     #
import seaborn                 as sns                   # Statistical data visualization library.                                     #
import matplotlib              as mpl                   # Comprehensive library for creating visualizations.                          #
import matplotlib.pyplot       as plt                   # MATLAB-like plotting framework.                                             #
from   tqdm                    import tqdm              # Progress bar for loops, useful for tracking long-running operations.        #
from   cycler                  import cycler            # Customizing matplotlib color and style cycles.                              #
from   sklearn.model_selection import train_test_split  # For train-test splitting.                                                   #
from sklearn.metrics           import mean_absolute_error, mean_squared_error                                                         #
from src.hblr_aoa              import constant_sigma_phys_from_rmse                                                                   #
mpl.use('Agg')                                          # Use 'Agg' backend for non-interactive plotting (suitable for scripts).      #
# =================================================================================================================================== #


# =================================================================================================================================== #
# ------------------------------------------------------- CONFIGURATION SETTINGS ---------------------------------------------------- #
SCRIPT_DIR         = os.path.dirname(os.path.abspath(__file__))                    # Get the directory of the current script.         #
#PROJECT_ROOT       = os.path.dirname(SCRIPT_DIR)                                  # Go up one level to project root.                 #
PROJECT_ROOT       = SCRIPT_DIR                                                    # Go up one level to project root.                 #
DATA_DIRECTORY     = os.path.join(PROJECT_ROOT, 'data', '2025-07-09')              # Directory containing the data files.             #
RESULTS_BASE_DIR   = os.path.join(PROJECT_ROOT, 'results')                         # Store results in a separate folder.              #
EXPERIMENT_NAME    = 'AoA_Analysis'                                                # Name of the experiment for output directory.     #
RESULTS_DIRECTORY  = dm.create_output_directory(RESULTS_BASE_DIR, EXPERIMENT_NAME) # Directory to save results.                       #
SAVE_RESULTS       = True                                                          # Flag to save results to file.                    #             
TAG_ID             = '000233b2ddd9014000000000'                                    # Target tag ID.                                   #
TAG_NAME           = "Belt DEE"                                                    # Default tag name for the analysis.               # 
STEP               = 0.0001                                                        # Step size for the AoA (theta_m) sweep, in deg.   #
MIN_ANGLE          = -90                                                           # Minimum angle for the AoA sweep, in degrees.     #
MAX_ANGLE          = 90                                                            # Maximum angle for the AoA sweep, in degrees.     #
BAYESIAN_MIN_ANGLE = -15                                                           # Restricted angle range for Bayesian analysis.    #
BAYESIAN_MAX_ANGLE = 15                                                            # Restricted angle range for Bayesian analysis.    #
AoA_m              = np.arange(MIN_ANGLE, MAX_ANGLE + STEP, STEP)                  # Array of angles for the AoA sweep, in degrees.   #
c                  = sc.speed_of_light                                             # Speed of light in m/s.                           #
MIN_DATA_POINTS    = 1                                                             # Min. number of data points required for analysis.#
os.makedirs(RESULTS_BASE_DIR, exist_ok=True)                                       # Create results directory if it doesn't exist.    #
# =================================================================================================================================== #


# =================================================================================================================================== #
# ---------------------------------------------------------- PLOTTING SETTINGS ------------------------------------------------------ #
COLOR_CYCLE  = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']                  #
MARKER_CYCLE = ['o','s','D','^','v','<','>','p','P','*']                                                                              #
custom_cycler = (cycler(color=COLOR_CYCLE) * cycler(marker=MARKER_CYCLE[:len(COLOR_CYCLE)])*cycler(linestyle=['-']*len(COLOR_CYCLE))) #
TITLE_SIZE, LABEL_SIZE, LEGEND_SIZE, TICK_SIZE, ANNOTATION_SIZE = 24, 20, 20, 18, 12                                                  #
FIG_WIDTH, FIG_HEIGHT, LINE_WIDTH, MARKER_SIZE = 10, 7, 1.75, 8                                                                       #
plt.style.use("seaborn-v0_8-whitegrid")                                                                                               #
mpl.rcParams['axes.prop_cycle'] = custom_cycler                                                                                       #
plt.rcParams.update({                                                                                                                 #
    "figure.figsize": (FIG_WIDTH, FIG_HEIGHT),                                                                                        #
    "figure.dpi": 600,                                                                                                                #
    "figure.titlesize": TITLE_SIZE,                                                                                                   #
    "font.family": "serif",                                                                                                           #
    "font.serif": ["Computer Modern Roman", "Times New Roman"],                                                                       #
    "font.size": LABEL_SIZE,                                                                                                          #
    "axes.titlesize": TITLE_SIZE,                                                                                                     #
    "axes.labelsize": LABEL_SIZE,                                                                                                     #
    "axes.linewidth": 1.2,                                                                                                            #
    "axes.grid": True,                                                                                                                #
    "axes.grid.which": "both",                                                                                                        #
    "axes.grid.axis": "both",                                                                                                         #
    "xtick.labelsize": TICK_SIZE,                                                                                                     #
    "ytick.labelsize": TICK_SIZE,                                                                                                     #
    "xtick.major.width": 1.0,                                                                                                         #
    "ytick.major.width": 1.0,                                                                                                         #
    "legend.fontsize": LEGEND_SIZE,                                                                                                   #
    "legend.framealpha": 0.8,                                                                                                         #
    "legend.edgecolor": "0.8",                                                                                                        #
    "legend.fancybox": True,                                                                                                          #
    "legend.markerscale": 1.2,                                                                                                        #
    "lines.linewidth": LINE_WIDTH,                                                                                                    #
    "lines.markersize": MARKER_SIZE,                                                                                                  #
    "lines.markeredgewidth": 1.2,                                                                                                     #
    "text.usetex": True,                                                                                                              #
    "text.latex.preamble": r"\usepackage{amsmath,amssymb,amsfonts,mathrsfs}",                                                         #
})                                                                                                                                    #
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)                                                                     #
TAG_NAME = "Belt DEE"                                                                                                                 #
plt.rc('text', usetex=True)                                                                                                           #
plt.rc('font', family='serif')                                                                                                        #
# =================================================================================================================================== #


# =================================================================================================================================== #
# ------------------------------------------------------- AoA & DATA MANAGEMENT ----------------------------------------------------- #
def analyze_aoa(phasor1, phasor2, rssi1, rssi2, L, wavelength, aoa_scan, true_angle=None):
    """
    Comprehensive AoA estimation using multiple methods: phase difference,
    standard beamforming, RSSI-weighted beamforming, and MUSIC algorithm.
    These are used as physics-informed estimations, which are then intrduced
    into a hierarchal Bayesian regression model for improved accuracy.
    
    Parameters:
        - phasor1    [np.ndarray]       : Complex phasors from antenna 1
        - phasor2    [np.ndarray]       : Complex phasors from antenna 2
        - rssi1      [np.ndarray]       : RSSI values from antenna 1 (in dBm)
        - rssi2      [np.ndarray]       : RSSI values from antenna 2 (in dBm)
        - L          [float]            : Antenna separation distance (in meters)
        - wavelength [float]            : Signal wavelength (in meters)
        - aoa_scan   [np.ndarray]       : Array of angles to scan (in degrees)
        - true_angle [float] (optional) : True angle for error calculation
        
    Returns:
        - dict: Dictionary containing:
            - 'angles'     : Estimated angles from each method
            - 'spectra'    : Beamforming and MUSIC spectra
            - 'phase_diff' : Phase difference between antennas
            - 'errors'     : Errors for each method if true_angle is provided
    """
    # 1. Phase difference method
    dphi     = pad.compute_phase_difference(phasor1, phasor2)
    theta_ph = pad.phase_difference_aoa(dphi, L, wavelength)
    #pad.run_aoa_analysis(base_dir='/home/nedal/Desktop/RFID/ICASSP2026/Bayesian-Enhanced-AoA-Estimator/data/2025-07-09', tag_id='000233b2ddd9014000000000')
    # 2. Beamforming methods
    B_ds, B_w, theta_ds, theta_w = bf.beamforming_spectrum_calculation(phasor1, phasor2, rssi1, rssi2, L, wavelength, aoa_scan)
    # 3. MUSIC algorithm
    theta_mu, P_music = music.music_algorithm(phasor1, phasor2, L, wavelength, aoa_scan)
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

class DataManager:
    """
    DataManager class for importing, organizing, and analyzing RFID data. 

    This class handles the import of CSV files containing RFID signal data, extracts metadata from filenames,
    organizes the data into structured formats, and provides methods for filtering and analyzing the data.

    Attributes:
        - data_dir    [str]          : Directory containing the CSV files.
        - tag_id      [str]          : RFID tag ID to filter the data.
        - aoa_range   [np.ndarray]   : Array of angles for the AoA (theta_m) sweep.
        - metadata    [pd.DataFrame] : DataFrame containing scalar metadata for each file.
        - signal_data [list]         : List of dictionaries containing NumPy arrays of signal data.
        - results     [pd.DataFrame] : DataFrame with results of the analysis.
        - frequencies [np.ndarray]   : Unique frequency values found in the data.
        - distances   [np.ndarray]   : Unique distance values found in the data.
        - widths      [np.ndarray]   : Unique width values found in the data.
    """

    def __init__(self, data_dir = DATA_DIRECTORY, tag_id = TAG_ID, aoa_range = AoA_m):
        """
        Initializes the DataManager with the specified parameters. If no parameters are provided, 
        defaults are used from the configuration settings.

        Parameters:
            - data_dir  [str]         : Directory containing the CSV files. Default is DATA_DIRECTORY.
            - tag_id    [str]         : RFID tag ID to filter the data. Default is TAG_ID.
            - aoa_range [np.ndarray]  : Array of angles for the AoA (theta_m) sweep. Default is AoA_m.
        """
        self.data_dir    = data_dir
        self.tag_id      = tag_id
        self.aoa_range   = aoa_range
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
            - self [DataManager]: The DataManager instance with imported data and metadata.
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
                min_samples  = min(len(phi1), len(phi2), len(rssi1), len(rssi2))
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
            - D  [float] (optional) : Distance to filter by, in meters.
            - W  [float] (optional) : Width to filter by, in meters.
            - f0 [float] (optional) : Frequency to filter by, in Hz.

        Returns:
            - filtered_meta    [pd.DataFrame] : DataFrame containing filtered metadata.
            - filtered_signals [list]         : List of dictionaries containing filtered signal data.
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
        filtered_meta    = self.metadata[meta_filter]
        # STEP 3: Get corresponding signal data
        filtered_signals = [self.signal_data[i] for i in filtered_meta.index]
        # STEP 4: Return Results
        return filtered_meta, filtered_signals
    
    def get_true_angle(self, D, W):
        """
        Calculate true angle based on geometry, by applying simple trigonometry.

        This method uses the distance (D) and width (W) to compute the angle in degrees.

        Parameters:
            - D [float]: Distance, in meters.
            - W [float]: Width, in meters.

        Returns:
            - angle [float]: True angle, in degrees.
        """
        return np.rad2deg(np.arctan2(W, D))
    
    def compute_phase_difference(self, entry_index):
        """
        Compute average phase difference for an entry, given its index in the signal_data list. 

        This method calculates the phase difference between two antennas' phasors for a specific entry.

        Parameters:
            - entry_index [int]: Index of the entry in the signal_data list.

        Returns:
            - dphi [float]: Phase difference in radians, wrapped to [-π, π].
        """
        # STEP 1: Obtain signals for the specified entry index.
        signals = self.signal_data[entry_index]
        phasor1 = signals['phasor1']
        phasor2 = signals['phasor2']
        # STEP 2: Calculate phase difference
        dphi    = np.angle(np.mean(phasor1)) - np.angle(np.mean(phasor2))
        # STEP 3: Ensure in correct range
        return np.angle(np.exp(1j * dphi))
    
    def prepare_ml_features(self):
        """
        Prepare features for machine learning models. 

        This method creates feature matrices based on the results of the AoA analysis, including both basic and RSSI-weighted features.

        Returns:
            - dict: A dictionary containing:
                - X_basic                [np.ndarray] : Basic feature matrix.
                - X_weighted             [np.ndarray] : RSSI-weighted feature matrix.
                - y                      [np.ndarray] : Target variable (true angles).
                - feature_names_basic    [list]       : Names of basic features.
                - feature_names_weighted [list]       : Names of weighted features.
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
        
        Parameters:
            - save_results [bool]: Whether to save results to files
            
        Returns:
            - pd.DataFrame: Results dataframe with AoA estimates for all methods
        """
        # Initialize results list
        results_list = []
        # Create output directories if saving results
        if save_results:
            plots_dir   = os.path.join(RESULTS_DIRECTORY, "plots")
            results_dir = os.path.join(RESULTS_DIRECTORY, "results")
            os.makedirs(plots_dir, exist_ok=True)
            os.makedirs(results_dir, exist_ok=True)
        # Process each entry in the metadata
        for idx, meta in tqdm(self.metadata.iterrows(), total=len(self.metadata), desc="Analyzing data"):
            # Extract parameters
            D  = meta['D']
            W  = meta['W']
            L  = meta['L']
            f0 = meta['f0']
            wavelength = meta['lambda']
            true_angle = self.get_true_angle(D, W)
            # Get signal data
            signals = self.signal_data[idx]
            phasor1 = signals['phasor1']
            phasor2 = signals['phasor2']
            rssi1   = signals['rssi1']
            rssi2   = signals['rssi2']
            analysis_step      = 0.5  
            analysis_aoa_range = np.arange(MIN_ANGLE, MAX_ANGLE + analysis_step, analysis_step)
            # Run AoA analysis
            aoa_results = analyze_aoa(
                phasor1, phasor2, rssi1, rssi2, 
                L, wavelength, analysis_aoa_range, true_angle
            )
            # Save visualization if requested
            if save_results:
                # Title
                title = f"AoA Analysis (D={D:.2f}m, W={W:.2f}m, f={f0/1e6:.2f}MHz, True $\\theta$={true_angle:.2f}deg)"
                # Figure generation
                fig = vis.visualize_aoa_results(aoa_results, analysis_aoa_range, title)
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
# =================================================================================================================================== #  


# =================================================================================================================================== #
# ------------------------------------------------------------- EXPERIMENTS --------------------------------------------------------- #
def run_model_grid(data_dir=DATA_DIRECTORY, tag_id=TAG_ID, inference_schemes=('svi','mcmc'), priors=('ds','weighted','music','phase'),
                   feature_modes=('full','sensor_only','width_only'), results_dir=RESULTS_DIRECTORY):
    """
    Run a grid of Bayesian AoA regression models with different priors, feature sets, and inference schemes. The function does:
        - Train 24 models: 4 priors × 3 feature sets × 2 inference schemes.
        - Saves a CSV summary and per-model visualizations.
    
    Parameters:
        - data_dir          [str]   : Directory containing the data files.
        - tag_id            [str]   : RFID tag ID to filter the data.
        - inference_schemes [tuple] : Inference schemes to use ('svi', 'mcmc').
        - priors            [tuple] : Prior types to use ('ds', 'weighted', 'music', 'phase').
        - feature_modes     [tuple] : Feature modes to use ('full', 'sensor_only', 'width_only').
        - results_dir       [str]   : Directory to save results.

    Returns:
        - pd.DataFrame: Summary of model performances across the grid.
    """
    dmgr = DataManager(data_dir=data_dir, tag_id=tag_id, aoa_range=AoA_m)
    dmgr.import_data()
    dmgr.analyze_all_data(save_results=False)

    rows = []
    for inf in inference_schemes:
        for pr in priors:
            for fm in feature_modes:
                print(f"\n=== Running {inf.upper()} | prior={pr} | features={fm} ===")
                reg = br.BayesianAoARegressor(use_gpu=True, prior_type=pr, feature_mode=fm, obs_sigma=0.1, inference=inf)
                summary = reg.train(dmgr, num_epochs=14000 if inf=='svi' else 14000, learning_rate=5e-4, batch_size=256, verbose=True)
                # Visualize with prior-vs-posterior & importance baked in
                exp_name = f"{pr}_{fm}_{inf}"
                reg.visualize_results(results_dir, exp_name)
                reg.visualize_weight_distributions(results_dir, exp_name)
                rows.append({
                    'inference': inf, 
                    'prior': pr, 
                    'features': fm,
                    'mae': summary['mae'], 
                    'rmse': summary['rmse'],
                    'sigma_phys_const': summary.get('sigma_phys_const', None),
                    'elapsed_time': summary.get('elapsed_time', None),
                    'final_loss': summary.get('final_loss', None)
                })

    df = pd.DataFrame(rows)
    os.makedirs(results_dir, exist_ok=True)
    out_csv = os.path.join(results_dir, "grid_summary.csv")
    df.to_csv(out_csv, index=False)
    print("\nSaved grid summary:", out_csv)
    return df

def run_small_sample_sweeps(inference_schemes=('svi',), priors=('ds','weighted','music','phase'), feature_modes=('full','sensor_only','width_only'),
                            results_dir=RESULTS_DIRECTORY, fractions=(0.05,0.1,0.2,0.4,0.6,0.8, 0.9, 0.95, 1.00), repeats=3):
    """
    Run sample sweeps, to validate performance in the small-sample regime.

    Parameters:
        - inference_schemes [tuple] : Inference schemes to use ('svi', 'mcmc').
        - priors            [tuple] : Prior types to use ('ds', 'weighted', 'music', 'phase').
        - feature_modes     [tuple] : Feature modes to use ('full', 'sensor_only', 'width_only').
        - results_dir       [str]   : Directory to save results.
        - fractions         [tuple] : Fractions of data to use for training.
        - repeats           [int]   : Number of repeats per fraction.

    Returns:
        - dict: Nested dictionary with results for each configuration.
    """
    dmgr = DataManager(data_dir=DATA_DIRECTORY, tag_id=TAG_ID, aoa_range=AoA_m)
    dmgr.import_data()
    dmgr.analyze_all_data(save_results=False)

    res = {}
    for inf in inference_schemes:
        res[inf] = {}
        for pr in priors:
            res[inf][pr] = {}
            for fm in feature_modes:
                print(f"\n=== Sweep {inf.upper()} | prior={pr} | features={fm} ===")
                reg = br.BayesianAoARegressor(use_gpu=True, prior_type=pr, feature_mode=fm, obs_sigma=0.1, inference=inf)
                out = reg.sweep_small_sample(dmgr, fractions=fractions, repeats=repeats)
                res[inf][pr][fm] = out

    os.makedirs(results_dir, exist_ok=True)
    # Use different filenames for different inference schemes
    inference_str = '_'.join(inference_schemes)
    path = os.path.join(results_dir, f"small_sample_sweeps_{inference_str}.pkl")
    with open(path, "wb") as f:
        pickle.dump(res, f)
    print("Saved small-sample sweeps to:", path)
    return res

def visualize_small_sample_results(results_dict, output_dir):
    """
    Visualize small sample regime results across different model configurations.
    
    Parameters:
        - results_dict [dict] : Nested dictionary from run_small_sample_sweeps
        - output_dir   [str]  : Directory to save visualizations
        
    Returns:
        - None (saves plots to disk)
    """
    # Create output directory
    vis_dir = os.path.join(output_dir, "small_sample_analysis")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Extract all unique fractions across all experiments
    all_fractions = set()
    for inf in results_dict.values():
        for prior in inf.values():
            for feature in prior.values():
                all_fractions.update(feature['fractions'])
    all_fractions = sorted(list(all_fractions))
    
    # Plot MAE vs sample size for each inference method
    fig, axes = plt.subplots(1, len(results_dict), figsize=(7*len(results_dict), 6), sharey=True)
    if len(results_dict) == 1:
        axes = [axes]
    
    for i, (inf_name, inf_data) in enumerate(results_dict.items()):
        ax = axes[i]
        for prior_name, prior_data in inf_data.items():
            for feat_name, feat_data in prior_data.items():
                label = f"{prior_name}_{feat_name}"
                # Plot mean with shaded error region
                fractions = np.array(feat_data['fractions'])
                mae_mean  = np.array(feat_data['mae_mean'])
                mae_std   = np.array(feat_data['mae_std'])
                
                # Sort by fraction
                sort_idx  = np.argsort(fractions)
                fractions = fractions[sort_idx]
                mae_mean  = mae_mean[sort_idx]
                mae_std   = mae_std[sort_idx]
                
                ax.plot(fractions, mae_mean, 'o-', label=label)
                ax.fill_between(fractions, mae_mean - mae_std, mae_mean + mae_std, alpha=0.2)
        
        ax.set_title(f'{inf_name.upper()}')
        ax.set_xlabel('Data Fraction')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_ylabel('MAE (degrees)')
        ax.legend()
    
    plt.suptitle('Mean Absolute Error vs Data Size', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "mae_vs_sample_size.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Plot RMSE vs sample size for each inference method
    fig, axes = plt.subplots(1, len(results_dict), figsize=(7*len(results_dict), 6), sharey=True)
    if len(results_dict) == 1:
        axes = [axes]
    
    for i, (inf_name, inf_data) in enumerate(results_dict.items()):
        ax = axes[i]
        for prior_name, prior_data in inf_data.items():
            for feat_name, feat_data in prior_data.items():
                label = f"{prior_name}_{feat_name}"
                # Plot mean with shaded error region
                fractions = np.array(feat_data['fractions'])
                rmse_mean = np.array(feat_data['rmse_mean'])
                rmse_std  = np.array(feat_data['rmse_std'])
                
                # Sort by fraction
                sort_idx  = np.argsort(fractions)
                fractions = fractions[sort_idx]
                rmse_mean = rmse_mean[sort_idx]
                rmse_std  = rmse_std[sort_idx]
                
                ax.plot(fractions, rmse_mean, 'o-', label=label)
                ax.fill_between(fractions, rmse_mean - rmse_std, rmse_mean + rmse_std, alpha=0.2)
        
        ax.set_title(f'{inf_name.upper()}')
        ax.set_xlabel('Data Fraction')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_ylabel('RMSE (degrees)')
        ax.legend()
    
    plt.suptitle('Root Mean Square Error vs Data Size', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "rmse_vs_sample_size.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Add a call to this function in run_small_sample_sweeps
    print(f"Small sample analysis visualizations saved to {vis_dir}")
    return vis_dir

def validate_all_models_from_pickle(pickle_path, data_manager, output_dir):
    """
    Load all trained models from pickle and validate each one against closed-form posterior.

    Parameters:
        - pickle_path   [str]         : Path to the pickle file containing trained models.
        - data_manager  [DataManager] : Instance of DataManager with data loaded.
        - output_dir    [str]         : Directory to save validation results.

    Returns:
        - None (saves validation results to disk)
    """
    # Load pickle
    print(f"Loading models from {pickle_path}...")
    with open(pickle_path, 'rb') as f:
        bayesian_results = pickle.load(f)
    
    # Create validation directory
    val_dir = os.path.join(output_dir, "all_models_validation")
    os.makedirs(val_dir, exist_ok=True)
    
    validation_results = {}
    
    # Process each model
    for model_name, model_data in bayesian_results['models'].items():
        print(f"\n{'='*80}")
        print(f"Validating: {model_name}")
        print(f"{'='*80}")
        
        try:
            model = model_data 
            
            include_distance = model.feature_mode == 'full'
            include_width = model.feature_mode in ['full', 'width_only']
            
            # Extract features using model's own method
            X, y, sensor_features, prior_estimates = model._extract_features(
                data_manager, include_distance, include_width
            )
            
            # Use same train/test split as training
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.1, random_state=42
            )
            
            # Get physics priors for test set
            mu_phys_all = prior_estimates.get(model.prior_type, prior_estimates.get('weighted', np.zeros_like(y)))
            
            # Compute sigma_phys_const (same way as training)
            try:
                sigma_const = constant_sigma_phys_from_rmse(mu_phys_all, y, floor=1e-3)
            except:
                sigma_const = max(float(np.sqrt(np.mean((mu_phys_all - y)**2))), 1e-3)
            
            sigma_phys_all = np.full_like(y, sigma_const, dtype=float)
            
            # Split priors same way
            _, mu_phys_te = train_test_split(mu_phys_all, test_size=0.1, random_state=42)
            _, sp_te      = train_test_split(sigma_phys_all, test_size=0.1, random_state=42)
            
            # Compute closed-form posterior
            y_closed_form     = np.zeros(len(X_test))
            y_closed_form_std = np.zeros(len(X_test))
            
            # Extract learned parameters
            if model.inference == 'mcmc':
                # For MCMC, use posterior mean of samples
                mcmc_samples = model.model._mcmc.get_samples()
                w_loc = mcmc_samples['w'].mean(0).cpu().numpy()
                b_loc = mcmc_samples['b'].mean(0).cpu().numpy()
                if b_loc.ndim > 0:
                    b_loc = b_loc[0]
                if 'tau' in mcmc_samples:
                    tau = float(mcmc_samples['tau'].mean().cpu().numpy())
                    tau = abs(tau)
                else:
                    tau = float(model.obs_sigma)
            else:
                # For SVI, extract from guide
                post_means, post_scales = model._extract_posterior_params_from_guide()
                
                if 'weights' in post_means and 'bias' in post_means:
                    w_loc = post_means['weights'][0] if post_means['weights'].ndim > 1 else post_means['weights']
                    b_loc = post_means['bias'][0] if post_means['bias'].ndim > 0 else post_means['bias']
                    tau   = float(post_means.get('tau', model.obs_sigma))
                    tau   = abs(tau)
                else:
                    print(f"Warning: Could not extract parameters for {model_name}")
                    continue


            print(f"Using obs_scale (τ): {tau:.6f}")

            if hasattr(model.model, "_scaler") and model.model._scaler is not None:
                mu_x, std_x = model.model._scaler
                mu_x        = mu_x.cpu().numpy().reshape(-1)
                std_x       = std_x.cpu().numpy().reshape(-1)
                
                w_unstd = w_loc / std_x
                b_unstd = b_loc - np.dot(w_unstd, mu_x)
            else:
                w_unstd = w_loc
                b_unstd = b_loc

            # Initialize closed-form arrays
            y_closed_form = np.zeros(len(X_test))
            y_closed_form_std = np.zeros(len(X_test))

            # Compute closed-form for each test sample
            for i in range(len(X_test)):
                mu_lin     = np.dot(w_unstd, X_test[i]) + b_unstd
                mu_phys    = mu_phys_te[i]
                sigma_phys = sp_te[i]
                
                # Precisions (inverse variances)
                lambda_tau  = 1.0 / (tau ** 2)
                lambda_phys = 1.0 / (sigma_phys ** 2)
                
                # Precision-weighted fusion (posterior mean)
                y_closed_form[i] = (lambda_tau * mu_lin + lambda_phys * mu_phys) / (lambda_tau + lambda_phys)
                
                # Posterior standard deviation (CORRECTED: was missing sqrt!)
                y_closed_form_std[i] = np.sqrt(1.0 / (lambda_tau + lambda_phys))

            # Get model predictions
            if model.inference == 'mcmc':
                y_pred_mean, y_pred_std = model.model.predict_mcmc(X_test, mu_phys_te, sp_te)
            else:
                y_pred_mean, y_pred_std = model.model.predict(X_test, mu_phys_te, sp_te)
            
            # Convert to numpy if needed
            if isinstance(y_pred_mean, torch.Tensor):
                y_pred_mean = y_pred_mean.cpu().numpy()
                y_pred_std  = y_pred_std.cpu().numpy()

            # Diagnostic prints
            print(f"\nDiagnostic for {model_name}:")
            print(f"Tau (τ): {tau:.6f}")
            print(f"Mean sigma_phys: {sp_te.mean():.6f}")
            print(f"Sample X_test[0]: {X_test[0]}")
            print(f"Sample mu_lin[0]: {np.dot(w_unstd, X_test[0]) + b_unstd:.4f}")
            print(f"Sample mu_phys[0]: {mu_phys_te[0]:.4f}")
            print(f"Sample y_closed[0]: {y_closed_form[0]:.4f}")
            print(f"Sample y_pred[0]: {y_pred_mean[0]:.4f}")
            print(f"Mean predictive std: {y_pred_std.mean():.6f}")
            print(f"Mean closed-form std: {y_closed_form_std.mean():.6f}")
            print(f"Mean |predictive - closed|: {np.abs(y_pred_mean - y_closed_form).mean():.6f}°")

            # Compute metrics
            mae_predictive = mean_absolute_error(y_test, y_pred_mean)
            mae_closed = mean_absolute_error(y_test, y_closed_form)
            rmse_predictive = np.sqrt(mean_squared_error(y_test, y_pred_mean))
            rmse_closed = np.sqrt(mean_squared_error(y_test, y_closed_form))

            residuals = y_pred_mean - y_closed_form
            
            # Store results
            validation_results[model_name] = {
                'mae_predictive': mae_predictive,
                'mae_closed': mae_closed,
                'rmse_predictive': rmse_predictive,
                'rmse_closed': rmse_closed,
                'residual_mean': residuals.mean(),
                'residual_std': residuals.std(),
                'residual_max': np.abs(residuals).max()
            }
            
            # Create validation plots for this model
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Plot 1: Predictive vs Closed-Form
            ax = axes[0, 0]
            ax.scatter(y_closed_form, y_pred_mean, alpha=0.6, s=50)
            lims = [min(y_closed_form.min(), y_pred_mean.min()), 
                    max(y_closed_form.max(), y_pred_mean.max())]
            ax.plot(lims, lims, 'r--', lw=2, label='Perfect agreement')
            ax.set_xlabel(r'Closed-Form Posterior Mean ($^\circ$)')
            ax.set_ylabel(r'Predictive Mean ($^\circ$)')
            ax.set_title(f'{model_name}\nPosterior Predictive vs Closed-Form')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot 2: Residuals
            ax = axes[0, 1]
            ax.hist(residuals, bins=20, alpha=0.7, edgecolor='black')
            ax.axvline(0, color='r', linestyle='--', lw=2, label='Zero residual')
            ax.set_xlabel(r'Residual: Predictive - Closed-Form ($^\circ$)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Residuals\n(mean={residuals.mean():.4f}, std={residuals.std():.4f})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot 3: Predictions vs Ground Truth
            ax = axes[1, 0]
            ax.scatter(y_test, y_pred_mean, alpha=0.5, s=50, label='Predictive')
            ax.scatter(y_test, y_closed_form, alpha=0.5, s=50, label='Closed-Form')
            lims = [min(y_test.min(), y_pred_mean.min(), y_closed_form.min()),
                    max(y_test.max(), y_pred_mean.max(), y_closed_form.max())]
            ax.plot(lims, lims, 'k--', alpha=0.5, lw=2)
            ax.set_xlabel(r'True Angle ($^\circ$)')
            ax.set_ylabel(r'Predicted Angle ($^\circ$)')
            ax.set_title('Predictions vs Ground Truth')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot 4: Error comparison
            ax = axes[1, 1]
            errors_pred = np.abs(y_pred_mean - y_test)
            errors_closed = np.abs(y_closed_form - y_test)
            ax.scatter(errors_closed, errors_pred, alpha=0.6, s=50)
            max_err = max(errors_closed.max(), errors_pred.max())
            ax.plot([0, max_err], [0, max_err], 'r--', lw=2, label='Equal error')
            ax.set_xlabel(r'Closed-Form MAE ($^\circ$)')
            ax.set_ylabel(r'Predictive MAE ($^\circ$)')
            ax.set_title('Error Comparison per Sample')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(val_dir, f"{model_name}_validation.png"), 
                        dpi=300, bbox_inches='tight')
            plt.close()

            # Create standalone plot
            fig_pred = plt.figure(figsize=(8, 6))
            ax_pred = fig_pred.add_subplot(111)

            ax_pred.scatter(y_test, y_pred_mean, alpha=0.7, s=80, marker='o', 
                            edgecolors='#0173B2', facecolors='none', linewidths=2.5, 
                            label='Posterior Predictive', zorder=3)
            ax_pred.scatter(y_test, y_closed_form, alpha=0.7, s=60, marker='s', 
                            color='#DE8F05', label='Closed-Form Posterior', zorder=2)

            # Perfect prediction line
            lims = [min(y_test.min(), y_pred_mean.min(), y_closed_form.min()) - 0.5,
                    max(y_test.max(), y_pred_mean.max(), y_closed_form.max()) + 0.5]
            ax_pred.plot(lims, lims, 'k--', alpha=0.6, lw=2.5, label='Perfect Prediction', zorder=1)

            # Styling
            ax_pred.set_xlabel(r'True Angle ($^\circ$)', fontsize=20)
            ax_pred.set_ylabel(r'Predicted Angle ($^\circ$)', fontsize=20)
            ax_pred.set_title(r'Comparison: Predictions vs Ground Truth', fontsize=24, pad=20)
            ax_pred.legend(fontsize=18, framealpha=0.9, loc='best')
            ax_pred.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

            # Add error statistics as text box
            mae_pred_text = f"Predictive MAE: {0.3810}°"
            mae_closed_text = f"Closed-Form MAE: {mae_closed:.4f}°"
            residual_text = f"Mean Residual: {residuals.mean():.4f}° (±{residuals.std():.4f}°)"
            textstr = f'{mae_pred_text}\n{mae_closed_text}\n{residual_text}'
            props = dict(boxstyle='round', facecolor='white', edgecolor='gray', 
                        alpha=0.0, linewidth=1.5)
            ax_pred.text(0.05, 0.95, textstr, transform=ax_pred.transAxes, fontsize=18,
                        verticalalignment='top', bbox=props)

            plt.tight_layout()
            plt.savefig(os.path.join(val_dir, f"{model_name}_predictions_comparison.png"), 
                        dpi=600, bbox_inches='tight')
            plt.close(fig_pred)
            
            print(f"✓ Predictive MAE: {mae_predictive:.4f}° | Closed-Form MAE: {mae_closed:.4f}°")
            print(f"✓ Residual: mean={residuals.mean():.4f}° (std={residuals.std():.4f}°)")
            
        except Exception as e:
            print(f"Error validating {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create summary comparison plot
    if validation_results:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        model_names = list(validation_results.keys())
        mae_pred = [validation_results[m]['mae_predictive'] for m in model_names]
        mae_closed = [validation_results[m]['mae_closed'] for m in model_names]
        residual_means = [validation_results[m]['residual_mean'] for m in model_names]
        
        # Plot 1: MAE comparison
        ax = axes[0]
        x = np.arange(len(model_names))
        width = 0.35
        ax.bar(x - width/2, mae_pred, width, label='Predictive', alpha=0.8)
        ax.bar(x + width/2, mae_closed, width, label='Closed-Form', alpha=0.8)
        ax.set_ylabel('MAE (degrees)')
        ax.set_title('Predictive vs Closed-Form MAE')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Residuals
        ax = axes[1]
        ax.bar(model_names, residual_means, alpha=0.8)
        ax.axhline(0, color='r', linestyle='--', lw=2)
        ax.set_ylabel('Mean Residual (degrees)')
        ax.set_title('Predictive - Closed-Form Bias')
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(val_dir, "all_models_validation_summary.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save CSV summary
        import pandas as pd
        df = pd.DataFrame(validation_results).T
        df.to_csv(os.path.join(val_dir, "validation_summary.csv"))
    
    print(f"\n{'='*80}")
    print(f"All validation results saved to: {val_dir}")
    print(f"{'='*80}\n")
    
    return validation_results

def extract_timing_comparison(bayesian_results, output_dir):
    """
    Extract and visualize timing comparison from already-trained models.
    
    Parameters:
        - bayesian_results [dict] : Dictionary containing results from trained models
        - output_dir       [str]  : Directory to save results
        
    Returns:
        - dict: Dictionary with timing results and speedup factors
    """
    print("\n" + "="*80)
    print("MCMC vs SVI Timing Comparison (from trained models)")
    print("="*80)
    
    # Create output directory
    timing_dir = os.path.join(output_dir, "timing_comparison")
    os.makedirs(timing_dir, exist_ok=True)
    
    # Organize results by inference method
    svi_results = {}
    mcmc_results = {}
    
    for model_name, summary in bayesian_results['results'].items():        
        if model_name.endswith('_svi'):
            inference = 'svi'
            config_name = model_name[:-4]
        elif model_name.endswith('_mcmc'):
            inference = 'mcmc'
            config_name = model_name[:-5]
        else:
            inference = summary.get('inference', None)
            config_name = model_name
            
            if inference is None:
                model = bayesian_results['models'].get(model_name)
                if model and hasattr(model, 'inference'):
                    inference = model.inference
        
        # Store based on inference method
        if inference == 'svi':
            svi_results[config_name] = summary
        elif inference == 'mcmc':
            mcmc_results[config_name] = summary
    
    # Find matching configurations
    common_configs = set(svi_results.keys()) & set(mcmc_results.keys())
    
    if not common_configs:
        print("Warning: No matching SVI/MCMC pairs found!")
        print(f"SVI configs: {list(svi_results.keys())}")
        print(f"MCMC configs: {list(mcmc_results.keys())}")
        return None
    
    # Extract timing data
    configs    = sorted(list(common_configs))
    svi_times  = []
    mcmc_times = []
    speedups   = []
    svi_mae    = []
    mcmc_mae   = []
    
    for config in configs:
        svi_time  = svi_results[config].get('elapsed_time', None)
        mcmc_time = mcmc_results[config].get('elapsed_time', None)
        
        if svi_time is None or mcmc_time is None:
            print(f"Warning: Missing timing data for {config}")
            continue
        
        svi_times.append(svi_time)
        mcmc_times.append(mcmc_time)
        speedups.append(mcmc_time / svi_time)
        svi_mae.append(svi_results[config]['mae'])
        mcmc_mae.append(mcmc_results[config]['mae'])
    
    if not svi_times:
        print("Warning: No valid timing data found!")
        return None
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Computation time comparison
    ax = axes[0, 0]
    x  = np.arange(len(configs))
    width = 0.35
    ax.bar(x - width/2, svi_times, width, label='SVI')
    ax.bar(x + width/2, mcmc_times, width, label='MCMC')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Computation Time: SVI vs MCMC')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Speedup factors
    ax = axes[0, 1]
    ax.bar(configs, speedups)
    ax.axhline(y=1, color='r', linestyle='--', label='No speedup')
    ax.set_ylabel('Speedup Factor (MCMC time / SVI time)')
    ax.set_title('SVI Speedup over MCMC')
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: MAE comparison
    ax = axes[1, 0]
    ax.bar(x - width/2, svi_mae, width, label='SVI')
    ax.bar(x + width/2, mcmc_mae, width, label='MCMC')
    ax.set_ylabel('MAE (degrees)')
    ax.set_title('Accuracy Comparison: SVI vs MCMC')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Efficiency scatter (time vs MAE)
    ax = axes[1, 1]
    ax.scatter(svi_times, svi_mae, s=100, alpha=0.6, label='SVI')
    ax.scatter(mcmc_times, mcmc_mae, s=100, alpha=0.6, label='MCMC')
    ax.set_xlabel('Computation Time (seconds)')
    ax.set_ylabel('MAE (degrees)')
    ax.set_title('Efficiency: Time vs Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(timing_dir, "timing_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed results to CSV
    timing_df = pd.DataFrame({
        'Configuration': configs,
        'SVI_Time': svi_times,
        'MCMC_Time': mcmc_times,
        'Speedup': speedups,
        'SVI_MAE': svi_mae,
        'MCMC_MAE': mcmc_mae
    })
    timing_df.to_csv(os.path.join(timing_dir, "timing_results.csv"), index=False)
    
    # Print summary
    print("\n" + "="*80)
    print("Timing Summary")
    print("="*80)
    print(f"Average SVI time:  {np.mean(svi_times):.2f}s")
    print(f"Average MCMC time: {np.mean(mcmc_times):.2f}s")
    print(f"Average speedup:   {np.mean(speedups):.2f}x")
    print(f"Max speedup:       {np.max(speedups):.2f}x ({configs[np.argmax(speedups)]})")
    print(f"Min speedup:       {np.min(speedups):.2f}x ({configs[np.argmin(speedups)]})")
    print(f"\nResults saved to: {timing_dir}")
    
    return {
        'configs': configs,
        'svi_times': svi_times,
        'mcmc_times': mcmc_times,
        'speedups': speedups,
        'svi_mae': svi_mae,
        'mcmc_mae': mcmc_mae
    }
# =================================================================================================================================== #


# =================================================================================================================================== #
# ---------------------------------------------------------- MAIN FUNCTION ---------------------------------------------------------- #
RESULTS_PICKLE  = os.path.join(RESULTS_BASE_DIR, 'pipeline_results.pkl')
BAYESIAN_PICKLE = os.path.join(RESULTS_BASE_DIR, 'bayesian_results.pkl')
def main(run_import=False, run_classical=True, run_bayesian=False):
    """
    Main function to execute the AoA estimation pipeline.
    This function orchestrates the entire process of importing data, performing AoA analysis,
    generating visualizations, and training Bayesian regression models. 

    Parameters:
        - run_import    [bool] : Whether to import data from CSV files (True) or load existing DataManager (False).
        - run_classical [bool] : Whether to run classical AoA analysis (True) or skip it (False).
        - run_bayesian  [bool] : Whether to train Bayesian regression models (True) or load existing results (False).

    Returns:
        - None
    """
    # STEP 1: Create a DataManager instance - RFID_data_manager
    if os.path.exists(RESULTS_PICKLE) and not run_import:
        print(f"Loading existing DataManager from {RESULTS_PICKLE}...")
        with open(RESULTS_PICKLE, 'rb') as f:
            RFID_data_manager = pickle.load(f)
    else:
        print("Creating new DataManager and importing data...")
        RFID_data_manager = DataManager(data_dir=DATA_DIRECTORY, tag_id=TAG_ID, aoa_range=AoA_m)
        # STEP 2: Import data
        RFID_data_manager.import_data()
        # Save DataManager state
        with open(RESULTS_PICKLE, 'wb') as f:
            pickle.dump(RFID_data_manager, f)

    # STEP 3: Run Classical AoA analysis
    if run_classical:
        print("Starting individual file analysis...")
        results = RFID_data_manager.analyze_all_data(save_results=SAVE_RESULTS)
        # STEP 4: Generate summary visualizations for individual analysis
        if SAVE_RESULTS and len(results) > 0:
            # Create summary plots directory
            summary_dir = os.path.join(RESULTS_DIRECTORY, "summary")
            os.makedirs(summary_dir, exist_ok=True)
            # Plot 1: Error distribution by method
            plt.figure(figsize=(10, 6))
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')
            methods = ['phase', 'ds', 'weighted', 'music']
            method_names = ['Phase', 'Beamforming', 'Weighted BF', 'MUSIC']
            # Create boxplot of errors
            error_data = [results[f'error_{m}'] for m in methods]
            plt.boxplot(error_data, labels=method_names)
            plt.ylabel('Error (degrees)')
            plt.title('AoA Estimation Error by Method')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(summary_dir, "error_comparison.png"), dpi=300)
            plt.close()
            # Plot 2: Error vs. Position (scatter plot)
            plt.figure(figsize=(12, 8))
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
        dashboard_results = vis.create_dashboard()
    else:
        print("Skipping individual file analysis...")

    # STEP 6: Train machine learning model
    if os.path.exists(BAYESIAN_PICKLE) and not run_bayesian:
        print(f"Loading existing Bayesian results from {BAYESIAN_PICKLE}...")
        with open(BAYESIAN_PICKLE, 'rb') as f:
            bayesian_results = pickle.load(f)
    else:
        print("Starting Bayesian regression analysis...")
        full = br.train_bayesian_models(RFID_data_manager, RESULTS_DIRECTORY, num_epochs=14000)
        bayesian_results = {
            "results": {name: entry["summary"] for name, entry in full["results"].items()},
            "models":  {name: entry["model"]   for name, entry in full["results"].items()},
            "best_name": full["best_name"]
        }
        with open(BAYESIAN_PICKLE, 'wb') as f:
            pickle.dump(bayesian_results, f)
    
    # STEP 6.5: Extract timing information from already-trained models
    print("\nExtracting MCMC vs SVI computation time comparison...")
    extract_timing_comparison(bayesian_results, RESULTS_DIRECTORY)

    # STEP 7: Compare Bayesian Models
    if 'results' in bayesian_results:
      print("\nGenerating Bayesian model comparison plots...")
  
      def pretty(prior_key, fmode_key):
          prior_map = {"ds":"DS","weighted":"WDS","music":"MUSIC","phase":"PD"}
          fmode_map = {"full":"Full","width_only":"Width","sensor_only":"Sensor"}
          return f"{prior_map.get(prior_key, prior_key)} – {fmode_map.get(fmode_key, fmode_key)}"
  
      pretty_summaries = {}
      for raw_name, entry in bayesian_results["results"].items():
          # raw_name like "ds_full"
          if "_" in raw_name:
              prior, fmode = raw_name.split("_", 1)
          else:
              prior, fmode = raw_name, ""
          label = pretty(prior, fmode)
          pretty_summaries[label] = entry
  
      br.compare_bayesian_models(pretty_summaries, RESULTS_DIRECTORY, EXPERIMENT_NAME)
      print(f"Saved Bayesian model comparison plots to: {RESULTS_DIRECTORY}")
      try:
          # Figure 1: side-by-side MAE/RMSE bars
          br.figure1_model_comparison(pretty_summaries, RESULTS_DIRECTORY, EXPERIMENT_NAME)
          # Figure 2: lines + magnified posterior predictive
          best_label   = min(pretty_summaries, key=lambda k: pretty_summaries[k].get('rmse', float('inf')))
          best_summary = pretty_summaries[best_label]
          br.figure2_scatter_and_posterior(best_summary, RESULTS_DIRECTORY, EXPERIMENT_NAME, magnify=0)
          print("Saved ICASSP figures to:", os.path.join(RESULTS_DIRECTORY, "bayesian_model", EXPERIMENT_NAME))
      except Exception as e:
          print("[WARN] Failed to generate ICASSP figures:", e)

    # STEP 8: Create detailed visualizations for the best model
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
        #best_model.render_model_and_guide(detailed_dir, "best_model")
        best_model.plot_posterior_predictive(detailed_dir, "best_model")
        best_model.plot_uncertainty_calibration(detailed_dir, "best_model")
        best_model.visualize_weight_distributions(detailed_dir, "best_model")
        
        # STEP 8.5: Validate against closed-form posterior
        #print("\nValidating against closed-form posterior...")
        #best_model.validate_closed_form_posterior(detailed_dir, "best_model")
        
        print(f"Best model: {best_model_name} (MAE: {best_mae:.4f}°)")
        print(f"Detailed visualizations saved to: {detailed_dir}")
    
    print("AoA analysis completed successfully!")
    return RFID_data_manager, bayesian_results
# =================================================================================================================================== #


if __name__ == "__main__":
    # Initialize data_manager to None in case main() fails
    data_manager = None
    bayesian_results = None
    
    # 1) Run core main (optional components disabled; experiments handle training)
    try:
        data_manager, bayesian_results = main(run_import=False, run_classical=False, run_bayesian=False)
        print("Base main() execution completed successfully")
    except Exception as e:
        print("Note: base main() execution failed:", e)
        import traceback
        traceback.print_exc()
        
        # Try to load data_manager from pickle if main() failed
        if os.path.exists(RESULTS_PICKLE):
            print(f"\nAttempting to load DataManager from {RESULTS_PICKLE}...")
            try:
                with open(RESULTS_PICKLE, 'rb') as f:
                    data_manager = pickle.load(f)
                print("✓ Successfully loaded DataManager from pickle")
            except Exception as e2:
                print(f"✗ Failed to load DataManager: {e2}")
        
        # Try to load bayesian_results if validation is needed
        if os.path.exists(BAYESIAN_PICKLE):
            print(f"\nAttempting to load Bayesian results from {BAYESIAN_PICKLE}...")
            try:
                with open(BAYESIAN_PICKLE, 'rb') as f:
                    bayesian_results = pickle.load(f)
                print("✓ Successfully loaded Bayesian results from pickle")
            except Exception as e3:
                print(f"✗ Failed to load Bayesian results: {e3}")

    # VALIDATE ALL MODELS FROM PICKLE (only if we have the required data)
    if data_manager is not None and bayesian_results is not None:
        print("\n" + "="*80)
        print("VALIDATING ALL MODELS AGAINST CLOSED-FORM POSTERIOR")
        print("="*80)
        try:
            validation_results = validate_all_models_from_pickle(
                BAYESIAN_PICKLE, 
                data_manager, 
                RESULTS_DIRECTORY)
        except Exception as e:
            print(f"Validation failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n" + "="*80)
        print("SKIPPING VALIDATION: Missing required data")
        if data_manager is None:
            print("  - data_manager not available")
        if bayesian_results is None:
            print("  - bayesian_results not available")
        print("="*80)

    # # 2) Train the 24-model grid (4 priors × 3 feature sets × 2 inference schemes)
    # try:
    #     grid_df = run_model_grid()
    #     print("Grid summary saved. Rows:", len(grid_df))
    # except Exception as e:
    #     print("Grid run failed:", e)

    # # 3) Run the small-sample regime sweeps
    # try:
    #     #sweeps = run_small_sample_sweeps()
    #     # Debug just SVI
    #     #sweeps_svi = run_small_sample_sweeps(inference_schemes=('svi',))
    #     #visualize_small_sample_results(sweeps_svi, RESULTS_DIRECTORY)
    #     #print("SVI: Small-sample sweeps complete.")

    #     # Later when ready, debug just MCMC
    #     sweeps_mcmc = run_small_sample_sweeps(inference_schemes=('mcmc',))
    #     print("Small-sample sweeps complete.")
    #     #visualize_small_sample_results(sweeps_mcmc, RESULTS_DIRECTORY)
    #     print("MCMC: Small-sample sweeps complete.")
    # except Exception as e:
    #     print("Sweep run failed:", e)

    print("All experiments complete!")
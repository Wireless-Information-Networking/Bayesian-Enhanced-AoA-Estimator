# =================================================================================================================================== #
# ----------------------------------------------------------- DESCRIPTION ----------------------------------------------------------- #
# This script provides the main functionality for analyzing Angle of Arrival (AoA) data from RFID systems. It imports data,           #
# performs AoA estimation using various methods, and visualizes the results. The script also includes machine learning components     #
# for regression analysis on AoA data.                                                                                                #  
# =================================================================================================================================== #


# =================================================================================================================================== #
# --------------------------------------------------------- EXTERNAL IMPORTS -------------------------------------------------------- #
import os                                            # Operating system interfaces for file and directory manipulation.               #
import glob                                          # Unix style pathname pattern expansion for file searching.                      #
import re                                            # Regular expression operations for pattern matching in filenames.               #
import numpy                   as np                 # Mathematical functions.                                                        #
import pandas                  as pd                 # Data manipulation and analysis library.                                        #
import scipy.constants         as sc                 # Physical and mathematical constants.                                           #
import matplotlib.pyplot       as plt                # Data visualization.                                                            #
import src.data_management     as dm                 # Data management functions for importing and organizing data.                   #
import src.phase_difference    as pad                # Phase difference calculations for AoA estimation.                              #  
import src.beamforming         as bf                 # Beamforming methods for AoA estimation.                                        #
import src.music               as music              # MUSIC algorithm for high-resolution AoA estimation.                            #
import src.bayesian_regression as br                 # Bayesian regression for machine learning models on AoA data.                   #
import src.visualization       as vis                # Visualization functions for AoA analysis results.                              #
from   tqdm                    import tqdm           # Progress bar for loops, useful for tracking long-running operations.           #
# =================================================================================================================================== #


# =================================================================================================================================== #
# ------------------------------------------------------- CONFIGURATION SETTINGS ---------------------------------------------------- #
SCRIPT_DIR         = os.path.dirname(os.path.abspath(__file__))                    # Get the directory of the current script.         #
# PROJECT_ROOT     = os.path.dirname(SCRIPT_DIR)                                   # Go up one level to project root.                 #
PROJECT_ROOT       = SCRIPT_DIR                                                    # Use the script directory as the project root.    #
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
# ------------------------------------------------------- AoA & DATA MANAGEMENT ----------------------------------------------------- #
def analyze_aoa(phasor1, phasor2, rssi1, rssi2, L, wavelength, aoa_scan, true_angle=None):
    """
    Comprehensive AoA estimation using multiple methods: phase difference,
    standard beamforming, RSSI-weighted beamforming, and MUSIC algorithm.
    
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
        - dict: Dictionary with AoA estimates and spectra for all methods
    """
    # 1. Phase difference method
    dphi     = pad.compute_phase_difference(phasor1, phasor2)
    theta_ph = pad.phase_difference_aoa(dphi, L, wavelength)
    # 2. Beamforming methods
    B_ds, B_w, theta_ds, theta_w = bf.beamforming_spectrum_calculation(
        phasor1, phasor2, rssi1, rssi2, L, wavelength, aoa_scan)
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
        dphi = np.angle(np.mean(phasor1)) - np.angle(np.mean(phasor2))
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
# ---------------------------------------------------------- MAIN FUNCTION ---------------------------------------------------------- #
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
    dashboard_results = vis.create_dashboard()
    # STEP 6: Train machine learning model
    bayesian_results = br.train_bayesian_models(RFID_data_manager, RESULTS_DIRECTORY, num_epochs=14000)
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
# =================================================================================================================================== #
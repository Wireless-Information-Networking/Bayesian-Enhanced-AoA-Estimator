# =================================================================================================================================== #
# --------------------------------------------------------- EXTERNAL IMPORTS -------------------------------------------------------- #
import os                                # Operating system interfaces for file and directory manipulation.                           #
import re                                # Regular expression operations for pattern matching in filenames.                           #
import pandas as pd                      # Data manipulation and analysis.                                                            #
import matplotlib.pyplot as plt          # Data visualization.                                                                        #
import seaborn as sns                    # Statistical data visualization based on matplotlib.                                        #
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting.                                                                               #
import scipy.constants as sc             # Physical and mathematical constants.                                                       #
import numpy as np                       # Mathematical functions.                                                                    #
import src.data_management as dm         # Data management functions.                                                                 #
import datetime                          # Date and time manipulation.                                                                #
from scipy.stats import norm             # Statistical functions for normal distribution fitting.                                     #
# =================================================================================================================================== #


# =================================================================================================================================== #
# ---------------------------------------------------------- PLOTTING SETTINGS ------------------------------------------------------ #
plt.style.use(["science", "no-latex"])                                                                                                #
plt.rcParams.update({                                                                                                                 #
    "font.size": 8,               # Base font size for all text in the plot.                                                          #
    "axes.titlesize": 9,          # Title size for axes.                                                                              #
    "axes.labelsize": 9,          # Axis labels size.                                                                                 #
    "xtick.labelsize": 8,         # X-axis tick labels size.                                                                          #
    "ytick.labelsize": 8,         # Y-axis tick labels size.                                                                          #
    "legend.fontsize": 12,        # Legend font size for all text in the legend.                                                      #
    "figure.titlesize": 9,        # Overall figure title size for all text in the figure.                                             #
})                                                                                                                                    #
sns.set_theme(style="whitegrid", context="paper") # Set seaborn theme for additional aesthetics and context.                          #                                                                             #
plt.rcParams["figure.figsize"] = (6, 4)  # Set default figure size for all plots to 6x4.                                              #
TAG_NAME = "Belt DEE"             # Default tag name for the analysis.                                                                #
# =================================================================================================================================== #


# =================================================================================================================================== #
# ---------------------------------------------------------- HELPER FUNCTIONS ------------------------------------------------------- #
def MHz_to_Hz(frequency: float) -> float:
    """
    Convert the frequency from MHz to Hz.
    
    Parameters:
        frequency (float): The frequency to convert from MHz to Hz.
    
    Returns:
        The frequency in Hz. [float]
    """
    return frequency * 1e6

def dB_to_dBm(power: float) -> float:
    """
    Convert the power from dB to dBm.
    
    Parameters:
        power (float): The power to convert from dB to dBm.
    
    Returns:
        The power in dBm. [float]
    """
    return power + 30

def dBm_to_dB(power: float) -> float:
    """
    Convert the power from dBm to dB.
    
    Parameters:
        power (float): The power to convert from dBm to dB.
    
    Returns:
        The power in dB. [float]
    """
    return power - 30

def dBm_to_W(power: float) -> float:
    """
    Convert the power from dBm to W.
    
    Parameters:
        power (float): The power to convert from dBm to W.
    
    Returns:
        The power in W. [float]
    """
    return 10 ** (power / 10) / 1000

def get_lambda(frequency: float) -> float:
    """
    Calculate the wavelength (lambda) from the frequency.
    
    Parameters:
        frequency (float): The frequency to calculate the wavelength from [Hz].
    
    Returns:
        The wavelength (lambda) from the frequency. [float]
    """
    return sc.speed_of_light / frequency
# =================================================================================================================================== #


# =================================================================================================================================== #
# ----------------------------------------------------------- DATA FUNCTIONS -------------------------------------------------------- #
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
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Created output directory: {output_dir}")
    return output_dir

def process_file(file_path, data_list):
    """Process a single file and add all its data points to the appropriate list"""
    try:
        # Extract metadata from filename
        filename = os.path.basename(file_path)
        
        # Parse filename: YYYYMMDD_FFF.F_D.DDD_L.LLL_W.WWW.csv
        pattern = r'(\d{8})_(\d+\.\d+)_(\d+\.\d+)_(\d+\.\d+)_([+-]?\d+\.\d+)\.csv'
        match = re.match(pattern, filename)
        
        if not match:
            print(f"Warning: Filename {filename} doesn't match expected pattern.")
            return
            
        date_str, freq_str, dist_str, spacing_str, width_str = match.groups()
        
        # Convert to appropriate types
        frequency = float(freq_str) * 1e6  # Convert MHz to Hz
        distance = float(dist_str)         # Vertical distance in meters
        antenna_spacing = float(spacing_str) # Inter-antenna spacing in meters
        width = float(width_str)           # Horizontal offset in meters
        power = 29.2  # Default power if not in filename
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Check for required columns
        required_cols = ['peakRssi', 'phase', 'channel', 'idHex', 'antenna']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            print(f"Warning: Missing columns {missing} in {filename}")
            return
        
        # For each row in the CSV, create a data point
        for _, row in df.iterrows():
            # Create a row with metadata and measurements
            row_data = {
                'EPC': row['idHex'].strip(),
                'rssi': row['peakRssi'],
                'phase': row['phase'],
                'frequency': row['channel'],
                'frequencyHz': frequency,
                'frequencyMHz': float(freq_str),
                'antenna': row['antenna'],
                'distance': distance,
                'antenna_spacing': antenna_spacing,
                'width': width,
                'power_dbm': power,
                'filename': filename,
                'date': date_str
            }
            
            data_list.append(row_data)
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def load_aoa_data(base_dir, tag_id=None):
    """
    Load and organize AoA measurement data.
    
    Args:
        base_dir: Base directory containing experiment data
        tag_id: Optional tag ID to filter by
        
    Returns:
        Dictionary with structure: {distance: {replica: {frequency: dataframe}}}
    """
    experiment_data = {}
    
    # Get distance directories
    distance_dirs = sorted([d for d in os.listdir(base_dir) 
                           if os.path.isdir(os.path.join(base_dir, d))])
    
    for dist_dir in distance_dirs:
        distance_path = os.path.join(base_dir, dist_dir)
        experiment_data[dist_dir] = {}
        
        # Get replica directories
        replica_dirs = sorted([d for d in os.listdir(distance_path) 
                              if os.path.isdir(os.path.join(distance_path, d))])
        
        for replica_dir in replica_dirs:
            replica_path = os.path.join(distance_path, replica_dir)
            experiment_data[dist_dir][replica_dir] = {}
            
            # Get frequency directories
            freq_dirs = sorted([d for d in os.listdir(replica_path) 
                               if os.path.isdir(os.path.join(replica_path, d))])
            
            for freq_dir in freq_dirs:
                freq_path = os.path.join(replica_path, freq_dir)
                
                # Get all CSV files
                csv_files = [f for f in os.listdir(freq_path) if f.endswith('.csv')]
                
                # Process files
                data_list = []
                for csv_file in csv_files:
                    file_path = os.path.join(freq_path, csv_file)
                    process_file(file_path, data_list)
                
                if data_list:
                    # Create dataframe
                    df = pd.DataFrame(data_list)
                    
                    # Filter by tag ID if specified
                    if tag_id and not df.empty:
                        df = df[df['EPC'] == tag_id]
                    
                    if not df.empty:
                        experiment_data[dist_dir][replica_dir][freq_dir] = df
                    else:
                        print(f"No valid data for {dist_dir}/{replica_dir}/{freq_dir}")
    
    return experiment_data

def summarize_aoa_data(data):
    """Print a summary of all dataframes in the AoA data structure."""
    total_samples = 0
    summary_rows = []
    
    print(f"{'Distance':<10} {'Replica':<10} {'Frequency':<10} {'Rows':<8} {'EPC Tags':<15} {'Antennas':<10}")
    print("-" * 70)
    
    for distance in data:
        for replica in data[distance]:
            for frequency in data[distance][replica]:
                df = data[distance][replica][frequency]
                rows = df.shape[0]
                total_samples += rows
                tags = len(df['EPC'].unique()) if 'EPC' in df.columns else 'N/A'
                antennas = df['antenna'].unique() if 'antenna' in df.columns else 'N/A'
                
                print(f"{distance:<10} {replica:<10} {frequency:<10} {rows:<8} {tags:<15} {antennas}")
                
                summary_rows.append({
                    'Distance': distance,
                    'Replica': replica,
                    'Frequency': frequency,
                    'Rows': rows,
                    'Tags': tags,
                    'Antennas': str(antennas),
                    'DataFrame': df  # Store reference to actual dataframe
                })
    
    print("-" * 70)
    print(f"Total Dataframes: {len(summary_rows)}")
    print(f"Total Samples: {total_samples}")
    
    # Return a dataframe with the summary info for further analysis
    return pd.DataFrame(summary_rows)
# =================================================================================================================================== #


# =================================================================================================================================== #
# ------------------------------------------------------------ AoA ANALYSIS --------------------------------------------------------- #
def analyze_phase_distribution(df, bins=36):
    """Analyze phase distribution and fit with a Gaussian."""
    from scipy.optimize import curve_fit
    
    # Ensure phase values are in range [0, 360]
    phases = df['phase'].values
    phases = (phases + 360) % 360
    
    # Create histogram
    hist_data, bin_edges = np.histogram(phases, bins=bins, range=(0, 360))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Gaussian function for fitting
    def gaussian(x, amplitude, mean, stddev):
        return amplitude * np.exp(-((x - mean) % 360) ** 2 / (2 * stddev ** 2))
    
    # Fit Gaussian
    try:
        params, _ = curve_fit(gaussian, bin_centers, hist_data, 
                            p0=[hist_data.max(), bin_centers[hist_data.argmax()], 30])
    except:
        params = [0, 0, 0]  # Default if fitting fails
    
    return hist_data, bin_edges, params

def calculate_aoa(df, antenna_spacing):
    """Calculate Angle of Arrival from phase measurements."""
    # Group by antenna
    ant1_data = df[df['antenna'] == 1]
    ant2_data = df[df['antenna'] == 2]
    
    if len(ant1_data) == 0 or len(ant2_data) == 0:
        print("Warning: Missing data for one or both antennas")
        return None
    
    # Calculate phase difference
    phase1 = dm.circular_mean_deg(ant1_data['phase'])
    phase2 = dm.circular_mean_deg(ant2_data['phase'])
    phase_diff = phase1 - phase2
    
    # Get parameters
    frequency = df['frequencyHz'].iloc[0]
    width = df['width'].iloc[0]
    distance = df['distance'].iloc[0]
    
    # Calculate wavelength
    wavelength = sc.c / frequency
    
    # Calculate actual AoA from geometry
    actual_aoa = np.degrees(np.arctan2(width, distance))
    
    # Calculate estimated AoA from phase difference
    estimated_aoa = np.degrees(np.arcsin((wavelength / (2 * np.pi * antenna_spacing)) * 
                                         np.deg2rad(phase_diff)))
    
    return {
        'actual_aoa': actual_aoa,
        'phase_diff': phase_diff,
        'estimated_aoa': estimated_aoa,
        'error': estimated_aoa - actual_aoa,
        'wavelength': wavelength,
        'frequency': frequency
    }

def generate_summary(results, base_dir, tag_id):
    """
    Generate comprehensive summary of all results.
    
    Args:
        results: Nested dictionary with all results
        base_dir: Base directory of the experiment
        tag_id: Tag ID used for filtering (if any)
    """
    # Create summary tables
    summary_data = []
    
    # Process each distance
    for distance, replicas in results.items():
        for replica, frequencies in replicas.items():
            for freq_name, freq_results in frequencies.items():
                # Extract frequency in MHz
                freq_mhz = freq_results['frequency'] / 1e6
                
                # Collect data for this measurement
                row = {
                    'Distance': distance,
                    'Replica': replica,
                    'Frequency (MHz)': freq_mhz,
                    'Actual AoA (°)': freq_results['actual_aoa'],
                    'Estimated AoA (°)': freq_results['estimated_aoa'],
                    'Error (°)': freq_results['error'],
                    'Phase Diff (°)': freq_results['phase_diff'],
                    'Wavelength (m)': freq_results['wavelength']
                }
                
                summary_data.append(row)
    
    # Create dataframe
    summary_df = pd.DataFrame(summary_data)
    
    # Sort by distance, replica, and frequency
    summary_df = summary_df.sort_values(['Distance', 'Replica', 'Frequency (MHz)'])
    
    # Calculate aggregate statistics
    agg_stats = summary_df.groupby(['Distance', 'Frequency (MHz)']).agg({
        'Error (°)': ['mean', 'std', 'min', 'max'],
        'Estimated AoA (°)': ['mean', 'std'],
        'Phase Diff (°)': ['mean', 'std']
    }).reset_index()
    
    # Calculate RMSE by distance and frequency
    rmse_data = []
    for (dist, freq), group in summary_df.groupby(['Distance', 'Frequency (MHz)']):
        rmse = np.sqrt(np.mean(group['Error (°)']**2))
        rmse_data.append({
            'Distance': dist,
            'Frequency (MHz)': freq,
            'RMSE (°)': rmse
        })
    
    rmse_df = pd.DataFrame(rmse_data)
    
    # Create formatted output
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create output file
    output_file = os.path.join(os.path.dirname(base_dir), 
                              f"aoa_summary_{os.path.basename(base_dir)}_{timestamp}.txt")
    
    with open(output_file, 'w') as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write(f"AoA ANALYSIS SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Experiment: {os.path.basename(base_dir)}\n")
        f.write(f"Tag ID: {tag_id if tag_id else 'All tags'}\n")
        f.write(f"Generated: {timestamp}\n")
        f.write("=" * 80 + "\n\n")
        
        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total measurements: {len(summary_df)}\n")
        f.write(f"Overall RMSE: {np.sqrt(np.mean(summary_df['Error (°)']**2)):.2f}°\n")
        f.write(f"Mean absolute error: {np.mean(np.abs(summary_df['Error (°)'])):.2f}°\n")
        f.write(f"Maximum absolute error: {np.max(np.abs(summary_df['Error (°)'])):.2f}°\n\n")
        
        # RMSE by distance and frequency
        f.write("RMSE BY DISTANCE AND FREQUENCY\n")
        f.write("-" * 80 + "\n")
        f.write(rmse_df.to_string(index=False) + "\n\n")
        
        # Detailed results
        f.write("DETAILED RESULTS\n")
        f.write("-" * 80 + "\n")
        f.write(summary_df.to_string(index=False) + "\n\n")
    
    print(f"Summary report generated: {output_file}")
    
    # Also save as CSV for easier processing
    csv_file = output_file.replace(".txt", ".csv")
    summary_df.to_csv(csv_file, index=False)
    print(f"CSV data saved: {csv_file}")
    
    # Create a comprehensive figure with key metrics
    plt.figure(figsize=(12, 10))
    #  Use LaTeX for plot typography
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    
    # 1. RMSE by distance and frequency
    plt.subplot(2, 1, 1)
    rmse_pivot = rmse_df.pivot(index='Distance', columns='Frequency (MHz)', values='RMSE (°)')
    
    import seaborn as sns
    sns.heatmap(rmse_pivot, annot=True, cmap='viridis', fmt='.2f', linewidths=.5)
    
    plt.title('RMSE by Distance and Frequency (degrees)', fontsize=14)
    
    # 2. Error distribution
    plt.subplot(2, 1, 2)
    sns.boxplot(x='Distance', y='Error (°)', hue='Frequency (MHz)', data=summary_df)
    
    plt.title('Error Distribution by Distance and Frequency', fontsize=14)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_file.replace(".txt", "_summary.png"), dpi=300)
    plt.close()
    
    return summary_df, rmse_df

def aoa_analysis_from_df(df, tag_name, tag_id=None, output_dir=None):
    """
    Run AoA analysis using a pre-loaded DataFrame and generate plot.
    
    Args:
        df: DataFrame containing the measurement data
        tag_name: Name identifier for the tag
        tag_id: Tag ID (hex) to filter by, if None use all data
        output_dir: Directory to save the plot (if None, only display)
        
    Returns:
        Dictionary with analysis results
    """
    # Extract parameters from the dataframe
    frequency = df['frequencyMHz'].iloc[0]
    distance = df['distance'].iloc[0]  
    distance_antennas = df['antenna_spacing'].iloc[0]
    power_tx = df['power_dbm'].iloc[0] if 'power_dbm' in df.columns else 27.0
    
    # Calculate wavelength
    c = sc.c
    f = MHz_to_Hz(frequency) 
    lam = c / f
    L = distance_antennas
    D = distance
    
    # Filter by tag ID if provided
    if tag_id:
        if 'EPC' in df.columns:
            tagdf = df[df['EPC'] == tag_id]
        elif 'idHex' in df.columns:
            tagdf = df[df['idHex'] == tag_id]
        else:
            tagdf = df  # Assume already filtered
    else:
        tagdf = df
    
    # Initialize result lists
    aoa_experimental = []          # Storage for experimental AoA (θ)
    phi_experimental_tx = []       # Storage for experimental phase diff (from Tx)
    phi_experimental_tag = []      # Storage for experimental phase diff (from Tag)
    theta_calc_tx = []             # Storage for calculated AoA (θ) (from Tx)
    theta_calc_tag = []            # Storage for calculated AoA (θ) (from Tag)
    mean_pwr_antenna1 = []         # Storage for mean power (from Tx) (Antenna 1)
    std_pwr_antenna1 = []          # Storage for std power (from Tx) (Antenna 1)
    mean_pwr_antenna2 = []         # Storage for mean power (from Tx) (Antenna 2)
    std_pwr_antenna2 = []          # Storage for std power (from Tx) (Antenna 2)
    
    # Initialize phase offsets
    delta0_tx = 0.0
    delta0_tag = 0.0
    
    # Try to calculate hardware offset (optional)
    try:
        if 'phi_hw_offset_tx_from_df' in dir(dm):
            delta0_tx = dm.phi_hw_offset_tx_from_df(tagdf)
            delta0_tag = dm.phi_hw_offset_tag_from_df(tagdf)
        
        if np.isnan(delta0_tx) or np.isnan(delta0_tag):
            print(f"Warning: NaN hardware offset. Using defaults (0.0).")
            delta0_tx = 0.0
            delta0_tag = 0.0
    except Exception as e:
        print(f"Error calculating hardware offset: {e}. Using defaults (0.0).")
        delta0_tx = 0.0
        delta0_tag = 0.0
    
    # Get unique width values (horizontal offsets)
    widths = sorted(tagdf['width'].unique())
    
    # Process each width (horizontal offset)
    valid_measurements = []
    for i, w in enumerate(widths):
        # Filter data for this width
        width_df = tagdf[tagdf['width'] == w]
        
        # Check if both antennas have readings
        antenna1_readings = width_df[width_df['antenna'] == 1]
        antenna2_readings = width_df[width_df['antenna'] == 2]
        
        if len(antenna1_readings) == 0 or len(antenna2_readings) == 0:
            print(f"Warning: Missing antenna readings for width {w}. Skipping this measurement.")
            continue
        
        # Phase = from tag
        m1_tag = (dm.circular_mean_deg(antenna1_readings['phase']))/2
        m2_tag = (dm.circular_mean_deg(antenna2_readings['phase']))/2

        # Phase = from transceiver
        m1_tx = (dm.circular_mean_deg(antenna1_readings['phase']))
        m2_tx = (dm.circular_mean_deg(antenna2_readings['phase']))
        
        # Check for NaN values
        if np.isnan(m1_tag) or np.isnan(m2_tag) or np.isnan(m1_tx) or np.isnan(m2_tx):
            print(f"Warning: NaN phase values for width {w}. Skipping this measurement.")
            continue
        
        # Phase differences
        dphi_tag = (m1_tag - m2_tag)
        dphi_tx = (m1_tx - m2_tx)
        
        # Apply offset correction
        dphi_tag_corr = dphi_tag - delta0_tag
        dphi_tx_corr = dphi_tx - delta0_tx
        
        # AoA for this width
        theta = np.degrees(np.arctan2(w, D))
        
        # Store results
        aoa_experimental.append(theta)
        phi_experimental_tx.append(dphi_tx_corr)
        phi_experimental_tag.append(dphi_tag_corr)
        theta_calc_tx.append(np.degrees(np.arcsin(((lam)/(2*np.pi*L))*(np.deg2rad(dphi_tx_corr)))))
        theta_calc_tag.append(np.degrees(np.arcsin(((lam)/(2*np.pi*L))*(np.deg2rad(dphi_tag_corr)))))
        mean_pwr_antenna1.append(np.mean(antenna1_readings['rssi'].values))
        std_pwr_antenna1.append(np.std(antenna1_readings['rssi'].values))
        mean_pwr_antenna2.append(np.mean(antenna2_readings['rssi'].values))
        std_pwr_antenna2.append(np.std(antenna2_readings['rssi'].values))
        
        # Keep track of valid measurement index
        valid_measurements.append(i)
    
    # Calculate RMSE if we have data
    if aoa_experimental:
        rmse_tx = np.sqrt(np.mean((np.array(theta_calc_tx) - np.array(aoa_experimental))**2))
        rmse_tag = np.sqrt(np.mean((np.array(theta_calc_tag) - np.array(aoa_experimental))**2))
    else:
        rmse_tx = None
        rmse_tag = None
    
    # ------------------------ THEORETICAL CURVE (Δϕ₀ = 0) ----------------------- #
    if aoa_experimental:
        θ_range = np.linspace(min(aoa_experimental)-5, max(aoa_experimental)+5, 200)
        phi_th_tag = np.degrees((2*np.pi * L / lam) * np.sin(np.radians(θ_range))) - delta0_tag
        phi_th_tx = np.degrees((2*np.pi * L / lam) * np.sin(np.radians(θ_range))) - delta0_tx
    else:
        θ_range = np.linspace(-45, 45, 200)
        phi_th_tag = []
        phi_th_tx = []
    
    # ----------------------- STANDALONE PLOTS ----------------------- #
    if aoa_experimental:
        parts = tag_name.split(' at ')
        tag_base_name = parts[0]
        
        replica_str = "Unknown"
        if len(parts) > 1:
            loc_parts = parts[1].split(', ')
            if len(loc_parts) > 1:
                replica_str = loc_parts[1]
                
        # Format replica name
        replica_match = re.search(r'replica[-_]?(\d+)', replica_str, re.IGNORECASE)
        if replica_match:
            replica_num = replica_match.group(1)
            formatted_replica = f"Replica {replica_num}"
        else:
            formatted_replica = replica_str
        
        # Format distance (using the actual distance value we extracted earlier)
        formatted_dist = f"{distance:.3f}m"
        
        # Create a formatted title
        formatted_title = f"Estimated AoA - {tag_base_name} ({formatted_dist}, {formatted_replica})"

        # 1. Create a standalone figure with just the tag angle estimation plot
        plt.figure(figsize=(10, 8))
        #  Use LaTeX for plot typography
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.plot(aoa_experimental, aoa_experimental, 'g--', label='Ideal (y=x)')
        plt.plot(aoa_experimental, theta_calc_tag, 'ro', label='Estimated')
        plt.title(f"{formatted_title} @ {frequency} MHz, {power_tx}dBm", fontsize=14)
        plt.xlabel(r"Actual Angle $\theta$ [degrees]", fontsize=12)
        plt.ylabel(r"Estimated Angle $\theta$ [degrees]", fontsize=12)
        plt.grid(True)
        
        # Add RMSE information
        box_props = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='black')
        plt.text(0.05, 0.95, 
                f"RMSE = {rmse_tag:.2f}°\nL = {L} m\nf = {frequency} MHz\nD = {D} m", 
                transform=plt.gca().transAxes, fontsize=10, 
                verticalalignment='top', bbox=box_props)
        
        plt.legend()
        plt.tight_layout()
        # Save plot if output directory is provided
        if output_dir:
            # Create a safe filename from the formatted title
            safe_title = formatted_title.replace(" ", "_").replace("/", "_").replace(":", "")
            plot_filename = f"{safe_title}_{frequency}MHz.png"
            plot_path = os.path.join(output_dir, plot_filename)
            try:
                plt.savefig(plot_path, dpi=300)
                print(f"Saved tag plot to: {plot_path}")
            except Exception as e:
                print(f"Error saving tag plot: {e}")
        plt.close()
        
        # 2. Create a standalone figure with the transceiver angle estimation plot
        plt.figure(figsize=(10, 8))
        #  Use LaTeX for plot typography
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.plot(aoa_experimental, aoa_experimental, 'g--', label='Ideal (y=x)')
        plt.plot(aoa_experimental, theta_calc_tx, 'bo', label='Estimated')
        plt.title(f"{formatted_title} (Transceiver) @ {frequency} MHz, {power_tx}dBm", fontsize=14)
        plt.xlabel(r"Actual Angle $\theta$ [degrees]", fontsize=12)
        plt.ylabel(r"Estimated Angle $\theta$ [degrees]", fontsize=12)
        plt.grid(True)
        
        # Add RMSE information
        box_props = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='black')
        plt.text(0.05, 0.95, 
                f"RMSE = {rmse_tx:.2f}°\nL = {L} m\nf = {frequency} MHz\nD = {D} m", 
                transform=plt.gca().transAxes, fontsize=10, 
                verticalalignment='top', bbox=box_props)
        
        plt.legend()
        plt.tight_layout()

        # Save plot if output directory is provided
        if output_dir:
            # Create a safe filename from the formatted title
            safe_title = formatted_title.replace(" ", "_").replace("/", "_").replace(":", "")
            plot_filename = f"{safe_title}_transceiver_{frequency}MHz.png"
            plot_path = os.path.join(output_dir, plot_filename)
            try:
                plt.savefig(plot_path, dpi=300)
                print(f"Saved transceiver plot to: {plot_path}")
            except Exception as e:
                print(f"Error saving transceiver plot: {e}")
        plt.close()
    else:
        print("No valid data points to plot")
    
    # Return results dictionary
    return {
        'tag_name': tag_name,
        'frequency': frequency,
        'distance': distance,
        'antenna_spacing': distance_antennas,
        'power_tx': power_tx,
        'wavelength': lam,
        'aoa_experimental': aoa_experimental,
        'phi_experimental_tx': phi_experimental_tx,
        'phi_experimental_tag': phi_experimental_tag,
        'theta_calc_tx': theta_calc_tx,
        'theta_calc_tag': theta_calc_tag,
        'rmse_tx': rmse_tx,
        'rmse_tag': rmse_tag,
        'widths': [widths[i] for i in valid_measurements],
        'mean_pwr_antenna1': mean_pwr_antenna1,
        'std_pwr_antenna1': std_pwr_antenna1,
        'mean_pwr_antenna2': mean_pwr_antenna2,
        'std_pwr_antenna2': std_pwr_antenna2
    }

def analyze_phase_density(df, distance_key, replica_key, freq_key, output_dir=None):
    """
    Analyze and plot phase density for all frequencies to check for Gaussian distribution.
    
    Args:
        df: DataFrame containing phase data
        distance_key: Distance identifier
        replica_key: Replica identifier
        freq_key: Frequency identifier
        output_dir: Directory to save the plot (if None, only display)
    
    Returns:
        Dictionary with phase distribution analysis results
    """
    # Extract frequency in MHz for the title
    frequency = df['frequencyMHz'].iloc[0] if 'frequencyMHz' in df.columns else float(freq_key)

    # Get actual distance value
    distance = df['distance'].iloc[0]
    
    # Format replica name
    replica_match = re.search(r'replica[-_]?(\d+)', replica_key, re.IGNORECASE)
    if replica_match:
        replica_num = replica_match.group(1)
        formatted_replica = f"Replica {replica_num}"
    else:
        formatted_replica = replica_key
    
    # Format distance
    formatted_dist = f"{distance:.3f}m"
    
    # Create a formatted title
    formatted_title = f"Phase Density Distribution at {frequency} MHz ({formatted_dist}, {formatted_replica})"
    
    # Convert phase values to range [-π, π] if needed
    phase_values = df['phase'].values
    
    # Create figure for the phase density plot
    plt.figure(figsize=(10, 6))
    
    # Create histogram with density curve
    sns.histplot(phase_values, kde=True, bins=300)
    
    # Fit normal distribution to data
    from scipy.stats import norm
    mu, sigma = norm.fit(phase_values)
    x = np.linspace(min(phase_values), max(phase_values), 100)
    p = norm.pdf(x, mu, sigma)
    
    # Plot the fitted Gaussian curve
    plt.plot(x, p, 'r-', linewidth=2, 
             label=f'Gaussian fit:\n $\mu$={mu:.2f}, $\sigma$={sigma:.2f}')
    
    plt.title(formatted_title, fontsize=14)
    plt.xlabel('Phase (radians)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot if output directory is provided
    if output_dir:
        # Create a safe filename from the formatted title
        safe_title = formatted_title.replace(" ", "_").replace("/", "_").replace(":", "").replace("(", "").replace(")", "")
        plot_filename = f"{safe_title}_{frequency}MHz.png"
        plot_path = os.path.join(output_dir, plot_filename)
        try:
            plt.savefig(plot_path, dpi=300)
            print(f"Saved phase density plot to: {plot_path}")
        except Exception as e:
            print(f"Error saving phase density plot: {e}")
    plt.close()
    
    # Calculate additional statistics for normality assessment
    from scipy import stats
    k2, p_value = stats.normaltest(phase_values)
    
    return {
        'mu': mu,
        'sigma': sigma,
        'normality_test_statistic': k2,
        'normality_p_value': p_value,
        'is_gaussian': p_value > 0.05  # Typically p > 0.05 suggests Gaussian
    }

def analyze_hw_offset_vs_frequency(df, distance_key, replica_key, output_dir=None):
    """
    Analyze and plot hardware offset as a function of frequency.
    
    Args:
        df: DataFrame containing the hardware offset and frequency data
        distance_key: Distance identifier
        replica_key: Replica identifier
        output_dir: Directory to save the plot (if None, only display)
    
    Returns:
        Dictionary with analysis results
    """
    # Calculate hardware offset for each antenna combination
    try:

        # Get actual distance value
        distance = df['distance'].iloc[0]
        
        # Format replica name
        replica_match = re.search(r'replica[-_]?(\d+)', replica_key, re.IGNORECASE)
        if replica_match:
            replica_num = replica_match.group(1)
            formatted_replica = f"Replica {replica_num}"
        else:
            formatted_replica = replica_key
        
        # Format distance
        formatted_dist = f"{distance:.3f}m"
        
        # Create a formatted title
        formatted_title = f"Hardware Offset vs Frequency ({formatted_dist}, {formatted_replica})"

        # Group by frequency
        frequencies = sorted(df['frequencyMHz'].unique() if 'frequencyMHz' in df.columns else df['frequency'].unique())
        
        hw_offsets = []
        
        for freq in frequencies:
            # Filter by frequency
            if 'frequencyMHz' in df.columns:
                freq_df = df[df['frequencyMHz'] == freq]
            else:
                freq_df = df[df['frequency'] == freq]
            
            # Calculate hardware offset
            if 'phi_hw_offset_tx_from_df' in dir(dm):
                hw_offset = dm.phi_hw_offset_tx_from_df(freq_df)
            else:
                # Alternative calculation if function not available
                # Group by antenna
                ant1_data = freq_df[freq_df['antenna'] == 1]
                ant2_data = freq_df[freq_df['antenna'] == 2]
                
                if len(ant1_data) == 0 or len(ant2_data) == 0:
                    hw_offset = np.nan
                else:
                    # Use simple phase difference at boresight as approximation
                    boresight_data = freq_df[abs(freq_df['width']) < 0.01]  # near zero width
                    if len(boresight_data) > 0:
                        ant1_boresight = boresight_data[boresight_data['antenna'] == 1]
                        ant2_boresight = boresight_data[boresight_data['antenna'] == 2]
                        if len(ant1_boresight) > 0 and len(ant2_boresight) > 0:
                            phase1 = dm.circular_mean_deg(ant1_boresight['phase'])
                            phase2 = dm.circular_mean_deg(ant2_boresight['phase'])
                            hw_offset = phase1 - phase2
                        else:
                            hw_offset = np.nan
                    else:
                        hw_offset = np.nan
            
            hw_offsets.append({
                'frequency': freq,
                'hw_offset': hw_offset
            })
        
        # Create dataframe from results
        offset_df = pd.DataFrame(hw_offsets)
        offset_df = offset_df.dropna()
        
        if len(offset_df) > 0:
            # Create figure for the HW offset vs frequency plot
            plt.figure(figsize=(10, 6))
            
            plt.scatter(offset_df['frequency'], offset_df['hw_offset'], alpha=0.7)
            plt.plot(offset_df['frequency'], offset_df['hw_offset'], 'r-', alpha=0.5)
            
            plt.title(formatted_title, fontsize=14)
            plt.xlabel('Frequency (MHz)', fontsize=12)
            plt.ylabel('Hardware Phase Offset (degrees)', fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # Add polynomial fit
            if len(offset_df) > 2:
                from numpy.polynomial.polynomial import polyfit
                x = offset_df['frequency']
                y = offset_df['hw_offset']
                
                # Linear fit
                b, m = polyfit(x, y, 1)
                plt.plot(x, b + m*x, 'g--', label=f'Linear fit: {m:.4f}x + {b:.2f}')
                
                # 2nd order polynomial fit if we have enough data
                if len(offset_df) > 3:
                    coefs = polyfit(x, y, 2)
                    poly_y = coefs[0] + coefs[1]*x + coefs[2]*x**2
                    plt.plot(x, poly_y, 'b--', 
                             label=f'Quadratic fit: {coefs[2]:.4e}x² + {coefs[1]:.4f}x + {coefs[0]:.2f}')
                
                plt.legend()
            
            # Save plot if output directory is provided
            if output_dir:
                # Create a safe filename from the formatted title
                safe_title = formatted_title.replace(" ", "_").replace("/", "_").replace(":", "").replace("(", "").replace(")", "")
                plot_filename = f"{safe_title}.png"
                plot_path = os.path.join(output_dir, plot_filename)
                try:
                    plt.savefig(plot_path, dpi=300)
                    print(f"Saved HW offset vs frequency plot to: {plot_path}")
                except Exception as e:
                    print(f"Error saving HW offset plot: {e}")
            plt.close()
            
            return {
                'frequencies': offset_df['frequency'].tolist(),
                'hw_offsets': offset_df['hw_offset'].tolist(),
                'mean_offset': offset_df['hw_offset'].mean(),
                'std_offset': offset_df['hw_offset'].std()
            }
        else:
            print(f"No valid hardware offset data for {distance_key}, {replica_key}")
            return None
    except Exception as e:
        print(f"Error analyzing hardware offset vs frequency: {e}")
        return None
    
def analyze_phase_density_by_replica(freq_dataframes, distance_key, replica_key, output_dir=None):
    """
    Analyze and plot phase density for all frequencies in a replica on a single plot.
    
    Args:
        freq_dataframes: Dictionary mapping frequency keys to dataframes
        distance_key: Distance identifier
        replica_key: Replica identifier
        output_dir: Directory to save the plot (if None, only display)
    
    Returns:
        Dictionary with phase distribution analysis results for all frequencies
    """
    # Get a representative dataframe to extract metadata
    first_df = next(iter(freq_dataframes.values()))
    
    # Get actual distance value
    distance = first_df['distance'].iloc[0]
    
    # Format replica name
    replica_match = re.search(r'replica[-_]?(\d+)', replica_key, re.IGNORECASE)
    if replica_match:
        replica_num = replica_match.group(1)
        formatted_replica = f"Replica {replica_num}"
    else:
        formatted_replica = replica_key
    
    # Format distance
    formatted_dist = f"{distance:.3f}m"
    
    # Create a formatted title
    formatted_title = f"Phase Density Distributions ({formatted_dist}, {formatted_replica})"

    # Create figure for the combined phase density plot
    plt.figure(figsize=(12, 8))
    
    # Color palette for different frequencies
    colors = plt.cm.viridis(np.linspace(0, 1, len(freq_dataframes)))
    
    # Store results for each frequency
    frequency_results = {}
    
    # Plot each frequency's phase distribution
    for i, (freq_key, df) in enumerate(sorted(freq_dataframes.items())):
        # Extract frequency in MHz for the title
        frequency = df['frequencyMHz'].iloc[0] if 'frequencyMHz' in df.columns else float(freq_key)
        
        # Convert phase values to range [-π, π] if needed
        phase_values = df['phase'].values
        
        # Create KDE plot
        sns.kdeplot(phase_values, color=colors[i], label=f'{frequency} MHz', fill=True, alpha=0.3)
        
        # Fit normal distribution to data
        mu, sigma = norm.fit(phase_values)
        x = np.linspace(min(phase_values), max(phase_values), 100)
        p = norm.pdf(x, mu, sigma)
        
        # Plot the fitted Gaussian curve
        plt.plot(x, p, color=colors[i], linestyle='--', 
                 label=f'Gaussian fit {frequency} MHz:\n $\mu$={mu:.2f}, $\sigma$={sigma:.2f}')
        
        # Calculate normality statistics
        from scipy import stats
        k2, p_value = stats.normaltest(phase_values)
        
        # Store results for this frequency
        frequency_results[freq_key] = {
            'mu': mu,
            'sigma': sigma,
            'normality_test_statistic': k2,
            'normality_p_value': p_value,
            'is_gaussian': p_value > 0.05  # Typically p > 0.05 suggests Gaussian
        }
    
    plt.title(formatted_title, fontsize=14)
    plt.xlabel('Phase (radians)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Improve legend
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1), ncol=1)
    
    # Save plot if output directory is provided
    if output_dir:
        # Create a safe filename from the formatted title
        safe_title = formatted_title.replace(" ", "_").replace("/", "_").replace(":", "").replace("(", "").replace(")", "")
        plot_filename = f"{safe_title}.png"
        plot_path = os.path.join(output_dir, plot_filename)
        try:
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Saved combined phase density plot to: {plot_path}")
        except Exception as e:
            print(f"Error saving combined phase density plot: {e}")
    plt.close()
    
    return frequency_results

def analyze_aoa_by_distance(replica_results, distance_key, output_dir=None):
    """
    Create a combined AoA estimator plot showing all frequencies for a given distance.
    
    Args:
        replica_results: Dictionary of results for all replicas at this distance
        distance_key: Distance identifier
        output_dir: Directory to save the plot
    
    Returns:
        Dictionary with combined analysis results
    """
    # Extract a sample dataframe to get distance value
    sample_replica = next(iter(replica_results.values()))
    sample_freq = next((k for k in sample_replica.keys() if k != 'phase_density_by_replica' and k != 'hw_offset_analysis'), None)
    if not sample_freq:
        print(f"No frequency data found for distance {distance_key}")
        return None
    
    sample_df = sample_replica[sample_freq]
    distance = sample_df['distance']
    formatted_dist = f"{distance:.3f}m"
    
    # Create figure for combined AoA plot
    plt.figure(figsize=(12, 8))
    
    # Color palette for different frequencies
    freq_keys = set()
    for replica_key, frequencies in replica_results.items():
        for freq_key in frequencies:
            if freq_key not in ['phase_density_by_replica', 'hw_offset_analysis']:
                freq_keys.add(freq_key)
    
    freq_keys = sorted(freq_keys, key=lambda x: float(x) if x.replace('.', '', 1).isdigit() else x)
    colors = plt.cm.viridis(np.linspace(0, 1, len(freq_keys)))

    # Define marker shapes for better accessibility (for color blind readers)
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'H', '+', 'x', '<', '>']
    
    # Plot ideal line
    plt.plot([-15, 20], [-15, 20], 'k--', label='Ideal (y=x)', alpha=0.5)
    
    # Combined results for summary
    combined_results = {}
    
    # Plot each frequency's AoA estimations
    for i, freq_key in enumerate(freq_keys):
        all_actual = []
        all_estimated = []
        
        # Collect data from all replicas for this frequency
        for replica_key, frequencies in replica_results.items():
            if freq_key in frequencies:
                freq_data = frequencies[freq_key]
                if 'aoa_experimental' in freq_data and len(freq_data['aoa_experimental']) > 0:
                    all_actual.extend(freq_data['aoa_experimental'])
                    all_estimated.extend(freq_data['theta_calc_tag'])
        
        if all_actual and all_estimated:
            # Extract frequency value
            for replica_key, frequencies in replica_results.items():
                if freq_key in frequencies:
                    frequency = frequencies[freq_key]['frequency']
                    break
            
            # Calculate RMSE
            rmse = np.sqrt(np.mean((np.array(all_estimated) - np.array(all_actual))**2))
            
            # Plot this frequency's data
            marker_idx = i % len(markers)
            plt.scatter(all_actual, all_estimated, c=[colors[i]], marker=markers[marker_idx], s=77, label=f'{frequency} MHz (RMSE: {rmse:.2f}°)', alpha=0.8, edgecolors='k', linewidths=0.3)
            
            # Store combined results
            combined_results[freq_key] = {
                'frequency': frequency,
                'actual_aoa': all_actual,
                'estimated_aoa': all_estimated,
                'rmse': rmse
            }
    
    formatted_title = f"Combined AoA Estimation for All Frequencies at {formatted_dist}"
    plt.title(formatted_title, fontsize=16)
    plt.xlabel(r"Actual Angle $\theta$ [degrees]", fontsize=14)
    plt.ylabel(r"Estimated Angle $\theta$ [degrees]", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right', framealpha=0.9)

    # Set axis limits to focus on the -15 to 15 degree range
    plt.xlim(-15, 20)
    plt.ylim(-15, 20)
    
    # Add overall RMSE
    all_actual_combined = []
    all_estimated_combined = []
    for freq_data in combined_results.values():
        all_actual_combined.extend(freq_data['actual_aoa'])
        all_estimated_combined.extend(freq_data['estimated_aoa'])
    
    if all_actual_combined and all_estimated_combined:
        overall_rmse = np.sqrt(np.mean((np.array(all_estimated_combined) - np.array(all_actual_combined))**2))
        
        # Add text box with overall RMSE
        box_props = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='black')
        plt.text(0.05, 0.95, f"Overall RMSE: {overall_rmse:.2f}°\nDistance: {formatted_dist}", 
                transform=plt.gca().transAxes, fontsize=12, 
                verticalalignment='top', bbox=box_props)
    
    # Save plot if output directory is provided
    if output_dir:
        safe_filename = f"{formatted_title}.png"
        safe_filename = safe_filename.replace(" ", "_").replace("/", "_").replace(":", "").replace("(", "").replace(")", "")
        plot_path = os.path.join(output_dir, safe_filename)
        try:
            plt.savefig(plot_path, dpi=300)
            print(f"Saved combined AoA plot to: {plot_path}")
        except Exception as e:
            print(f"Error saving combined AoA plot: {e}")
    
    plt.close()
    
    return combined_results

def analyze_phase_density_by_distance(replica_results, distance_key, output_dir=None, data=None):
    """
    Create a combined phase density plot for all frequencies at a given distance.
    
    Args:
        replica_results: Dictionary of results for all replicas at this distance
        distance_key: Distance identifier
        output_dir: Directory to save the plot
    
    Returns:
        Dictionary with combined phase distribution results
    """
    # Extract a sample dataframe to get distance value
    sample_replica = next(iter(replica_results.values()))
    sample_freq = next((k for k in sample_replica.keys() if k != 'phase_density_by_replica' and k != 'hw_offset_analysis'), None)
    if not sample_freq:
        print(f"No frequency data found for distance {distance_key}")
        return None
    
    sample_df = sample_replica[sample_freq]
    distance = sample_df['distance']
    formatted_dist = f"{distance:.3f}m"
    
    # Create figure for combined phase density plot
    plt.figure(figsize=(12, 8))
    
    # Get all frequency keys
    freq_keys = set()
    for replica_key, frequencies in replica_results.items():
        for freq_key in frequencies:
            if freq_key not in ['phase_density_by_replica', 'hw_offset_analysis']:
                freq_keys.add(freq_key)
    
    freq_keys = sorted(freq_keys, key=lambda x: float(x) if x.replace('.', '', 1).isdigit() else x)
    colors = plt.cm.viridis(np.linspace(0, 1, len(freq_keys)))
    
    # Store results for each frequency
    frequency_results = {}
    
    # Plot each frequency's phase distribution
    for i, freq_key in enumerate(freq_keys):
        # Collect all phase values for this frequency across all replicas
        all_phases = []
        frequency = None
        
        # If data is provided, extract phases directly from source dataframes
        if data and distance_key in data:
            for replica_key, frequencies in data[distance_key].items():
                if freq_key in frequencies:
                    df = frequencies[freq_key]
                    all_phases.extend(df['phase'].values)
                    if frequency is None:
                        frequency = df['frequencyMHz'].iloc[0] if 'frequencyMHz' in df.columns else float(freq_key)
        else:
            # Try to extract phase values from results
            for replica_key, frequencies in replica_results.items():
                if freq_key in frequencies:
                    freq_data = frequencies[freq_key]
                    # The results might have aggregated data but not raw phases
                    # This is a fallback approach that might be incomplete
                    if 'phase_values' in freq_data:
                        all_phases.extend(freq_data['phase_values'])
                    if frequency is None and 'frequency' in freq_data:
                        frequency = freq_data['frequency']
        
        if all_phases:
            # Create KDE plot
            sns.kdeplot(all_phases, color=colors[i], label=f'{frequency} MHz', fill=True, alpha=0.3)
            
            # Fit normal distribution to data
            mu, sigma = norm.fit(all_phases)
            x = np.linspace(min(all_phases), max(all_phases), 100)
            p = norm.pdf(x, mu, sigma)
            
            # Plot the fitted Gaussian curve
            plt.plot(x, p, color=colors[i], linestyle='--', 
                     label=f'Gaussian fit {frequency} MHz:\n $\mu$={mu:.2f}, $\sigma$={sigma:.2f}')
            
            # Calculate normality statistics
            from scipy import stats
            k2, p_value = stats.normaltest(all_phases)
            
            # Store results for this frequency
            frequency_results[freq_key] = {
                'frequency': frequency,
                'mu': mu,
                'sigma': sigma,
                'normality_test_statistic': k2,
                'normality_p_value': p_value,
                'is_gaussian': p_value > 0.05  # Typically p > 0.05 suggests Gaussian
            }
    
    formatted_title = f"Phase Density Distributions at {formatted_dist} (All Replicas)"
    plt.title(formatted_title, fontsize=14)
    plt.xlabel('Phase (radians)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Improve legend
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1), ncol=1)
    
    # Save plot if output directory is provided
    if output_dir:
        safe_filename = f"{formatted_title}.png"
        safe_filename = safe_filename.replace(" ", "_").replace("/", "_").replace(":", "").replace("(", "").replace(")", "")
        plot_path = os.path.join(output_dir, safe_filename)
        try:
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Saved combined phase density plot to: {plot_path}")
        except Exception as e:
            print(f"Error saving combined phase density plot: {e}")
    
    return frequency_results

def analyze_hw_offset_by_distance(replica_results, distance_key, output_dir=None):
    """
    Create a hardware offset vs frequency plot combining data from all replicas at a given distance.
    
    Args:
        replica_results: Dictionary of results for all replicas at this distance
        distance_key: Distance identifier
        output_dir: Directory to save the plot
    
    Returns:
        Dictionary with combined hardware offset analysis
    """
    # Extract a sample dataframe to get distance value
    sample_replica = next(iter(replica_results.values()))
    sample_freq = next((k for k in sample_replica.keys() if k != 'phase_density_by_replica' and k != 'hw_offset_analysis'), None)
    if not sample_freq:
        print(f"No frequency data found for distance {distance_key}")
        return None
    
    sample_df = sample_replica[sample_freq]
    distance = sample_df['distance']
    formatted_dist = f"{distance:.3f}m"
    
    # Collect hardware offset data from all replicas
    hw_offset_data = []
    
    for replica_key, replica_data in replica_results.items():
        if 'hw_offset_analysis' in replica_data and replica_data['hw_offset_analysis']:
            offset_results = replica_data['hw_offset_analysis']
            frequencies = offset_results['frequencies']
            hw_offsets = offset_results['hw_offsets']
            
            for freq, offset in zip(frequencies, hw_offsets):
                hw_offset_data.append({
                    'frequency': freq,
                    'hw_offset': offset,
                    'replica': replica_key
                })
    
    if not hw_offset_data:
        print(f"No hardware offset data available for distance {distance_key}")
        return None
    
    # Create dataframe from collected data
    offset_df = pd.DataFrame(hw_offset_data)
    
    # Calculate mean and std by frequency
    freq_stats = offset_df.groupby('frequency').agg({
        'hw_offset': ['mean', 'std', 'count']
    }).reset_index()
    
    # Flatten the multi-index columns
    freq_stats.columns = ['frequency', 'mean_offset', 'std_offset', 'count']
    
    # Create figure for the combined HW offset plot
    plt.figure(figsize=(10, 6))
    
    # Plot individual points
    plt.scatter(offset_df['frequency'], offset_df['hw_offset'], alpha=0.4, label='Individual measurements')
    
    # Plot mean with error bars
    plt.errorbar(freq_stats['frequency'], freq_stats['mean_offset'], 
                yerr=freq_stats['std_offset'], fmt='ro-', elinewidth=2, capsize=6,
                label='Mean ± Std Dev')
    
    formatted_title = f"Hardware Offset vs Frequency at {formatted_dist} (All Replicas)"
    plt.title(formatted_title, fontsize=14)
    plt.xlabel('Frequency (MHz)', fontsize=12)
    plt.ylabel('Hardware Phase Offset (degrees)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add polynomial fits if we have enough data points
    if len(freq_stats) > 2:
        x = freq_stats['frequency']
        y = freq_stats['mean_offset']
        
        # Linear fit
        from numpy.polynomial.polynomial import polyfit
        b, m = polyfit(x, y, 1)
        plt.plot(x, b + m*x, 'g--', label=f'Linear fit: {m:.4f}x + {b:.2f}')
        
        # 2nd order polynomial fit if we have enough data
        if len(freq_stats) > 3:
            coefs = polyfit(x, y, 2)
            poly_y = coefs[0] + coefs[1]*x + coefs[2]*x**2
            plt.plot(x, poly_y, 'b--', 
                     label=f'Quadratic fit: {coefs[2]:.4e}x² + {coefs[1]:.4f}x + {coefs[0]:.2f}')
    
    plt.legend()
    
    # Save plot if output directory is provided
    if output_dir:
        safe_filename = f"{formatted_title}.png"
        safe_filename = safe_filename.replace(" ", "_").replace("/", "_").replace(":", "").replace("(", "").replace(")", "")
        plot_path = os.path.join(output_dir, safe_filename)
        try:
            plt.savefig(plot_path, dpi=300)
            print(f"Saved combined HW offset plot to: {plot_path}")
        except Exception as e:
            print(f"Error saving combined HW offset plot: {e}")
    
    plt.close()
    
    return {
        'frequencies': freq_stats['frequency'].tolist(),
        'mean_offsets': freq_stats['mean_offset'].tolist(),
        'std_offsets': freq_stats['std_offset'].tolist(),
        'counts': freq_stats['count'].tolist()
    }

def analyze_hw_offset_system(results, output_dir=None):
    """
    Analyze hardware offset across the entire system, showing relationships
    between offset, distance, and frequency.
    
    Args:
        results: Complete results dictionary with all distance/replica data
        output_dir: Directory to save the plots
    
    Returns:
        Dictionary with system-level hardware offset analysis
    """
    # Collect all hardware offset data across distances and frequencies
    all_offset_data = []
    
    for distance_key, replicas in results.items():
        # Skip combined analysis keys
        if distance_key in ['combined_aoa', 'combined_phase_density', 'combined_hw_offset']:
            continue
            
        # Extract actual distance value from first available replica/frequency
        first_replica = next(iter(replicas.values()))
        first_freq = next((k for k in first_replica.keys() if k not in ['phase_density_by_replica', 'hw_offset_analysis']), None)
        if not first_freq:
            continue
            
        actual_distance = first_replica[first_freq]['distance']
        
        # Process each replica for this distance
        for replica_key, replica_data in replicas.items():
            # Skip special keys
            if replica_key in ['combined_aoa', 'combined_phase_density', 'combined_hw_offset']:
                continue
                
            # Get hardware offset data if available
            if 'hw_offset_analysis' in replica_data and replica_data['hw_offset_analysis']:
                offset_results = replica_data['hw_offset_analysis']
                frequencies = offset_results['frequencies']
                hw_offsets = offset_results['hw_offsets']
                
                # Add each frequency/offset pair to our collection
                for freq, offset in zip(frequencies, hw_offsets):
                    all_offset_data.append({
                        'distance': actual_distance,
                        'distance_key': distance_key,
                        'replica': replica_key,
                        'frequency': freq,
                        'hw_offset': offset
                    })
    
    if not all_offset_data:
        print("No hardware offset data available for system analysis")
        return None
        
    # Create dataframe from collected data
    offset_df = pd.DataFrame(all_offset_data)
    
    # Calculate aggregate statistics by distance and frequency
    system_stats = offset_df.groupby(['distance', 'frequency']).agg({
        'hw_offset': ['mean', 'std', 'count']
    }).reset_index()
    
    # Flatten the multi-index columns
    system_stats.columns = ['distance', 'frequency', 'mean_offset', 'std_offset', 'count']
    
    # 1. Create a heatmap visualization
    plt.figure(figsize=(12, 8))
    
    # Pivot the data for the heatmap
    heatmap_data = offset_df.pivot_table(
        index='distance', 
        columns='frequency', 
        values='hw_offset',
        aggfunc='mean'
    )
    
    # Create heatmap
    sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt='.1f', linewidths=.5)
    
    plt.title('Hardware Offset by Distance and Frequency', fontsize=14)
    plt.xlabel('Frequency (MHz)', fontsize=12)
    plt.ylabel('Distance (m)', fontsize=12)
    
    # Save heatmap if output directory is provided
    if output_dir:
        heatmap_path = os.path.join(output_dir, "hw_offset_heatmap.png")
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        print(f"Saved hardware offset heatmap to: {heatmap_path}")
    
    plt.close()
    
    # 2. Create a multi-line plot with error bands
    plt.figure(figsize=(12, 8))
    
    # Get unique distances for color mapping
    unique_distances = sorted(offset_df['distance'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_distances)))
    
    for i, dist in enumerate(unique_distances):
        # Get data for this distance
        dist_data = system_stats[system_stats['distance'] == dist]
        
        # Sort by frequency
        dist_data = dist_data.sort_values('frequency')
        
        # Plot mean line
        plt.plot(dist_data['frequency'], dist_data['mean_offset'], 
                 color=colors[i], marker='o', label=f'Distance: {dist:.2f}m')
        
        # Add error bands
        plt.fill_between(
            dist_data['frequency'],
            dist_data['mean_offset'] - dist_data['std_offset'],
            dist_data['mean_offset'] + dist_data['std_offset'],
            alpha=0.2, color=colors[i]
        )
    
    plt.title('Hardware Offset vs. Frequency for Different Distances', fontsize=14)
    plt.xlabel('Frequency (MHz)', fontsize=12)
    plt.ylabel('Hardware Phase Offset (degrees)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    
    # Save multi-line plot if output directory is provided
    if output_dir:
        line_path = os.path.join(output_dir, "hw_offset_by_distance_frequency.png")
        plt.savefig(line_path, dpi=300, bbox_inches='tight')
        print(f"Saved hardware offset multi-line plot to: {line_path}")
    
    plt.close()
    
    # 3. Create a 3D surface plot for a more comprehensive visualization
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a pivot table for easier surface plotting
    pivot_data = offset_df.pivot_table(
        index='distance', 
        columns='frequency', 
        values='hw_offset',
        aggfunc='mean'
    )
    
    # Get X, Y, Z data for surface plot
    X, Y = np.meshgrid(pivot_data.columns, pivot_data.index)
    Z = pivot_data.values
    
    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.viridis, alpha=0.8, 
                          linewidth=0, antialiased=True)
    
    # Add scatter points for actual measurements
    for dist in unique_distances:
        dist_data = offset_df[offset_df['distance'] == dist]
        ax.scatter(dist_data['frequency'], 
                   dist_data['distance'], 
                   dist_data['hw_offset'],
                   color='red', alpha=0.5, s=10)
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Hardware Offset (degrees)')
    
    # Labels and title
    ax.set_xlabel('Frequency (MHz)', fontsize=12)
    ax.set_ylabel('Distance (m)', fontsize=12)
    ax.set_zlabel('Hardware Offset (degrees)', fontsize=12)
    ax.set_title('3D Surface: Hardware Offset by Distance and Frequency', fontsize=14)
    
    # Save 3D plot if output directory is provided
    if output_dir:
        surface_path = os.path.join(output_dir, "hw_offset_3d_surface.png")
        plt.savefig(surface_path, dpi=300, bbox_inches='tight')
        print(f"Saved hardware offset 3D surface plot to: {surface_path}")
    
    plt.close()
    
    return {
        'offset_data': offset_df,
        'system_stats': system_stats,
        'heatmap_data': heatmap_data
    }
# =================================================================================================================================== #


# =================================================================================================================================== #
# ---------------------------------------------------------------- MAIN ------------------------------------------------------------- #
def run_aoa_analysis(base_dir, tag_id=None):
    """Run complete AoA analysis across all distances, replicas, and frequencies."""
    # Create timestamped output directory
    output_parent_dir = os.path.dirname(base_dir)
    experiment_name = os.path.basename(base_dir)
    output_dir = create_output_directory(output_parent_dir, experiment_name)
    
    # Create subdirectories for plots and summary
    plots_dir = os.path.join(output_dir, "plots")
    summary_dir = os.path.join(output_dir, "summary")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    # Load data
    print("Loading experiment data...")
    data = load_aoa_data(base_dir, tag_id)

    # Get a summary of all dataframes
    summary_df = summarize_aoa_data(data)
    # Save data summary
    summary_df.to_csv(os.path.join(summary_dir, "data_summary.csv"), index=False)

    # Results storage
    results = {}
    
    # For storing summary data
    summary_rows = []

    # Process each distance/replica/frequency combination
    for distance_key, replicas in data.items():
        # Create distance directory
        distance_dir = os.path.join(plots_dir, distance_key)
        os.makedirs(distance_dir, exist_ok=True)

        results[distance_key] = {}

        for replica_key, frequencies in replicas.items():
            # Create replica directory
            replica_dir = os.path.join(distance_dir, replica_key)
            os.makedirs(replica_dir, exist_ok=True)

            results[distance_key][replica_key] = {}

            for freq_key, df in frequencies.items():
                print(f"Processing Distance: {distance_key}, Replica: {replica_key}, Frequency: {freq_key}...")

                # Run analysis with plot
                tag_name = f"{TAG_NAME} at {distance_key}, {replica_key}"
                analysis_results = aoa_analysis_from_df(df, tag_name, tag_id, replica_dir)
                
                # Store original phase values for distance-level analysis
                analysis_results['phase_values'] = df['phase'].values.tolist()

                # Phase density analysis by frequency
                phase_density_results = analyze_phase_density(df, distance_key, replica_key, freq_key, replica_dir)
                analysis_results['phase_density'] = phase_density_results

                # Hardware offset vs frequency analysis is done per replica, not per frequency
                # So we'll collect data for all frequencies and do the analysis once per replica
                if freq_key == list(frequencies.keys())[-1]:  # If this is the last frequency
                    # Combined phase density analysis for all frequencies in this replica
                    phase_density_by_replica = analyze_phase_density_by_replica(
                        frequencies, distance_key, replica_key, replica_dir)
                    # Store in replica-level results
                    if 'phase_density_by_replica' not in results[distance_key][replica_key]:
                        results[distance_key][replica_key]['phase_density_by_replica'] = phase_density_by_replica
                    # Combine all frequency data for this replica
                    combined_df = pd.concat([frequencies[f] for f in frequencies])
                    hw_offset_results = analyze_hw_offset_vs_frequency(combined_df, distance_key, replica_key, replica_dir)
                    # Store in replica-level results
                    if 'hw_offset_analysis' not in results[distance_key][replica_key]:
                        results[distance_key][replica_key]['hw_offset_analysis'] = hw_offset_results

                # Store results
                results[distance_key][replica_key][freq_key] = analysis_results
                
                # Add to summary rows for summary dataframe
                if analysis_results['rmse_tag'] is not None:
                    frequency_mhz = analysis_results['frequency'] / 1e6
                    summary_rows.append({
                        'Distance': distance_key,
                        'Distance_Value': analysis_results['distance'],
                        'Replica': replica_key,
                        'Frequency': freq_key,
                        'Frequency_MHz': frequency_mhz,
                        'RMSE_Tag': analysis_results['rmse_tag'],
                        'RMSE_Tx': analysis_results['rmse_tx'],
                        'Num_Measurements': len(analysis_results['aoa_experimental']),
                        'Antenna_Spacing': analysis_results['antenna_spacing'],
                        'Mean_RSSI_Ant1': np.mean(analysis_results['mean_pwr_antenna1']) if analysis_results['mean_pwr_antenna1'] else None,
                        'Mean_RSSI_Ant2': np.mean(analysis_results['mean_pwr_antenna2']) if analysis_results['mean_pwr_antenna2'] else None
                    })

        # After processing all replicas for this distance, do distance-level analysis
        print(f"Performing distance-level analysis for {distance_key}...")
        
        # Create a directory for distance-level plots
        distance_level_dir = os.path.join(distance_dir, "combined_analysis")
        os.makedirs(distance_level_dir, exist_ok=True)
        
        # Perform distance-level analyses
        results[distance_key]['combined_aoa'] = analyze_aoa_by_distance(
            results[distance_key], distance_key, distance_level_dir)
        
        results[distance_key]['combined_phase_density'] = analyze_phase_density_by_distance(
            results[distance_key], distance_key, distance_level_dir, data)
        
        results[distance_key]['combined_hw_offset'] = analyze_hw_offset_by_distance(
            results[distance_key], distance_key, distance_level_dir)
        
    # Create a summary dataframe from collected rows
    summary_df = pd.DataFrame(summary_rows)
    
    # Save analysis summary
    if not summary_df.empty:
        summary_df.to_csv(os.path.join(summary_dir, "analysis_summary.csv"), index=False)
    
        # Generate summary report
        report_path = os.path.join(summary_dir, "aoa_analysis_report.txt")
        with open(report_path, 'w') as f:
            # Header
            f.write("=" * 80 + "\n")
            f.write("AoA ANALYSIS SUMMARY\n")
            f.write("=" * 80 + "\n")
            f.write(f"Experiment: {experiment_name}\n")
            f.write(f"Tag ID: {tag_id if tag_id else 'All tags'}\n")
            f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            # Overall statistics
            f.write("OVERALL STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total measurements: {len(summary_df)}\n")
            f.write(f"Overall RMSE (Tag): {summary_df['RMSE_Tag'].mean():.2f}°\n")
            f.write(f"Best RMSE (Tag): {summary_df['RMSE_Tag'].min():.2f}°\n")
            f.write(f"Worst RMSE (Tag): {summary_df['RMSE_Tag'].max():.2f}°\n\n")
            
            # Summary by distance and frequency
            f.write("RMSE BY DISTANCE\n")
            f.write("-" * 80 + "\n")
            dist_summary = summary_df.groupby('Distance')['RMSE_Tag'].agg(['mean', 'min', 'max']).reset_index()
            f.write(dist_summary.to_string(index=False) + "\n\n")
            
            f.write("RMSE BY FREQUENCY\n")
            f.write("-" * 80 + "\n")
            freq_summary = summary_df.groupby('Frequency_MHz')['RMSE_Tag'].agg(['mean', 'min', 'max']).reset_index()
            f.write(freq_summary.to_string(index=False) + "\n\n")
            
            # Detailed results
            f.write("DETAILED RESULTS\n")
            f.write("-" * 80 + "\n")
            f.write(summary_df.to_string(index=False) + "\n\n")
        
        # Create summary visualizations
        # 1. RMSE by distance
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Distance', y='RMSE_Tag', data=summary_df)
        plt.title('RMSE by Distance', fontsize=14)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(summary_dir, "rmse_by_distance.png"), dpi=300)
        plt.close()
        
        # 2. RMSE by frequency
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Frequency_MHz', y='RMSE_Tag', data=summary_df)
        plt.title('RMSE by Frequency', fontsize=14)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(summary_dir, "rmse_by_frequency.png"), dpi=300)
        plt.close()
        
        # 3. Heatmap of RMSE by distance and frequency
        plt.figure(figsize=(12, 8))
        rmse_pivot = summary_df.pivot_table(index='Distance', columns='Frequency_MHz', values='RMSE_Tag')
        sns.heatmap(rmse_pivot, annot=True, cmap='viridis', fmt='.2f', linewidths=.5)
        plt.title('RMSE Heatmap by Distance and Frequency', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(summary_dir, "rmse_heatmap.png"), dpi=300)
        plt.close()
    else:
        print("Warning: No valid summary data was generated.")
    
    print("\nPerforming system-level hardware offset analysis...")
    system_dir = os.path.join(output_dir, "system_analysis")
    os.makedirs(system_dir, exist_ok=True)
    
    # Perform system-level hardware offset analysis
    results['system_hw_offset'] = analyze_hw_offset_system(results, system_dir)
    
    print("\nAnalysis Summary:")
    if not summary_df.empty:
        print(summary_df)
    else:
        print("No valid summary data to display.")
    print(f"\nAll results saved to: {output_dir}")
    
    return results, summary_df, output_dir
# =================================================================================================================================== #
# =================================================================================================================================== #
# ----------------------------------------------------------- DESCRIPTION ----------------------------------------------------------- #
# Contains functions to manage real data.                                                                                             #
# =================================================================================================================================== #


# =================================================================================================================================== #
# --------------------------------------------------------- EXTERNAL IMPORTS -------------------------------------------------------- #
import os                                       # Operating system dependent functionality.                                           #
import re                                       # Regular expression operations.                                                      #
import pandas          as pd                    # Data manipulation and analysis.                                                     #
import numpy           as np                    # Mathematical functions.                                                             #
import scipy.constants as sc                    # Physical and mathematical constants.                                                #
# =================================================================================================================================== #


# =================================================================================================================================== #
# ------------------------------------------------- CONVERSIONS & PARAMETER EXTRACTIONS --------------------------------------------- #
#   ----------------------------------------------------- FREQUENCY CONVERSIONS ---------------------------------------------------   #
def MHz_to_Hz(frequency: float) -> float:                                                                                             #
    """                                                                                                                               #                                   
    Convert the frequency from MHz to Hz.                                                                                             #
                                                                                                                                      #
    Parameters:                                                                                                                       #
        frequency [float]: The frequency to convert from MHz to Hz.                                                                   #
                                                                                                                                      #
    Returns:                                                                                                                          #
        The frequency in Hz. [float]                                                                                                  #
    """                                                                                                                               #
    return frequency * 1e6                                                                                                            #
                                                                                                                                      #
def get_lambda(frequency: float) -> float:                                                                                            #
    """                                                                                                                               #
    Calculate the wavelength (lambda) from the frequency.                                                                             #
                                                                                                                                      #
    Parameters:                                                                                                                       #
        frequency [float]: The frequency to calculate the wavelength from [Hz].                                                       #
                                                                                                                                      #
    Returns:                                                                                                                          #
        The wavelength (lambda) from the frequency. [float]                                                                           #
    """                                                                                                                               #
    return sc.speed_of_light / frequency                                                                                              #
#   ------------------------------------------------------- POWER CONVERSIONS -----------------------------------------------------   #
def dB_to_dBm(power: float) -> float:                                                                                                 #
    """                                                                                                                               #
    Convert the power from dB to dBm.                                                                                                 #
                                                                                                                                      # 
    Parameters:                                                                                                                       #
        power [float]: The power to convert from dB to dBm.                                                                           #
                                                                                                                                      #       
    Returns:                                                                                                                          #
        The power in dBm. [float]                                                                                                     #
    """                                                                                                                               #
    return power + 30                                                                                                                 #
                                                                                                                                      #
def dBm_to_dB(power: float) -> float:                                                                                                 #
    """                                                                                                                               #
    Convert the power from dBm to dB.                                                                                                 #
                                                                                                                                      # 
    Parameters:                                                                                                                       #
        power [float]: The power to convert from dBm to dB.                                                                           #
                                                                                                                                      #
    Returns:                                                                                                                          #
        The power in dB. [float]                                                                                                      #
    """                                                                                                                               #
    return power - 30                                                                                                                 #
                                                                                                                                      #
def dBm_to_W(power: float) -> float:                                                                                                  #
    """                                                                                                                               #
    Convert the power from dBm to W.                                                                                                  #
                                                                                                                                      #
    Parameters:                                                                                                                       #
        power [float]: The power to convert from dBm to W.                                                                            #
                                                                                                                                      #
    Returns:                                                                                                                          #
        The power in W. [float]                                                                                                       #
    """                                                                                                                               #
    return 10 ** (power / 10) / 1000                                                                                                  #
# =================================================================================================================================== #


# =================================================================================================================================== #
# ------------------------------------------------------ PARAMETER MANAGEMENT ------------------------------------------------------- #
def parameter_setup(frequency:float = 865.7, transmission_power: float = 27.0, antenna_gain:float = 6.0, tag_gain:float = 0.0, 
                    start_distance:float = 0.5, end_distance:float = 5.0, step_distance:float = 0.5) -> tuple:
    """
    Set up the parameters for the real data.

    Parameters:
        frequency (float): The frequency of the RFID system.
        transmission_power (float): The transmission power of the RFID system.
        antenna_gain (float): The gain of the RFID system's antenna.
        tag_gain (float): The gain of the RFID tag.
        start_distance (float): The starting distance for the measurements.
        end_distance (float): The ending distance for the measurements.
        step_distance (float): The step distance for the measurements.

    Returns:
        tuple: A tuple containing the frequency, transmission power, antenna gain, tag gain, and a list of expected
                distances.
    """
    expected_distances = np.arange(start_distance, end_distance + step_distance, step_distance).tolist()
    return frequency, transmission_power, antenna_gain, tag_gain, expected_distances
# =================================================================================================================================== #


# =================================================================================================================================== #
# ------------------------------------------------------ DIRECTORY MANAGEMENT ------------------------------------------------------- #
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
# =================================================================================================================================== #


# =================================================================================================================================== #
# -------------------------------------------------------- PHASE DIFFERENCE --------------------------------------------------------- #
def circular_mean_deg(phases):
    """
    Compute the circular mean of a set of phase angles in degrees.

    Parameters:
        - phases [array-like]: Array of phase angles in degrees.
           
    Returns:
        - Circular mean in degrees.
    """                                                 
    ang = np.radians(phases % 360)          # Convert to radians [0,2π)
    vec = np.exp(1j * ang).mean()           # Compute mean vector in complex plane
    return np.degrees(np.angle(vec)) % 360  # Convert back to degrees [0,360)

def phi_hw_offset_tag(theta_zero_file:str, tag:str):
    """
    Calculate the hardware offset for a specific tag based on phase measurements.

    Parameters:
        - theta_zero_file [str]: Path to the CSV file containing phase measurements, at zero degrees, for calibration.
        - tag [str]: The tag ID to filter measurements.

    Returns:
        - delta0 [float]: The calculated hardware offset for the tag, in degrees.
    """
    df = pd.read_csv(theta_zero_file)
    # Filter out non‐tag measurements
    df['idHex'] = df['idHex'].astype(str).str.strip()
    tagdf       = df[df['idHex'] == tag]
    # Arithmetic means per antenna
    m1_arithmetic = (circular_mean_deg(tagdf[tagdf['antenna']==1]['phase']))/2
    m2_arithmetic = (circular_mean_deg(tagdf[tagdf['antenna']==2]['phase']))/2
    # Phase differences
    dphi_arithmetic = (m1_arithmetic - m2_arithmetic)
    delta0 = dphi_arithmetic
    return delta0

def phi_hw_offset_tx(theta_zero_file:str, tag:str):
    """
    Calculate the hardware offset for a specific tag based on phase measurements, considering a full path.

    Parameters:
        - theta_zero_file [str]: Path to the CSV file containing phase measurements, at zero degrees, for calibration.
        - tag [str]: The tag ID to filter measurements.

    Returns:
        - delta0 [float]: The calculated hardware offset for the tag, in degrees.
    """
    df = pd.read_csv(theta_zero_file)
    # Filter out non‐tag measurements
    df['idHex'] = df['idHex'].astype(str).str.strip()
    tagdf       = df[df['idHex'] == tag]
    # Arithmetic means per antenna
    m1_arithmetic = (circular_mean_deg(tagdf[tagdf['antenna']==1]['phase']))
    m2_arithmetic = (circular_mean_deg(tagdf[tagdf['antenna']==2]['phase']))
    # Phase differences
    dphi_arithmetic = (m1_arithmetic - m2_arithmetic)
    delta0          = dphi_arithmetic
    return delta0

def process_file(file_path, data_list):
    """
    Process a single file and add all its data points to the appropriate list. 

    Parameters:
        - file_path [str]: Path to the CSV file to process.
        - data_list [list]: List to append processed data points.

    Returns:
        - None: The function modifies data_list in place.
    """
    try:
        # Extract metadata from filename
        filename = os.path.basename(file_path)
        # Parse filename: YYYYMMDD_FFF.F_D.DDD_L.LLL_W.WWW.csv
        pattern = r'(\d{8})_(\d+\.\d+)_(\d+\.\d+)_(\d+\.\d+)_([+-]?\d+\.\d+)\.csv'
        match   = re.match(pattern, filename)
        if not match:
            print(f"Warning: Filename {filename} doesn't match expected pattern.")
            return 
        date_str, freq_str, dist_str, spacing_str, width_str = match.groups()
        # Convert to appropriate types
        frequency       = MHz_to_Hz(float(freq_str)) # Convert MHz to Hz
        distance        = float(dist_str)            # Vertical distance in meters
        antenna_spacing = float(spacing_str)         # Inter-antenna spacing in meters
        width           = float(width_str)           # Horizontal offset in meters
        power           = 29.2                       # Default power
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
    
    Parameters:
        - base_dir: Base directory containing experiment data
        - tag_id: Optional tag ID to filter by
        
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
    """
    Print a summary of all dataframes in the AoA data structure.

    Parameters:
        - data: Dictionary with structure {distance: {replica: {frequency: dataframe}}}

    Returns:
        - DataFrame: Summary of all dataframes with metadata.
    """
    total_samples = 0
    summary_rows  = []
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
                    'DataFrame': df
                })
    print("-" * 70)
    print(f"Total Dataframes: {len(summary_rows)}")
    print(f"Total Samples: {total_samples}")
    return pd.DataFrame(summary_rows)
# =================================================================================================================================== #
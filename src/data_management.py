# =================================================================================================================================== #
# ----------------------------------------------------------- DESCRIPTION ----------------------------------------------------------- #
# Contains functions to manage real data.                                                                                             #
# =================================================================================================================================== #


# =================================================================================================================================== #
# --------------------------------------------------------- EXTERNAL IMPORTS -------------------------------------------------------- #
import pandas as pd # Data manipulation and analysis.                                                                                 #
import numpy as np # Mathematical functions.                                                                                          #
import os # Operating system dependent functionality.                                                                                 #
import re # Regular expression operations.                                                                                            #
import matplotlib.pyplot as plt # Plotting library.                                                                                   #
import src.gaussian_mixture_models as gmm # Gaussian Mixture Models.                                                                  #
from matplotlib.colors import ListedColormap  # Colormap for plotting.                                                                #
import src.gaussian_mixture_models_3D as gmm3D # Gaussian Mixture Models for 3D.                                                      #
# =================================================================================================================================== #


# =================================================================================================================================== #
# ------------------------------------------------------ REAL DATA MANAGEMENT ------------------------------------------------------- #
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

def dBm_to_W(power: float) -> float:
    """
    Convert the power from dBm to W.
    
    Parameters:
        power (float): The power to convert from dBm to W.
    
    Returns:
        The power in W. [float]
    """
    return 10 ** (power / 10) / 1000

def read_csv_files(csv_files, distances):
    """
    Read CSV files and extract RSSI values with their corresponding distances.
    
    Parameters:
        - csv_files (list): List of CSV file paths to read.
        - distances (list): List of distances corresponding to each CSV file.
    
    Returns:
        - numpy.ndarray: Array containing distance and RSSI value pairs.
    """
    data = []
    for csv_file, distance in zip(csv_files, distances):
        df = pd.read_csv(csv_file)
        rssi_values = df['peakRssi'].values
        distance_values = np.full(rssi_values.shape, distance)
        data.append(np.column_stack((distance_values, rssi_values)))
    return np.vstack(data)

def read_csv_files_rssi_phase(csv_files):
    """
    Read CSV files and extract RSSI and phase values.
    
    Parameters:
        - csv_files (list): List of CSV file paths to read.
    
    Returns:
        - numpy.ndarray: Array containing phase and RSSI value pairs.
    """
    data = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        rssi_values = df['peakRssi'].values
        phase_values = df['phase'].values
        data.append(np.column_stack((phase_values, rssi_values)))
    return np.vstack(data)

def read_csv_files_rssi_phase_new(files, distance_start, distance_end, moisture):
    """
    Read CSV files and extract RSSI and phase values for data points within specified distance range and moisture level.
    
    Parameters:
        - files (list): List of CSV file paths to read.
        - distance_start (float): Minimum distance to include.
        - distance_end (float): Maximum distance to include.
        - moisture (float): Moisture level to filter by.
    
    Returns:
        - numpy.ndarray: Array containing phase and RSSI value pairs.
    """
    data = []
    for file in files:
        df = pd.read_csv(file)
        # For each line in the csv file, if the distance is within the range, and the moisture level is a certain value, append the data
        for index, row in df.iterrows():
            distance = row['pos_y']
            mois = row['moist']
            if (distance_start <= distance <= distance_end) and (mois == moisture):
                rssi_values = row['peakRssi']
                phase_values = row['phase']

                # Convert phase from degrees to radians
                phase_rad = np.deg2rad(phase_values)

                # Wrap phase to [-pi, pi)
                phase_wrapped = (phase_rad + np.pi) % (2 * np.pi) - np.pi

                data.append(np.array([phase_wrapped, rssi_values]))
    return np.vstack(data)

def create_real_dataset_rssi_phase(distances:list, power_tx: float, frequency:float, obstacle_list: list, file_list: list, nx: int = 100, ny: int = 100, margin: float = 0.1) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a dataset from RSSI and phase measurements and visualize the data.
    
    Parameters:
        - distances (list): List of distances for measurements.
        - power_tx (float): Transmission power in dBm.
        - frequency (float): Frequency in MHz.
        - obstacle_list (list): List of obstacle names for legend.
        - file_list (list): List of file paths containing the measurement data.
        - nx (int, optional): Number of grid points in x direction. Defaults to 100.
        - ny (int, optional): Number of grid points in y direction. Defaults to 100.
        - margin (float, optional): Margin for the grid. Defaults to 0.1.
    
    Returns:
        - tuple: Tuple containing:
            - X (np.ndarray): Features matrix (phase and RSSI values).
            - y (np.ndarray): Labels array.
            - xx (np.ndarray): X coordinates for the grid.
            - yy (np.ndarray): Y coordinates for the grid.
    """
    # Extract distance range
    distance_start = distances[0]
    distance_end = distances[-1]

    plot_title = f"{power_tx} dBm ({round(dBm_to_W(power_tx), 1)}W) @ {frequency} MHz ({round(distance_start, 2)}m - {round(distance_end, 2)}m)"
    #plot_title = f"{power_tx} dBm ({round(dBm_to_W(power_tx), 1)}W) @ {frequency} MHz 0 m - 0 m)"
    
    obstacle_data = []

    for files in file_list:
        data = read_csv_files_rssi_phase(files)
        obstacle_data.append(data)

    X = np.vstack(obstacle_data)
    y = np.hstack([
        np.full(len(obstacle_data[i]), i) for i in range(len(obstacle_data))
    ])
        
    # Generate meshgrid
    xx, yy = gmm.get_meshgrid(X[:, 0], X[:, 1], nx, ny, margin=margin)
    
    # Use LaTeX for plot typography
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Create the scatter plot and set the figure size
    plt.figure(figsize=(6, 4))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap="Set1", edgecolors="k", linewidth=0.15, s=50)
    plt.xlabel("Phase [degrees]")
    plt.ylabel("RSSI [dBm]")
    plt.title(plot_title)

    # Add legend
    handles, _ = scatter.legend_elements()
    legend_labels = obstacle_list
    plt.legend(handles, legend_labels, title=r"$VWC [\%]$")

    plt.show()

    return X, y, xx, yy

def create_real_dataset_rssi_phase_3D(distances:list, power_tx: float, frequency:float, obstacle_list: list, file_list: list, nx: int = 100, ny: int = 100, nz: int = 100, margin: float = 0.1) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a dataset from RSSI and phase measurements and visualize the data in 3D using circular statistics.
    
    The phase values (originally in degrees from -180 to +180) are converted into their circular representation:
    - x-axis: cosine of the phase
    - y-axis: sine of the phase
    - z-axis: RSSI values
    
    Parameters:
        - distances (list): List of distances for measurements.
        - power_tx (float): Transmission power in dBm.
        - frequency (float): Frequency in MHz.
        - obstacle_list (list): List of obstacle names for legend.
        - file_list (list): List of file paths containing the measurement data.
        - nx (int, optional): Number of grid points in x direction. Defaults to 100.
        - ny (int, optional): Number of grid points in y direction. Defaults to 100.
        - nz (int, optional): Number of grid points in z direction. Defaults to 100.
        - margin (float, optional): Margin for the grid. Defaults to 0.1.
    
    Returns:
        - tuple: Tuple containing:
            - X (np.ndarray): Features matrix with 3 dimensions [cos(phase), sin(phase), RSSI].
            - y (np.ndarray): Labels array.
            - xx (np.ndarray): X coordinates for the 3D grid.
            - yy (np.ndarray): Y coordinates for the 3D grid.
            - zz (np.ndarray): Z coordinates for the 3D grid.
    """
    # Extract distance range
    distance_start = distances[0]
    distance_end = distances[-1]

    plot_title = f"{power_tx} dBm ({round(dBm_to_W(power_tx), 1)}W) @ {frequency} MHz ({round(distance_start, 2)}m - {round(distance_end, 2)}m)"
    
    obstacle_data = []

    for files in file_list:
        data_2d = read_csv_files_rssi_phase(files)

        phase_rad = np.deg2rad(data_2d[:, 0]) # Convert phase from degrees to radians (circular stats)

        # 3D feature matrix: [cos(phase), sin(phase), RSSI]
        data_3d = np.column_stack((
            np.cos(phase_rad), # x: cos(phase) 
            np.sin(phase_rad), # y: sin(phase)
            data_2d[:, 1] # z: RSSI
        ))

        obstacle_data.append(data_3d)

    # Stack all data
    X = np.vstack(obstacle_data)
    y = np.hstack([
        np.full(len(obstacle_data[i]), i) for i in range(len(obstacle_data))
    ])
        
    # Generate meshgrid
    xx, yy, zz = gmm3D.get_meshgrid_3D(X[:, 0], X[:, 1], X[:, 2], nx, ny, nz, margin=margin)
    
    # Use LaTeX for plot typography
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Create a 3D figure and axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create the 3D scatter plot
    scatter = ax.scatter(
        X[:, 0], X[:, 1], X[:, 2],  # x, y, z coordinates
        c=y, cmap="Set1", edgecolors="k", 
        linewidth=0.15, s=50
    )
    
    # Set labels
    ax.set_xlabel(r"$\cos(\theta)$")
    ax.set_ylabel(r"$\sin(\theta)$")
    ax.set_zlabel("RSSI [dBm]")
    ax.set_title(plot_title)

    # Add legend
    handles, _ = scatter.legend_elements()
    legend_labels = obstacle_list
    ax.legend(handles, legend_labels, title=r"$VWC [\%]$")

    plt.show()

    return X, y, xx, yy, zz

def create_real_dataset_rssi_phase_new(distances:list, power_tx: float, frequency:float, obstacle_list: list, file: str, nx: int = 100, ny: int = 100, margin: float = 0.1) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a dataset from RSSI and phase measurements from a single file and visualize the data.
    
    Parameters:
        - distances (list): List of distances for measurements.
        - power_tx (float): Transmission power in dBm.
        - frequency (float): Frequency in MHz.
        - obstacle_list (list): List of obstacle names for legend.
        - file (str): Path to the file containing all measurement data.
        - nx (int, optional): Number of grid points in x direction. Defaults to 100.
        - ny (int, optional): Number of grid points in y direction. Defaults to 100.
        - margin (float, optional): Margin for the grid. Defaults to 0.1.
    
    Returns:
        - tuple: Tuple containing:
            - X (np.ndarray): Features matrix (phase and RSSI values).
            - y (np.ndarray): Labels array.
            - xx (np.ndarray): X coordinates for the grid.
            - yy (np.ndarray): Y coordinates for the grid.
    """
    # Extract distance range
    distance_start = distances[0]
    distance_end = distances[-1]

    plot_title = f"{power_tx} dBm ({round(dBm_to_W(power_tx), 1)}W) @ {frequency} MHz ({round(distance_start, 2)}m - {round(distance_end, 2)}m)"
    #plot_title = f"{power_tx} dBm ({round(dBm_to_W(power_tx), 1)}W) @ {frequency} MHz 0 m - 0 m)"
    
    obstacle_data = []

    data = read_csv_files_rssi_phase_new(file, distance_start, distance_end, 0.01)
    obstacle_data.append(data)
    data = read_csv_files_rssi_phase_new(file, distance_start, distance_end, 0.18)
    obstacle_data.append(data)

    X = np.vstack(obstacle_data)
    y = np.hstack([
        np.full(len(obstacle_data[i]), i) for i in range(len(obstacle_data))
    ])
        
    # Generate meshgrid
    xx, yy = gmm.get_meshgrid(X[:, 0], X[:, 1], nx, ny, margin=margin)
    
    # Use LaTeX for plot typography
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Create the scatter plot
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap="Set1", edgecolors="k", linewidth=0.15, s=50)
    plt.xlabel("Phase [degrees]")
    plt.ylabel("RSSI [dBm]")
    plt.title(plot_title)

    # Add legend
    handles, _ = scatter.legend_elements()
    legend_labels = obstacle_list
    plt.legend(handles, legend_labels, title=r"$VWC [\%]$")

    plt.show()

    return X, y, xx, yy

def get_measurements(tag:str, filepath:str, obstacle: str) -> tuple:
    """
    Obtain a list of all files of a certain tag with a certain obstacle in the given directory.
    Files will be ordered by the distance value, in ascending order.
    Remember, the filename format is: {date}_{timestamp}_{tag}_{freq}MHz_{distance}m_{obstacle}_{power}dBm.csv

    Args:
        tag (str): The tag to search for.
        filepath (str): The directory to search in.
        obstacle (str): The obstacle to search for.

    Returns:
        list: A list of all files with the given tag and no obstacle.
        distances: A list of all distances in the files.
    """
    files = []
    # Get files with the given tag
    for file in os.listdir(filepath):
        if tag in file and obstacle in file:
            files.append(file)
    # Sort files by distance
    files.sort(key=lambda x: float(re.search(r'(\d+\.\d+)m', x).group(1)))
    distances = [float(re.search(r'(\d+\.\d+)m', file).group(1)) for file in files]
        
    # Prepend the filepath to each filename
    files = [os.path.join(filepath, file) for file in files]
    
    return files, distances

def get_measurements_power(tag:str, filepath:str, obstacle: str, power:float) -> tuple:
    """
    Obtain a list of all files of a certain tag with a certain obstacle in the given directory.
    Files will be ordered by the distance value, in ascending order.
    Remember, the filename format is: {date}_{timestamp}_{tag}_{freq}MHz_{distance}m_{obstacle}_{power}dBm.csv

    Args:
        tag (str): The tag to search for.
        filepath (str): The directory to search in.
        obstacle (str): The obstacle to search for.
        power (float): The transmission power to search for.

    Returns:
        list: A list of all files with the given tag and no obstacle.
        distances: A list of all distances in the files.
    """
    power_string = f"{power}dBm"
    files = []
    # Get files with the given tag
    for file in os.listdir(filepath):
        if tag in file and obstacle in file and power_string in file:
            files.append(file)
    # Sort files by distance
    files.sort(key=lambda x: float(re.search(r'(\d+\.\d+)m', x).group(1)))
    distances = [float(re.search(r'(\d+\.\d+)m', file).group(1)) for file in files]
        
    # Prepend the filepath to each filename
    files = [os.path.join(filepath, file) for file in files]
    
    return files, distances

def get_measurements_power_distance(tag:str, filepath:str, obstacle: str, power:float, start_distance:float, end_distance:float) -> list:
    power_string = f"{power}dBm"
    files = []
    # Get files with the given tag and power
    for file in os.listdir(filepath):
        if tag in file and obstacle in file and power_string in file:
            files.append(file)
    # Remove files that are not in the desired distance range
    files = [file for file in files if start_distance <= float(re.search(r'(\d+\.\d+)m', file).group(1)) <= end_distance]

    # Prepend the filepath to each filename
    files = [os.path.join(filepath, file) for file in files]

    return files

def append_to_filename_in_directory(directory_path: str, appendage: str) -> None:
    """
    Append a string to the end of all filenames in a directory before the .csv termination.

    Args:
        directory_path (str): The directory to search in.
        appendage (str): The string to append to the end of each filename before the .csv termination.
    """
    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            base, ext = os.path.splitext(filename)
            new_filename = f"{base}{appendage}{ext}"
            os.rename(os.path.join(directory_path, filename), os.path.join(directory_path, new_filename))

def append_power(directory_path: str) -> None:
    """
    Read the 'power' column of the first row of data, and append that power to the filename.
    """
    for file in os.listdir(directory_path):
        if file.endswith('.csv'):
            file_path = os.path.join(directory_path, file)
            try:
                df = pd.read_csv(file_path)
                if 'power' in df.columns:
                    power_values = df['power'].values
                    if len(power_values) > 0:
                        power = power_values[0]
                        # Make it a float
                        power = float(power)
                        appendage = f"_{power}dBm"
                        base, ext = os.path.splitext(file)
                        new_filename = f"{base}{appendage}{ext}"
                        os.rename(file_path, os.path.join(directory_path, new_filename))
                    else:
                        print(f"No data in file: {file}")
                else:
                    print(f"Missing 'power' column in file: {file}")
            except FileNotFoundError:
                print(f"File not found: {file}")
            except pd.errors.EmptyDataError:
                print(f"Empty data in file: {file}")
            except Exception as e:
                print(f"Error processing file {file}: {e}")


def circular_mean_deg(phases):                                                 
    ang = np.radians(phases % 360)          # Convert to radians [0,2π)
    vec = np.exp(1j * ang).mean()           # Compute mean vector in complex plane
    return np.degrees(np.angle(vec)) % 360  # Convert back to degrees [0,360)


def phi_hw_offset_tag(theta_zero_file:str, tag:str):
    df = pd.read_csv(theta_zero_file)
    # Filter out non‐tag measurements
    df['idHex'] = df['idHex'].astype(str).str.strip()
    tagdf = df[df['idHex'] == tag]
    # Arithmetic means per antenna
    m1_arithmetic = (circular_mean_deg(tagdf[tagdf['antenna']==1]['phase']))/2
    m2_arithmetic = (circular_mean_deg(tagdf[tagdf['antenna']==2]['phase']))/2

    # Phase differences
    dphi_arithmetic = (m1_arithmetic - m2_arithmetic)

    delta0 = dphi_arithmetic
    return delta0

def phi_hw_offset_tx(theta_zero_file:str, tag:str):
    df = pd.read_csv(theta_zero_file)
    # Filter out non‐tag measurements
    df['idHex'] = df['idHex'].astype(str).str.strip()
    tagdf = df[df['idHex'] == tag]
    # Arithmetic means per antenna
    m1_arithmetic = (circular_mean_deg(tagdf[tagdf['antenna']==1]['phase']))
    m2_arithmetic = (circular_mean_deg(tagdf[tagdf['antenna']==2]['phase']))

    # Phase differences
    dphi_arithmetic = (m1_arithmetic - m2_arithmetic)

    delta0 = dphi_arithmetic
    return delta0
# =================================================================================================================================== #

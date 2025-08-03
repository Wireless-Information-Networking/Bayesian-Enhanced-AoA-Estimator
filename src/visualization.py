# =================================================================================================================================== #
# ----------------------------------------------------------- DESCRIPTION ----------------------------------------------------------- #
# This script connects to the MQTT broker, subscribes to the topics 'data/meas' and 'rfid/tde', and handles the received data.        #
# It sends a message to the topic 'data/meas' with the parameters 'pos_x', 'pos_y', 'pos_z', and 'moist'. The script receives the     #
# RFID data from the topic 'rfid/tde' and writes the data to a file.                                                                  #
# =================================================================================================================================== #


# =================================================================================================================================== #
# --------------------------------------------------------- EXTERNAL IMPORTS -------------------------------------------------------- #
import pandas as pd  # Data manipulation and analysis.                                                                                #
import matplotlib.pyplot as plt  # Data visualization.                                                                                #
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting.                                                                               #
import scipy.constants as sc  # Physical and mathematical constants.                                                                  #
import numpy as np  # Mathematical functions.                                                                                         #
import src.data_management as dm # Data management functions.                                                                         #
# =================================================================================================================================== #


# =================================================================================================================================== #
# ---------------------------------------------------- EXPECTED BEHAVIOUR (FRIIS) --------------------------------------------------- #
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

def friis_equation_outward(power_tx: float, gain_tx: float, gain_rx: float, distance: float, wavelength: float) -> float:
    """
    Calculate the expected power, in dB, by applying the Friis equation from the transmitter to the tag.

    Parameters:
        - power_tx: The power transmitted by the transmitter, in dB [float]
        - gain_tx: The gain of the transmitter, in dBiL [float]
        - gain_rx: The gain of the receiver, in dBiL. [float]
        - distance: The distance between the transmitter and the receiver, in meters. [float]
        - wavelength: The wavelength of the signal, in meters. [float]

    Returns:
        - The power received by the tag, in dB. [float]
    """
    return power_tx + gain_tx + gain_rx + 20 * np.log10(wavelength / (4 * np.pi * distance))

def friis_equation_return(power_tx: float, gain_tx: float, gain_rx: float, distance: float, wavelength: float) -> float:
    """
    Calculate the expected power, in dB, by applying the Friis equation from the tag to the receiver.

    Parameters:
        - power_tx: The power transmitted by the transmitter, in dB [float]
        - gain_tx: The gain of the transmitter, in dBiL [float]
        - gain_rx: The gain of the receiver, in dBiL. [float]
        - distance: The distance between the transmitter and the receiver, in meters. [float]
        - wavelength: The wavelength of the signal, in meters. [float]

    Returns:
        - The power received by the receiver, in dB. [float]
    """
    return power_tx + gain_tx + gain_rx + 20 * np.log10(wavelength / (4 * np.pi * distance))

def friis_equation(power_tx: float, gain_tx: float, gain_rx: float, distance: float, frequency: float) -> float:
    """
    Calculate the expected power, in dB, by applying the Friis equation twice. The first time, the power is calculated from the 
    transmitter to the tag, and the second time, from the tag to the receiver.

    Parameters:
        - power_tx: The power transmitted by the transmitter, in dB [float]
        - gain_tx: The gain of the transmitter, in dBiL [float]
        - gain_rx: The gain of the receiver, in dBiL. [float]
        - distance: The distance between the transmitter and the receiver, in meters. [float]
        - frequency: The frequency of the signal, in MHz. [float]

    Returns:
        - The power received by the receiver, in dB. [float]
    """
    wavelength = get_lambda(MHz_to_Hz(frequency))
    power_tag = friis_equation_outward(power_tx, gain_tx, gain_rx, distance, wavelength)
    return friis_equation_return(power_tag, gain_tx, gain_rx, distance, wavelength)

def expected_behaviour(power_tx: float, gain_tx: float, gain_rx: float, distance: list, frequency: float) -> tuple:
    """
    Obtains important information to plot: the double wavelength (Far Field), the triple wavelength (Far Field), and the expected
    behaviour of the signal.

    Parameters:
        - power_tx: The power transmitted by the transmitter, in dB [float]
        - gain_tx: The gain of the transmitter, in dBiL [float]
        - gain_rx: The gain of the receiver, in dBiL. [float]
        - distance: The distances between the transmitter and the receiver, in meters. [list]
        - frequency: The frequency of the signal, in MHz. [float]

    Returns:
        - The double wavelength (Far Field), the triple wavelength (Far Field), and the expected behaviour of the signal. [tuple]
    """
    wavelength = get_lambda(MHz_to_Hz(frequency))
    two_lambda = 2 * wavelength
    three_lambda = 3 * wavelength
    return two_lambda, three_lambda, [friis_equation(power_tx, gain_tx, gain_rx, d, frequency) for d in distance]

def expected_fitted(power_tx: float, gain_tx: float, gain_rx: float, distance: list, frequency: float) -> list:
    """
    Obtains important information to plot: the double wavelength (Far Field), the triple wavelength (Far Field), and the expected
    behaviour of the signal. The parameters are the ones obtained from fitting.

    Parameters:
        - power_tx: The power transmitted by the transmitter, in dB [float]
        - gain_tx: The gain of the transmitter, in dBiL [float]
        - gain_rx: The gain of the receiver, in dBiL. [float]
        - distance: The distances between the transmitter and the receiver, in meters. [list]
        - frequency: The frequency of the signal, in MHz. [float]

    Returns:
        - The expected behaviour of the signal. [list]
    """
    return [friis_equation(power_tx, gain_tx, gain_rx, d, frequency) for d in distance]
# =================================================================================================================================== #


# =================================================================================================================================== #
# ------------------------------------------------------ PLOTTING FUNCTIONS --------------------------------------------------------- #
def plot_rssi_distance(filenames: list, tag_name: str, frequency: float, obstacle: str, power_tx: float, gain_tx: float, gain_rx: float,
                        real_distances: list, distances: list, gain_antenna: float, gain_tag: float, fit: bool = False, individual_markers: bool = True):
    """
    Plot the RSSI data (scatter) from the specified files (CSV) with the expected behaviour of the signal. The expected behaviour is 
    plotted as a line (a function representing perfect conditions), while the RSSI data from the files is plotted underneath, in a 
    scatter subplot. The distances are plotted in the x-axis, and the RSSI values are plotted in the y-axis. The distance of two lambda 
    and three lambda are also marked, and set as different colored areas. The RSSI values are plotted in dBm, and the distances are 
    plotted in meters.

    Parameters:
        - filenames: The files to read the RSSI data from. Each file contains data from a specific tag at a specific distance. [list]
        - tag_name: The name of the tag. [str]
        - frequency: The frequency of the signal, in MHz. [float]
        - obstacle: The obstacle between the transmitter and the receiver. [str]
        - power_tx: The power transmitted by the transmitter, in dBm. [float]
        - gain_tx: The gain of the transmitter, in dBiL. [float]
        - gain_rx: The gain of the receiver, in dBiL. [float]
        - real_distances: The real distances between the transmitter and the receiver, in meters. [list]
        - distances: The distances to plot the expected behaviour of the signal. [list]
        - gain_antenna: The gain of the antenna, in dBiL. [float]
        - gain_tag: The gain of the tag, in dBiL. [float]
        - fit: Whether to plot the fitted expected behaviour. [bool]

    Returns:
        - None
    """
    # Use LaTeX for plot typography
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plot_title = f"{tag_name} @ {frequency} MHz, {power_tx} dBm ({round(dBm_to_W(power_tx), 1)}W) - Obstacle: {obstacle}"
    plot_x_axis_title = "Distance [m]"
    plot_y_axis_title = "RSSI [dBm]"

    # Get the expected behaviour of the signal
    two_lambda, three_lambda, expected = expected_behaviour(dBm_to_dB(power_tx), gain_tx, gain_rx, distances, frequency)
    expected_fit = expected_fitted(dBm_to_dB(power_tx), gain_antenna, gain_tag, distances, frequency)

    mean_rssi = []
    std_rssi = []
    for i, filename in enumerate(filenames):
        data = pd.read_csv(filename)
        mean_rssi.append(data['peakRssi'].mean())
        std_rssi.append(data['peakRssi'].std())

    # Plot each point with its own label
    for i in range(len(real_distances)):
        if individual_markers:
            plt.errorbar(real_distances[i], mean_rssi[i], yerr=std_rssi[i], fmt='o', label=f'{real_distances[i]:.3f} m')
        else:
            plt.errorbar(real_distances[i], mean_rssi[i], yerr=std_rssi[i], fmt='o', c='red', label='RSSI' if i == 0 else "")

    if fit:
        # Plot the expected behaviour of the signal
        plt.plot(distances, expected, c='black', label='Expected Behaviour')
        plt.plot(distances, expected_fit, c='red', label='Expected Behaviour (Fitted)')
        # Format the gains to 2 decimal places
        og_antenna_text = f'Antenna Gain: {gain_tx:.2f} dBiL'
        og_tag_text = f'Tag Gain: {gain_rx:.2f} dBiL'
        gain_antenna_text = f'Fitted Antenna Gain: {gain_antenna:.2f} dBiL'
        gain_tag_text = f'Fitted Tag Gain: {gain_tag:.2f} dBiL'
        
        # Define the box properties
        box_props = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='black')
        
        # Place the text in the bottom left corner with a semi-transparent box
        plt.text(0.10, 0.80, f'{og_antenna_text}\n{og_tag_text}\n{gain_antenna_text}\n{gain_tag_text}', transform=plt.gca().transAxes, fontsize=10, verticalalignment='bottom', bbox=box_props)

    plt.axvline(x=two_lambda/8, color='black', linestyle='--', label=r'$\frac{\lambda}{4}$')
    plt.axvline(x=two_lambda/4, color='gray', linestyle='--', label=r'$\frac{\lambda}{2}$')
    plt.axvline(x=two_lambda/2, color='blue', linestyle='--', label=r'$\lambda$')
    plt.axvline(x=two_lambda, color='red', linestyle='--', label=r'$2 \lambda$')
    plt.axvline(x=three_lambda, color='green', linestyle='--', label=r'$3 \lambda$')

    # Set the title and labels
    plt.title(plot_title)
    plt.xlabel(plot_x_axis_title)
    plt.ylabel(plot_y_axis_title)
    plt.legend()
    plt.show()

def plot_rssi_distance_phase(filenames: list, tag_name: str, frequency: float, obstacle: str, power_tx: float, gain_tx: float, gain_rx: float,
                        real_distances: list, distances: list, gain_antenna: float, gain_tag: float, fit: bool = False, individual_markers: bool = True):
    """
    Plot the RSSI data (scatter) from the specified files (CSV) with the expected behaviour of the signal. The expected behaviour is 
    plotted as a line (a function representing perfect conditions), while the RSSI data from the files is plotted underneath, in a 
    scatter subplot. The distances are plotted in the x-axis, and the RSSI values are plotted in the y-axis. The distance of two lambda 
    and three lambda are also marked, and set as different colored areas. The RSSI values are plotted in dBm, and the distances are 
    plotted in meters.

    Parameters:
        - filenames: The files to read the RSSI data from. Each file contains data from a specific tag at a specific distance. [list]
        - tag_name: The name of the tag. [str]
        - frequency: The frequency of the signal, in MHz. [float]
        - obstacle: The obstacle between the transmitter and the receiver. [str]
        - power_tx: The power transmitted by the transmitter, in dBm. [float]
        - gain_tx: The gain of the transmitter, in dBiL. [float]
        - gain_rx: The gain of the receiver, in dBiL. [float]
        - real_distances: The real distances between the transmitter and the receiver, in meters. [list]
        - distances: The distances to plot the expected behaviour of the signal. [list]
        - gain_antenna: The gain of the antenna, in dBiL. [float]
        - gain_tag: The gain of the tag, in dBiL. [float]
        - fit: Whether to plot the fitted expected behaviour. [bool]

    Returns:
        - None
    """
    # Use LaTeX for plot typography
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plot_title = f"{tag_name} @ {frequency} MHz, {power_tx} dBm ({round(dBm_to_W(power_tx), 1)}W) - Obstacle: {obstacle}"
    plot_x_axis_title = "Distance [m]"
    plot_y_axis_title = "RSSI [dBm]"
    # Phase
    plot_y_axis_title_phase = "Phase [degrees]"

    # Get the expected behaviour of the signal
    two_lambda, three_lambda, expected = expected_behaviour(dBm_to_dB(power_tx), gain_tx, gain_rx, distances, frequency)
    expected_fit = expected_fitted(dBm_to_dB(power_tx), gain_antenna, gain_tag, distances, frequency)

    mean_rssi = []
    std_rssi = []
    mean_phase = []
    std_phase = []
    for i, filename in enumerate(filenames):
        data = pd.read_csv(filename)
        mean_rssi.append(data['peakRssi'].mean())
        std_rssi.append(data['peakRssi'].std())
        mean_phase.append(data['phase'].mean())
        std_phase.append(data['phase'].std())

    fig, ax1 = plt.subplots()
    # Plot RSSI
    color = 'tab:red'
    ax1.set_xlabel(plot_x_axis_title)
    ax1.set_ylabel(plot_y_axis_title, color=color)
    for i in range(len(real_distances)):
        ax1.errorbar(real_distances[i], mean_rssi[i], yerr=std_rssi[i], fmt='o', color=color, label='RSSI' if i == 0 else "")
    ax1.tick_params(axis='y', labelcolor=color)

    # Create a second y-axis for phase
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel(plot_y_axis_title_phase, color=color)
    for i in range(len(real_distances)):
        ax2.errorbar(real_distances[i], mean_phase[i], yerr=std_phase[i], fmt='o', color=color, label='Phase' if i == 0 else "")
    ax2.tick_params(axis='y', labelcolor=color)

    if fit:
        # Plot the expected behaviour of the signal
        plt.plot(distances, expected, c='black', label='Expected Behaviour')
        plt.plot(distances, expected_fit, c='red', label='Expected Behaviour (Fitted)')
        # Format the gains to 2 decimal places
        og_antenna_text = f'Antenna Gain: {gain_tx:.2f} dBiL'
        og_tag_text = f'Tag Gain: {gain_rx:.2f} dBiL'
        gain_antenna_text = f'Fitted Antenna Gain: {gain_antenna:.2f} dBiL'
        gain_tag_text = f'Fitted Tag Gain: {gain_tag:.2f} dBiL'
        
        # Define the box properties
        box_props = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='black')
        
        # Place the text in the bottom left corner with a semi-transparent box
        plt.text(0.10, 0.80, f'{og_antenna_text}\n{og_tag_text}\n{gain_antenna_text}\n{gain_tag_text}', transform=plt.gca().transAxes, fontsize=10, verticalalignment='bottom', bbox=box_props)

    plt.axvline(x=two_lambda/8, color='black', linestyle='--', label=r'$\frac{\lambda}{4}$')
    plt.axvline(x=two_lambda/4, color='gray', linestyle='--', label=r'$\frac{\lambda}{2}$')
    plt.axvline(x=two_lambda/2, color='blue', linestyle='--', label=r'$\lambda$')
    plt.axvline(x=two_lambda, color='red', linestyle='--', label=r'$2 \lambda$')
    plt.axvline(x=three_lambda, color='green', linestyle='--', label=r'$3 \lambda$')

    # Set the title and labels
    plt.suptitle(plot_title)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    fig.tight_layout()
    plt.show()

def plot_rssi_distance_indoor_vs_outdoor(filenames_outdoor: list, filenames_indoor: list, tag_name: str, frequency: float, obstacle: str, 
                                         power_tx: float, gain_tx: float, gain_rx: float, real_distances_outdoor: list, 
                                         real_distances_indoor: list, distances: list, gain_antenna_outdoor: float, gain_tag_outdoor: float,
                                         gain_antenna_indoor: float, gain_tag_indoor: float, fit: bool = False):
    """
    Plot the RSSI data (scatter) from the specified files (CSV) with the expected behaviour of the signal. The expected behaviour is 
    plotted as a line (a function representing perfect conditions), while the RSSI data from the files is plotted underneath, in a 
    scatter subplot. The distances are plotted in the x-axis, and the RSSI values are plotted in the y-axis. The distance of two lambda 
    and three lambda are also marked, and set as different colored areas. The RSSI values are plotted in dBm, and the distances are 
    plotted in meters.

    Parameters:
        - filenames_outdoor: The files to read the RSSI data from (outdoor). Each file contains data from a specific tag at a specific distance. [list]
        - filenames_indoor: The files to read the RSSI data from (indoor). Each file contains data from a specific tag at a specific distance. [list]
        - tag_name: The name of the tag. [str]
        - frequency: The frequency of the signal, in MHz. [float]
        - obstacle: The obstacle between the transmitter and the receiver. [str]
        - power_tx: The power transmitted by the transmitter, in dBm. [float]
        - gain_tx: The gain of the transmitter, in dBiL. [float]
        - gain_rx: The gain of the receiver, in dBiL. [float]
        - real_distances_outdoor: The real distances between the transmitter and the receiver (outdoor), in meters. [list]
        - real_distances_indoor: The real distances between the transmitter and the receiver (indoor), in meters. [list]
        - distances: The distances to plot the expected behaviour of the signal. [list]
        - gain_antenna_outdoor: The gain of the antenna (outdoor), in dBiL. [float]
        - gain_tag_outdoor: The gain of the tag (outdoor), in dBiL. [float]
        - gain_antenna_indoor: The gain of the antenna (indoor), in dBiL. [float]
        - gain_tag_indoor: The gain of the tag (indoor), in dBiL. [float]
        - fit: Whether to plot the fitted expected behaviour. [bool]

    Returns:
        - None
    """
    # Use LaTeX for plot typography
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plot_title = f"{tag_name} @ {frequency} MHz, {power_tx} dBm ({round(dBm_to_W(power_tx), 1)}W) - Indoor vs Outdoor"
    plot_x_axis_title = "Distance [m]"
    plot_y_axis_title = "RSSI [dBm]"

    # Get the expected behaviour of the signal
    two_lambda, three_lambda, expected = expected_behaviour(dBm_to_dB(power_tx), gain_tx, gain_rx, distances, frequency)
    expected_fit_indoor = expected_fitted(dBm_to_dB(power_tx), gain_antenna_indoor, gain_tag_indoor, distances, frequency)
    expected_fit_outdoor = expected_fitted(dBm_to_dB(power_tx), gain_antenna_outdoor, gain_tag_outdoor, distances, frequency)

    # INDOOR
    mean_rssi_indoor = []
    std_rssi_indoor = []
    for i, filename in enumerate(filenames_indoor):
        data = pd.read_csv(filename)
        mean_rssi_indoor.append(data['peakRssi'].mean())
        std_rssi_indoor.append(data['peakRssi'].std())

    # Plot each point with its own label
    for i in range(len(real_distances_indoor)):
        plt.errorbar(real_distances_indoor[i], mean_rssi_indoor[i], yerr=std_rssi_indoor[i], fmt='o', color='red', label='Indoor RSSI' if i == 0 else "")

    # OUTDOOR
    mean_rssi_outdoor = []
    std_rssi_outdoor = []
    for i, filename in enumerate(filenames_outdoor):
        data = pd.read_csv(filename)
        mean_rssi_outdoor.append(data['peakRssi'].mean())
        std_rssi_outdoor.append(data['peakRssi'].std())

    # Plot each point with its own label
    for i in range(len(real_distances_outdoor)):
        plt.errorbar(real_distances_outdoor[i], mean_rssi_outdoor[i], yerr=std_rssi_outdoor[i], fmt='o', color='blue', label='Outdoor RSSI' if i == 0 else "")

    # Plot the expected behaviour of the signal
    plt.plot(distances, expected, c='black', label='Expected Behaviour')
    if fit:
        plt.plot(distances, expected_fit_outdoor, c='blue', label='Expected Behaviour Outdoor (Fitted)')
        plt.plot(distances, expected_fit_indoor, c='red', label='Expected Behaviour Indoor (Fitted)')

        # Format the gains to 2 decimal places
        indoor_gain_antenna_text = f'Indoor Gain Antenna: {gain_antenna_indoor:.2f} dBiL'
        indoor_gain_tag_text = f'Indoor Gain Tag: {gain_tag_indoor:.2f} dBiL'
        outdoor_gain_antenna_text = f'Outdoor Gain Antenna: {gain_antenna_outdoor:.2f} dBiL'
        outdoor_gain_tag_text = f'Outdoor Gain Tag: {gain_tag_outdoor:.2f} dBiL'
        
        # Define the box properties
        box_props = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.5, edgecolor='black')
        
        # Place the text in the bottom left corner with a semi-transparent box
        plt.text(0.02, 0.02, f'{indoor_gain_antenna_text}\n{indoor_gain_tag_text}\n{outdoor_gain_antenna_text}\n{outdoor_gain_tag_text}', transform=plt.gca().transAxes, fontsize=10, verticalalignment='bottom', bbox=box_props)

    plt.axvline(x=two_lambda/8, color='black', linestyle='--', label=r'$\frac{\lambda}{4}$')
    plt.axvline(x=two_lambda/4, color='gray', linestyle='--', label=r'$\frac{\lambda}{2}$')
    plt.axvline(x=two_lambda/2, color='blue', linestyle='--', label=r'$\lambda$')
    plt.axvline(x=two_lambda, color='red', linestyle='--', label=r'$2 \lambda$')
    plt.axvline(x=three_lambda, color='green', linestyle='--', label=r'$3 \lambda$')


    # Set the title and labels
    plt.title(plot_title)
    plt.xlabel(plot_x_axis_title)
    plt.ylabel(plot_y_axis_title)
    plt.legend()
    plt.show()

def plot_rssi_distance_no_obstacle_vs_wood_vs_water(filenames_no_obstacle: list, filenames_wood: list, filenames_water: list, tag_name: str,
                                                    frequency: float, power_tx: float, gain_tx: float, gain_rx: float, real_distances: list, 
                                                    distances: list, gain_antenna: float, gain_tag: float, fit: bool = False):
    """
    Plot the RSSI data (scatter) from the specified files (CSV) with the expected behaviour of the signal. The expected behaviour is
    plotted as a line (a function representing perfect conditions), while the RSSI data from the files is plotted underneath, in a
    scatter subplot. The distances are plotted in the x-axis, and the RSSI values are plotted in the y-axis. The distance of two lambda
    and three lambda are also marked, and set as different colored areas. The RSSI values are plotted in dBm, and the distances are
    plotted in meters.

    Parameters:
        - filenames_no_obstacle: The files to read the RSSI data from (no obstacle). Each file contains data from a specific tag at a specific distance. [list]
        - filenames_wood: The files to read the RSSI data from (wood). Each file contains data from a specific tag at a specific distance. [list]
        - filenames_water: The files to read the RSSI data from (water). Each file contains data from a specific tag at a specific distance. [list]
        - tag_name: The name of the tag. [str]
        - frequency: The frequency of the signal, in MHz. [float]
        - power_tx: The power transmitted by the transmitter, in dBm. [float]
        - gain_tx: The gain of the transmitter, in dBiL. [float]
        - gain_rx: The gain of the receiver, in dBiL. [float]

    Returns:
        - None
    """
    # Use LaTeX for plot typography
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plot_title = f"{tag_name} @ {frequency} MHz, {power_tx} dBm ({round(dBm_to_W(power_tx), 1)}W) - Obstacle Comparison"
    plot_x_axis_title = "Distance [m]"
    plot_y_axis_title = "RSSI [dBm]"

    # Get the expected behaviour of the signal
    two_lambda, three_lambda, expected = expected_behaviour(dBm_to_dB(power_tx), gain_tx, gain_rx, distances, frequency)
    expected_fit = expected_fitted(dBm_to_dB(power_tx), gain_antenna, gain_tag, distances, frequency)

    # NO OBSTACLE
    mean_rssi_no_obstacle = []
    std_rssi_no_obstacle = []
    for i, filename in enumerate(filenames_no_obstacle):
        data = pd.read_csv(filename)
        mean_rssi_no_obstacle.append(data['peakRssi'].mean())
        std_rssi_no_obstacle.append(data['peakRssi'].std())

    # Plot each point with its own label
    for i in range(len(real_distances)):
        plt.errorbar(real_distances[i], mean_rssi_no_obstacle[i], yerr=std_rssi_no_obstacle[i], fmt='o', color='red', label='No Obstacle' if i == 0 else "")
    plt.plot(real_distances, mean_rssi_no_obstacle, color='red')

    # WOOD
    mean_rssi_wood = []
    std_rssi_wood = []
    for i, filename in enumerate(filenames_wood):
        data = pd.read_csv(filename)
        mean_rssi_wood.append(data['peakRssi'].mean())
        std_rssi_wood.append(data['peakRssi'].std())

    # Plot each point with its own label
    for i in range(len(real_distances)):
        plt.errorbar(real_distances[i], mean_rssi_wood[i], yerr=std_rssi_wood[i], fmt='o', color='green', label='Wood Plank' if i == 0 else "")
    plt.plot(real_distances, mean_rssi_wood, color='green')

    # WATER
    mean_rssi_water = []
    std_rssi_water = []
    for i, filename in enumerate(filenames_water):
        data = pd.read_csv(filename)
        mean_rssi_water.append(data['peakRssi'].mean())
        std_rssi_water.append(data['peakRssi'].std())

    # Plot each point with its own label
    for i in range(len(real_distances)):
        plt.errorbar(real_distances[i], mean_rssi_water[i], yerr=std_rssi_water[i], fmt='o', color='blue', label='Container w/ Water' if i == 0 else "")
    plt.plot(real_distances, mean_rssi_water, color='blue')

    if fit:
        plt.plot(distances, expected, c='black', label='Expected Behaviour')
        
        plt.plot(distances, expected_fit, c='red', label='Expected Behaviour (Fitted)')
        # Format the gains to 2 decimal places
        og_antenna_text = f'Antenna Gain: {gain_tx:.2f} dBiL'
        og_tag_text = f'Tag Gain: {gain_rx:.2f} dBiL'
        gain_antenna_text = f'Fitted Antenna Gain: {gain_antenna:.2f} dBiL'
        gain_tag_text = f'Fitted Tag Gain: {gain_tag:.2f} dBiL'
        
        # Define the box properties
        box_props = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='black')
        
        # Place the text in the bottom left corner with a semi-transparent box
        plt.text(0.10, 0.80, f'{og_antenna_text}\n{og_tag_text}\n{gain_antenna_text}\n{gain_tag_text}', transform=plt.gca().transAxes, fontsize=10, verticalalignment='bottom', bbox=box_props)
    
    plt.axvline(x=two_lambda/8, color='black', linestyle='--', label=r'$\frac{\lambda}{4}$')
    plt.axvline(x=two_lambda/4, color='gray', linestyle='--', label=r'$\frac{\lambda}{2}$')
    plt.axvline(x=two_lambda/2, color='blue', linestyle='--', label=r'$\lambda$')
    plt.axvline(x=two_lambda, color='red', linestyle='--', label=r'$2 \lambda$')
    plt.axvline(x=three_lambda, color='green', linestyle='--', label=r'$3 \lambda$')

    # Set the title and labels
    plt.title(plot_title)
    plt.xlabel(plot_x_axis_title)
    plt.ylabel(plot_y_axis_title)
    plt.legend()
    plt.show()

def plot_rssi_distance_all_tags(filenames: list, tag_names: list, frequency: float, obstacle: str, power_tx: float, gain_tx: float, gain_rx: float,
                        real_distances: list, distances: list):
    """
    Plot the RSSI data (scatter) from the specified files (CSV) with the expected behaviour of the signal. The expected behaviour is 
    plotted as a line (a function representing perfect conditions), while the RSSI data from the files is plotted underneath, in a 
    scatter subplot. The distances are plotted in the x-axis, and the RSSI values are plotted in the y-axis. The distance of two lambda 
    and three lambda are also marked, and set as different colored areas. The RSSI values are plotted in dBm, and the distances are plotted 
    in meters.

    Parameters:
        - filenames: List of lists with the files to read the RSSI data from. Each list item is a list of the files from a specific tag
                        at a specific range of distances, specified in real_distances. [list]
        - tag_names: The names of the tags, in the same order as the list items in filenames [list]
        - frequency: The frequency of the signal, in MHz. [float]
        - obstacle: The obstacle between the transmitter and the receiver. [str]
        - power_tx: The power transmitted by the transmitter, in dBm. [float]
        - gain_tx: The gain of the transmitter, in dBiL. [float]
        - gain_rx: The gain of the receiver, in dBiL. [float]
        - real_distances: The real distances between the transmitter and the receiver, in meters. [list]
        - distances: The distances to plot the expected behaviour of the signal. [list]

    Returns:
        - The RSSI data (scatter) from the specified files (CSV) with the expected behaviour of the signal. [None]
    """
    # Use LaTeX for plot typography
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plot_title = f"{power_tx} dBm ({round(dBm_to_W(power_tx), 1)}W) @ {frequency} MHz - Obstacle: {obstacle}"
    plot_x_axis_title = "Distance [m]"
    plot_y_axis_title = "RSSI [dBm]"

    # Get the expected behaviour of the signal
    two_lambda, three_lambda, expected = expected_behaviour(dBm_to_dB(power_tx), gain_tx, gain_rx, distances, frequency)

    # Dictionaries to store the mean & standard deviation of RSSI values for each tag at each distance
    mean_rssi = {}
    std_rssi = {}
    for i, tag in enumerate(tag_names):
        mean_rssi[tag] = []
        std_rssi[tag] = []
        for j, filename in enumerate(filenames[i]):
            data = pd.read_csv(filename)
            mean_rssi[tag].append(data['peakRssi'].mean())
            std_rssi[tag].append(data['peakRssi'].std())
    
    # Get a colormap
    colors = plt.cm.get_cmap('tab10', len(tag_names))

     # Plot the RSSI data for each tag with its label
    for i, tag in enumerate(tag_names):
        plt.errorbar(real_distances, mean_rssi[tag], yerr=std_rssi[tag], fmt='o', label=tag, color=colors(i))

    # Plot the expected behaviour of the signal
    plt.plot(distances, expected, c='black', label='Expected Behaviour')

    # Plot the two lambda and three lambda distances
    plt.axvline(x=two_lambda, color='red', linestyle='--', label=r'$2 \lambda$')
    plt.axvline(x=three_lambda, color='green', linestyle='--', label=r'$3 \lambda$')

    # Set the title and labels
    plt.title(plot_title)
    plt.xlabel(plot_x_axis_title)
    plt.ylabel(plot_y_axis_title)
    plt.legend()
    plt.show()

def plot_rssi_distance_average(filenames: list, frequency: float, obstacle: str, power_tx: float, gain_tx: float, gain_rx: float,
                    real_distances: list, distances: list):
    """
    Plot the RSSI data (scatter) from the specified files (CSV) with the expected behaviour of the signal. The expected behaviour is
    plotted as a line (a function representing perfect conditions), while the RSSI data from the files is plotted underneath, in a
    scatter subplot (average and std of all tags at a certain distance). The distances are plotted in the x-axis, and the RSSI values 
    are plotted in the y-axis. The distance of two lambda and three lambda are also marked, and set as different colored areas. The RSSI 
    values are plotted in dBm, and the distances are plotted in meters.

    Parameters:
        - filenames: List of lists with the files to read the RSSI data from. Each list item is a list of the files from a specific tag
                        at a specific range of distances, specified in real_distances. [list]
        - frequency: The frequency of the signal, in MHz. [float]
        - obstacle: The obstacle between the transmitter and the receiver. [str]
        - power_tx: The power transmitted by the transmitter, in dBm. [float]
        - gain_tx: The gain of the transmitter, in dBiL. [float]
        - gain_rx: The gain of the receiver, in dBiL. [float]
        - real_distances: The real distances between the transmitter and the receiver, in meters. [list]
        - distances: The distances to plot the expected behaviour of the signal. [list]

    Returns:
        - The RSSI data (scatter) from the specified files (CSV) with the expected behaviour of the signal. [None]
    """
    # Use LaTeX for plot typography
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plot_title = f"{power_tx} dBm ({round(dBm_to_W(power_tx), 1)}W) @ {frequency} MHz - Obstacle: {obstacle}"
    plot_x_axis_title = "Distance [m]"
    plot_y_axis_title = "RSSI [dBm]"

    # Get the expected behaviour of the signal
    two_lambda, three_lambda, expected = expected_behaviour(dBm_to_dB(power_tx), gain_tx, gain_rx, distances, frequency)

    # Lists to store the mean & standard deviation of RSSI values at each distance
    mean_rssi = []
    std_rssi = []

    # Aggregate RSSI values across all tags for each distance
    for i in range(len(real_distances)):
        rssi_values = []
        for tag_files in filenames:
            data = pd.read_csv(tag_files[i])
            rssi_values.extend(data['peakRssi'].values)
        mean_rssi.append(np.mean(rssi_values))
        std_rssi.append(np.std(rssi_values))

    # Plot the mean RSSI values with their standard deviation
    plt.errorbar(real_distances, mean_rssi, yerr=std_rssi, fmt='o', label='Mean RSSI')

    # Plot the expected behaviour of the signal
    plt.plot(distances, expected, c='black', label='Expected Behaviour')

    # Plot the two lambda and three lambda distances
    plt.axvline(x=two_lambda, color='red', linestyle='--', label=r'$2 \lambda$')
    plt.axvline(x=three_lambda, color='green', linestyle='--', label=r'$3 \lambda$')

    # Set the title and labels
    plt.title(plot_title)
    plt.xlabel(plot_x_axis_title)
    plt.ylabel(plot_y_axis_title)
    plt.legend()
    plt.show()     

def plot_rssi_distance_average_indoor_outdoor(filenames_outdoor: list, filenames_indoor: list, 
                                              frequency: float, obstacle: str, power_tx: float, gain_tx: float, gain_rx: float,
                                              real_distances_outdoor: list, real_distances_indoor: list, distances: list):
    """
    Plot the RSSI data (scatter) from the specified files (CSV) with the expected behaviour of the signal. The expected behaviour is
    plotted as a line (a function representing perfect conditions), while the RSSI data from the files is plotted underneath, in a
    scatter subplot (average and std of all tags at a certain distance). The distances are plotted in the x-axis, and the RSSI values 
    are plotted in the y-axis. The distance of two lambda and three lambda are also marked, and set as different colored areas. The RSSI 
    values are plotted in dBm, and the distances are plotted in meters.

    Parameters:
        - filenames_outdoor: List of lists with the files to read the RSSI data from (outdoor). Each list item is a list of the files from a specific tag
                        at a specific range of distances, specified in real_distances_outdoor. [list]
        - filenames_indoor: List of lists with the files to read the RSSI data from (indoor). Each list item is a list of the files from a specific tag
                        at a specific range of distances, specified in real_distances_indoor. [list]
        - frequency: The frequency of the signal, in MHz. [float]
        - obstacle: The obstacle between the transmitter and the receiver. [str]
        - power_tx: The power transmitted by the transmitter, in dBm. [float]
        - gain_tx: The gain of the transmitter, in dBiL. [float]
        - gain_rx: The gain of the receiver, in dBiL. [float]
        - real_distances_outdoor: The real distances between the transmitter and the receiver (outdoor), in meters. [list]
        - real_distances_indoor: The real distances between the transmitter and the receiver (indoor), in meters. [list]
        - distances: The distances to plot the expected behaviour of the signal. [list]

    Returns:
        - The RSSI data (scatter) from the specified files (CSV) with the expected behaviour of the signal. [None]
    """
    # Use LaTeX for plot typography
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plot_title = f"{power_tx} dBm ({round(dBm_to_W(power_tx), 1)}W) @ {frequency} MHz - - Indoor vs Outdoor"
    plot_x_axis_title = "Distance [m]"
    plot_y_axis_title = "RSSI [dBm]"

    # Get the expected behaviour of the signal
    two_lambda, three_lambda, expected = expected_behaviour(dBm_to_dB(power_tx), gain_tx, gain_rx, distances, frequency)

    # INDOOR
    # Lists to store the mean & standard deviation of RSSI values at each distance
    mean_rssi_indoor = []
    std_rssi_indoor = []

    # Aggregate RSSI values across all tags for each distance
    for i in range(len(real_distances_indoor)):
        rssi_values = []
        for tag_files in filenames_indoor:
            data = pd.read_csv(tag_files[i])
            rssi_values.extend(data['peakRssi'].values)
        mean_rssi_indoor.append(np.mean(rssi_values))
        std_rssi_indoor.append(np.std(rssi_values))

    # Plot the mean RSSI values with their standard deviation
    plt.errorbar(real_distances_indoor, mean_rssi_indoor, yerr=std_rssi_indoor, fmt='o', label='Mean RSSI (Indoor)')

    # OUTDOOR
    # Lists to store the mean & standard deviation of RSSI values at each distance
    mean_rssi_outdoor = []
    std_rssi_outdoor = []

    # Aggregate RSSI values across all tags for each distance
    for i in range(len(real_distances_outdoor)):
        rssi_values = []
        for tag_files in filenames_outdoor:
            data = pd.read_csv(tag_files[i])
            rssi_values.extend(data['peakRssi'].values)
        mean_rssi_outdoor.append(np.mean(rssi_values))
        std_rssi_outdoor.append(np.std(rssi_values))

    # Plot the mean RSSI values with their standard deviation
    plt.errorbar(real_distances_outdoor, mean_rssi_outdoor, yerr=std_rssi_outdoor, fmt='o', label='Mean RSSI (Outdoor)')

    # Plot the expected behaviour of the signal
    plt.plot(distances, expected, c='black', label='Expected Behaviour')

    # Plot the two lambda and three lambda distances
    plt.axvline(x=two_lambda, color='red', linestyle='--', label=r'$2 \lambda$')
    plt.axvline(x=three_lambda, color='green', linestyle='--', label=r'$3 \lambda$')

    # Set the title and labels
    plt.title(plot_title)
    plt.xlabel(plot_x_axis_title)
    plt.ylabel(plot_y_axis_title)
    plt.legend()
    plt.show() 

def create_combined_plot(filenames_outdoor, filenames_indoor, tag_name, frequency, obstacle, power_tx, gain_tx, gain_rx, 
                         real_distances_outdoor, real_distances_indoor, distances, gain_antenna_outdoor, tag00_gain_outdoor, 
                         gain_antenna_indoor, tag00_gain_indoor):
    """
    Create a combined plot with four subplots.

    Parameters:
        - All parameters required by the plotting functions.

    Returns:
        - None
    """
     # Use LaTeX for plot typography
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # Subplot 1
    plt.sca(axs[0, 0])
    plot_rssi_distance(filenames_outdoor, tag_name, frequency, obstacle, power_tx, gain_tx, gain_rx,
                       real_distances_outdoor, distances, gain_antenna_outdoor, tag00_gain_outdoor)
    
    # Subplot 2
    plt.sca(axs[0, 1])
    plot_rssi_distance(filenames_outdoor, tag_name, frequency, obstacle, power_tx, gain_tx, gain_rx,
                       real_distances_outdoor, distances, gain_antenna_outdoor, tag00_gain_outdoor, True)
    
    # Subplot 3
    plt.sca(axs[1, 0])
    plot_rssi_distance_indoor_vs_outdoor(filenames_outdoor, filenames_indoor, tag_name, frequency, obstacle, power_tx, gain_tx, gain_rx, 
                                         real_distances_outdoor, real_distances_indoor, distances, gain_antenna_outdoor, tag00_gain_outdoor, 
                                         gain_antenna_indoor, tag00_gain_indoor)
    
    # Subplot 4
    plt.sca(axs[1, 1])
    plot_rssi_distance_indoor_vs_outdoor(filenames_outdoor, filenames_indoor, tag_name, frequency, obstacle, power_tx, gain_tx, gain_rx, 
                                         real_distances_outdoor, real_distances_indoor, distances, gain_antenna_outdoor, tag00_gain_outdoor, 
                                         gain_antenna_indoor, tag00_gain_indoor, True)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, hspace=0.4, wspace=0.4)
    plt.show()

def create_combined_plot_obstacles(no_obstacle, wood, water, tag_name, frequency, transmission_power, antenna_gain, tag_gain,
                                   real_distances, expected_distances, antenna_gain_fitted, tag_gain_fitted, fit):
    """
    Create a combined plot with four subplots.

    Parameters:
        - All parameters required by the plotting functions.

    Returns:
        - None
    """
    # Use LaTeX for plot typography
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # Subplot 1 - No Obstacle
    plt.sca(axs[0, 0])
    plot_rssi_distance(no_obstacle, tag_name, frequency, "None", transmission_power, antenna_gain,
                        tag_gain, real_distances, expected_distances, antenna_gain_fitted, tag_gain_fitted, fit, False)

    # Subplot 2 - Wood
    plt.sca(axs[0, 1])
    plot_rssi_distance(wood, tag_name, frequency, "Wood", transmission_power, antenna_gain,
                        tag_gain, real_distances, expected_distances, antenna_gain_fitted, tag_gain_fitted, fit, False)

    # Subplot 3 - Container with Water
    plt.sca(axs[1, 0])
    plot_rssi_distance(water, tag_name, frequency, "Container w/ Water", transmission_power, antenna_gain,
                        tag_gain, real_distances, expected_distances, antenna_gain_fitted, tag_gain_fitted, fit, False)

    # Subplot 4 - All Obstacles
    plt.sca(axs[1, 1])
    plot_rssi_distance_no_obstacle_vs_wood_vs_water(no_obstacle, wood, water, tag_name, frequency,
                                                    transmission_power, antenna_gain, tag_gain, real_distances, expected_distances, antenna_gain_fitted, tag_gain_fitted, fit)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, hspace=0.4, wspace=0.4)
    plt.show()
# =================================================================================================================================== #


# =================================================================================================================================== #
# ---------------------------------------------------- 3D Plots - RSSI & Phase ------------------------------------------------------ #
def basic_3D_plot2(filenames:list, distances:list, tag_name:str, frequency:float, obstacle:str, power_tx:float, gain_tx:float, gain_rx:float):
    # Use LaTeX for plot typography
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plot_title = f"{tag_name} @ {frequency} MHz, {power_tx} dBm ({round(dBm_to_W(power_tx), 1)}W) - Obstacle: {obstacle}"
    plot_x_axis_title = "Distance [m]"
    plot_z_axis_title = "RSSI [dBm]"
    plot_y_axis_title = "Phase [degrees]"

    # Obtain the RSSI and Phase data
    all_rssi = []
    all_phase = []
    all_distances = []
    for i, filename in enumerate(filenames):
        data = pd.read_csv(filename)
        all_rssi.extend(data['peakRssi'].values)
        all_phase.extend(data['phase'].values)
        all_distances.extend([distances[i]] * len(data))

    # Create the 3D plot
    fig = plt.figure(figsize=(14, 14))
    
    # First subplot
    ax1 = fig.add_subplot(221, projection='3d')
    scatter1 = ax1.scatter(all_distances, all_phase, all_rssi, c=all_distances, cmap='viridis', s=100, alpha=0.6)
    ax1.set_xlabel(plot_x_axis_title)
    ax1.set_ylabel(plot_y_axis_title)
    ax1.set_zlabel(plot_z_axis_title)
    ax1.set_title('View Angle 1')
    ax1.view_init(elev=20., azim=30)
    
    # Second subplot
    ax2 = fig.add_subplot(222, projection='3d')
    scatter2 = ax2.scatter(all_distances, all_phase, all_rssi, c=all_distances, cmap='viridis', s=100, alpha=0.6)
    ax2.set_xlabel(plot_x_axis_title)
    ax2.set_ylabel(plot_y_axis_title)
    ax2.set_zlabel(plot_z_axis_title)
    ax2.set_title('View Angle 2')
    ax2.view_init(elev=30., azim=60)
    
    # Third subplot
    ax3 = fig.add_subplot(223, projection='3d')
    scatter3 = ax3.scatter(all_distances, all_phase, all_rssi, c=all_distances, cmap='viridis', s=100, alpha=0.6)
    ax3.set_xlabel(plot_x_axis_title)
    ax3.set_ylabel(plot_y_axis_title)
    ax3.set_zlabel(plot_z_axis_title)
    ax3.set_title('View Angle 3')
    ax3.view_init(elev=40., azim=90)
    
    # Fourth subplot
    ax4 = fig.add_subplot(224, projection='3d')
    scatter4 = ax4.scatter(all_distances, all_phase, all_rssi, c=all_distances, cmap='viridis', s=100, alpha=0.6)
    ax4.set_xlabel(plot_x_axis_title)
    ax4.set_ylabel(plot_y_axis_title)
    ax4.set_zlabel(plot_z_axis_title)
    ax4.set_title('View Angle 4')
    ax4.view_init(elev=50., azim=120)
    
    fig.suptitle(plot_title)
    fig.colorbar(scatter1, ax=[ax1, ax2, ax3, ax4], label='Distance [m]')
    
    plt.show()

def plot_rssi_distance_all_points(filenames: list, tag_name: str, frequency: float, obstacle: str, power_tx: float, gain_tx: float, gain_rx: float,
                        real_distances: list, distances: list, gain_antenna: float, gain_tag: float, fit: bool = False, individual_markers: bool = True):
    """
    Plot the RSSI data (scatter) from the specified files (CSV) with the expected behaviour of the signal. The expected behaviour is 
    plotted as a line (a function representing perfect conditions), while the RSSI data from the files is plotted underneath, in a 
    scatter subplot. The distances are plotted in the x-axis, and the RSSI values are plotted in the y-axis. The distance of two lambda 
    and three lambda are also marked, and set as different colored areas. The RSSI values are plotted in dBm, and the distances are 
    plotted in meters.

    Parameters:
        - filenames: The files to read the RSSI data from. Each file contains data from a specific tag at a specific distance. [list]
        - tag_name: The name of the tag. [str]
        - frequency: The frequency of the signal, in MHz. [float]
        - obstacle: The obstacle between the transmitter and the receiver. [str]
        - power_tx: The power transmitted by the transmitter, in dBm. [float]
        - gain_tx: The gain of the transmitter, in dBiL. [float]
        - gain_rx: The gain of the receiver, in dBiL. [float]
        - real_distances: The real distances between the transmitter and the receiver, in meters. [list]
        - distances: The distances to plot the expected behaviour of the signal. [list]
        - gain_antenna: The gain of the antenna, in dBiL. [float]
        - gain_tag: The gain of the tag, in dBiL. [float]
        - fit: Whether to plot the fitted expected behaviour. [bool]

    Returns:
        - None
    """
        # Use LaTeX for plot typography
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plot_title = f"{tag_name} @ {frequency} MHz, {power_tx} dBm ({round(dBm_to_W(power_tx), 1)}W) - Obstacle: {obstacle}"
    plot_x_axis_title = "Distance [m]"
    plot_y_axis_title = "RSSI [dBm]"

    # Get the expected behaviour of the signal
    two_lambda, three_lambda, expected = expected_behaviour(dBm_to_dB(power_tx), gain_tx, gain_rx, distances, frequency)
    expected_fit = expected_fitted(dBm_to_dB(power_tx), gain_antenna, gain_tag, distances, frequency)

    all_rssi = []
    all_distances = []
    for i, filename in enumerate(filenames):
        data = pd.read_csv(filename)
        all_rssi.extend(data['peakRssi'].values)
        all_distances.extend([real_distances[i]] * len(data))

    # Plot each point
    plt.scatter(all_distances, all_rssi, c='red', label='RSSI')

    if fit:
        # Plot the expected behaviour of the signal
        plt.plot(distances, expected, c='black', label='Expected Behaviour')
        plt.plot(distances, expected_fit, c='red', label='Expected Behaviour (Fitted)')
        # Format the gains to 2 decimal places
        og_antenna_text = f'Antenna Gain: {gain_tx:.2f} dBiL'
        og_tag_text = f'Tag Gain: {gain_rx:.2f} dBiL'
        gain_antenna_text = f'Fitted Antenna Gain: {gain_antenna:.2f} dBiL'
        gain_tag_text = f'Fitted Tag Gain: {gain_tag:.2f} dBiL'
        
        # Define the box properties
        box_props = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='black')
        
        # Place the text in the bottom left corner with a semi-transparent box
        plt.text(0.10, 0.80, f'{og_antenna_text}\n{og_tag_text}\n{gain_antenna_text}\n{gain_tag_text}', transform=plt.gca().transAxes, fontsize=10, verticalalignment='bottom', bbox=box_props)

    plt.axvline(x=two_lambda/8, color='black', linestyle='--', label=r'$\frac{\lambda}{4}$')
    plt.axvline(x=two_lambda/4, color='gray', linestyle='--', label=r'$\frac{\lambda}{2}$')
    plt.axvline(x=two_lambda/2, color='blue', linestyle='--', label=r'$\lambda$')
    plt.axvline(x=two_lambda, color='red', linestyle='--', label=r'$2 \lambda$')
    plt.axvline(x=three_lambda, color='green', linestyle='--', label=r'$3 \lambda$')

    # Set the title and labels
    plt.title(plot_title)
    plt.xlabel(plot_x_axis_title)
    plt.ylabel(plot_y_axis_title)
    plt.legend()
    plt.show()

def plot_phase_distance_all_points(filenames: list, tag_name: str, frequency: float, obstacle: str, power_tx: float, gain_tx: float, gain_rx: float,
                        real_distances: list, distances: list, gain_antenna: float, gain_tag: float, fit: bool = False, individual_markers: bool = True):
    """
    Plot the RSSI data (scatter) from the specified files (CSV) with the expected behaviour of the signal. The expected behaviour is 
    plotted as a line (a function representing perfect conditions), while the RSSI data from the files is plotted underneath, in a 
    scatter subplot. The distances are plotted in the x-axis, and the RSSI values are plotted in the y-axis. The distance of two lambda 
    and three lambda are also marked, and set as different colored areas. The RSSI values are plotted in dBm, and the distances are 
    plotted in meters.

    Parameters:
        - filenames: The files to read the RSSI data from. Each file contains data from a specific tag at a specific distance. [list]
        - tag_name: The name of the tag. [str]
        - frequency: The frequency of the signal, in MHz. [float]
        - obstacle: The obstacle between the transmitter and the receiver. [str]
        - power_tx: The power transmitted by the transmitter, in dBm. [float]
        - gain_tx: The gain of the transmitter, in dBiL. [float]
        - gain_rx: The gain of the receiver, in dBiL. [float]
        - real_distances: The real distances between the transmitter and the receiver, in meters. [list]
        - distances: The distances to plot the expected behaviour of the signal. [list]
        - gain_antenna: The gain of the antenna, in dBiL. [float]
        - gain_tag: The gain of the tag, in dBiL. [float]
        - fit: Whether to plot the fitted expected behaviour. [bool]

    Returns:
        - None
    """
        # Use LaTeX for plot typography
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plot_title = f"{tag_name} @ {frequency} MHz, {power_tx} dBm ({round(dBm_to_W(power_tx), 1)}W) - Obstacle: {obstacle}"
    plot_x_axis_title = "Distance [m]"
    plot_y_axis_title = "Phase [degrees]"

    # Get the expected behaviour of the signal
    two_lambda, three_lambda, expected = expected_behaviour(dBm_to_dB(power_tx), gain_tx, gain_rx, distances, frequency)
    expected_fit = expected_fitted(dBm_to_dB(power_tx), gain_antenna, gain_tag, distances, frequency)

    all_rssi = []
    all_distances = []
    for i, filename in enumerate(filenames):
        data = pd.read_csv(filename)
        all_rssi.extend(data['phase'].values)
        all_distances.extend([real_distances[i]] * len(data))

    # Plot each point
    plt.scatter(all_distances, all_rssi, c='blue', label='Phase')

    if fit:
        # Plot the expected behaviour of the signal
        plt.plot(distances, expected, c='black', label='Expected Behaviour')
        plt.plot(distances, expected_fit, c='red', label='Expected Behaviour (Fitted)')
        # Format the gains to 2 decimal places
        og_antenna_text = f'Antenna Gain: {gain_tx:.2f} dBiL'
        og_tag_text = f'Tag Gain: {gain_rx:.2f} dBiL'
        gain_antenna_text = f'Fitted Antenna Gain: {gain_antenna:.2f} dBiL'
        gain_tag_text = f'Fitted Tag Gain: {gain_tag:.2f} dBiL'
        
        # Define the box properties
        box_props = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='black')
        
        # Place the text in the bottom left corner with a semi-transparent box
        plt.text(0.10, 0.80, f'{og_antenna_text}\n{og_tag_text}\n{gain_antenna_text}\n{gain_tag_text}', transform=plt.gca().transAxes, fontsize=10, verticalalignment='bottom', bbox=box_props)

    plt.axvline(x=two_lambda/8, color='black', linestyle='--', label=r'$\frac{\lambda}{4}$')
    plt.axvline(x=two_lambda/4, color='gray', linestyle='--', label=r'$\frac{\lambda}{2}$')
    plt.axvline(x=two_lambda/2, color='blue', linestyle='--', label=r'$\lambda$')
    plt.axvline(x=two_lambda, color='red', linestyle='--', label=r'$2 \lambda$')
    plt.axvline(x=three_lambda, color='green', linestyle='--', label=r'$3 \lambda$')

    # Set the title and labels
    plt.title(plot_title)
    plt.xlabel(plot_x_axis_title)
    plt.ylabel(plot_y_axis_title)
    plt.legend()
    plt.show()

def rssi_distance_two_obstacle_comparison_all_points(tag_name: str, power_tx: float, gain_tx: float, gain_rx: float, 
                                                       frequency:float, gain_antenna: float, gain_tag: float,
                                                       obstacle_1:str, obstacle_2:str, distances:list,
                                                       distance_1: list, distance_2: list,
                                                       filenames_obstacle1:list, filenames_obstacle_2:list, fit:bool =False):
    # Use LaTeX for plot typography
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plot_title = f"{tag_name} @ {frequency} MHz, {power_tx} dBm ({round(dBm_to_W(power_tx), 1)}W) - VWC Comparison"

    # Subplot of two figures
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))

    # First Subplot - RSSI
    axs[0].set_title(f"{tag_name} @ {frequency} MHz, {power_tx} dBm ({round(dBm_to_W(power_tx), 1)}W) - RSSI Comparison")
    axs[0].set_xlabel("Distance [m]")
    axs[0].set_ylabel("RSSI [dBm]")

    # Get the expected behaviour of the signal
    two_lambda, three_lambda, expected = expected_behaviour(dBm_to_dB(power_tx), gain_tx, gain_rx, distances, frequency)
    expected_fit = expected_fitted(dBm_to_dB(power_tx), gain_antenna, gain_tag, distances, frequency)

    # Obstacle 1
    all_rssi1 = []
    all_distances1 = []
    for i, filename in enumerate(filenames_obstacle1):
        data = pd.read_csv(filename)
        all_rssi1.extend(data['peakRssi'].values)
        all_distances1.extend([distance_1[i]] * len(data))

    # Plot each point
    axs[0].scatter(all_distances1, all_rssi1, c='red', label=obstacle_1)

    # Obstacle 2
    all_rssi2 = []
    all_distances2 = []
    for i, filename in enumerate(filenames_obstacle_2):
        data = pd.read_csv(filename)
        all_rssi2.extend(data['peakRssi'].values)
        all_distances2.extend([distance_2[i]] * len(data))

    # Plot each point
    axs[0].scatter(all_distances2, all_rssi2, c='blue', label=obstacle_2)

    if fit:
        # Plot the expected behaviour of the signal
        axs[0].plot(distances, expected, c='black', label='Expected Behaviour')
        axs[0].plot(distances, expected_fit, c='red', label='Expected Behaviour (Fitted)')
        # Format the gains to 2 decimal places
        og_antenna_text = f'Antenna Gain: {gain_tx:.2f} dBiL'
        og_tag_text = f'Tag Gain: {gain_rx:.2f} dBiL'
        gain_antenna_text = f'Fitted Antenna Gain: {gain_antenna:.2f} dBiL'
        gain_tag_text = f'Fitted Tag Gain: {gain_tag:.2f} dBiL'
        
        # Define the box properties
        box_props = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='black')
        
        # Place the text in the bottom left corner with a semi-transparent box
        axs[0].text(0.10, 0.80, f'{og_antenna_text}\n{og_tag_text}\n{gain_antenna_text}\n{gain_tag_text}', transform=axs[0].transAxes, fontsize=10, verticalalignment='bottom', bbox=box_props)

    axs[0].axvline(x=two_lambda/8, color='black', linestyle='--', label=r'$\frac{\lambda}{4}$')
    axs[0].axvline(x=two_lambda/4, color='gray', linestyle='--', label=r'$\frac{\lambda}{2}$')
    axs[0].axvline(x=two_lambda/2, color='blue', linestyle='--', label=r'$\lambda$')
    axs[0].axvline(x=two_lambda, color='red', linestyle='--', label=r'$2 \lambda$')
    axs[0].axvline(x=three_lambda, color='green', linestyle='--', label=r'$3 \lambda$')

    # Second Subplot - Phase
    axs[1].set_title(f"{tag_name} @ {frequency} MHz, {power_tx} dBm ({round(dBm_to_W(power_tx), 1)}W) - Phase Comparison")
    axs[1].set_xlabel("Distance [m]")
    axs[1].set_ylabel("Phase [degrees]")

    # Obstacle 1
    all_phase1 = []
    all_distances1 = []
    for i, filename in enumerate(filenames_obstacle1):
        data = pd.read_csv(filename)
        all_phase1.extend(data['phase'].values)
        all_distances1.extend([distance_1[i]] * len(data))

    # Plot each point
    axs[1].scatter(all_distances1, all_phase1, c='red', label=obstacle_1)

    # Obstacle 2
    all_phase2 = []
    all_distances2 = []
    for i, filename in enumerate(filenames_obstacle_2):
        data = pd.read_csv(filename)
        all_phase2.extend(data['phase'].values)
        all_distances2.extend([distance_2[i]] * len(data))

    # Plot each point
    axs[1].scatter(all_distances2, all_phase2, c='blue', label=obstacle_2)

    if fit:
        # Plot the expected behaviour of the signal
        axs[1].plot(distances, expected, c='black', label='Expected Behaviour')
        axs[1].plot(distances, expected_fit, c='red', label='Expected Behaviour (Fitted)')
        
        # Plot theoretical phase based on the equation: phase = -(4*pi*frequency*distance/c)
        freq_hz = MHz_to_Hz(frequency)
        theoretical_phase = [np.rad2deg(((-(4 * np.pi * freq_hz * d / sc.speed_of_light)) + np.pi) % (2 * np.pi) - np.pi) for d in distances]
        axs[1].plot(distances, theoretical_phase, c='purple', label='Theoretical Phase')
        
        # Define the box properties
        box_props = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='black')
        
        # Place the text in the bottom left corner with a semi-transparent box
        axs[1].text(0.10, 0.80, f'{og_antenna_text}\n{og_tag_text}\n{gain_antenna_text}\n{gain_tag_text}', transform=axs[1].transAxes, fontsize=10, verticalalignment='bottom', bbox=box_props)

    axs[1].axvline(x=two_lambda/8, color='black', linestyle='--', label=r'$\frac{\lambda}{4}$')
    axs[1].axvline(x=two_lambda/4, color='gray', linestyle='--', label=r'$\frac{\lambda}{2}$')
    axs[1].axvline(x=two_lambda/2, color='blue', linestyle='--', label=r'$\lambda$')
    axs[1].axvline(x=two_lambda, color='red', linestyle='--', label=r'$2 \lambda$')
    axs[1].axvline(x=three_lambda, color='green', linestyle='--', label=r'$3 \lambda$')

    # Combine legends from both subplots
    handles, labels = [], []
    for ax in axs:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=7)

    plt.suptitle(plot_title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def rssi_distance_two_obstacle_comparison_rssi_only(tag_name: str, power_tx: float, gain_tx: float, gain_rx: float, 
                                                    frequency: float, gain_antenna: float, gain_tag: float,
                                                    obstacle_1: str, obstacle_2: str, distances: list,
                                                    distance_1: list, distance_2: list,
                                                    filenames_obstacle1: list, filenames_obstacle_2: list, fit: bool = False):
    # Use LaTeX for plot typography
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plot_title = f"{tag_name} @ {frequency} MHz, {power_tx} dBm ({round(dBm_to_W(power_tx), 1)}W) - RSSI Comparison"

    # Create a single figure
    plt.figure(figsize=(6, 6))

    # Set axis labels
    plt.xlabel("Distance [m]")
    plt.ylabel("RSSI [dBm]")

    # Get the expected behaviour of the signal
    two_lambda, three_lambda, expected = expected_behaviour(dBm_to_dB(power_tx), gain_tx, gain_rx, distances, frequency)
    expected_fit = expected_fitted(dBm_to_dB(power_tx), gain_antenna, gain_tag, distances, frequency)

    # Obstacle 1
    all_rssi1 = []
    all_distances1 = []
    for i, filename in enumerate(filenames_obstacle1):
        data = pd.read_csv(filename)
        all_rssi1.extend(data['peakRssi'].values)
        all_distances1.extend([distance_1[i]] * len(data))

    # Plot each point for obstacle 1
    plt.scatter(all_distances1, all_rssi1, c='red', label=obstacle_1)

    # Obstacle 2
    all_rssi2 = []
    all_distances2 = []
    for i, filename in enumerate(filenames_obstacle_2):
        data = pd.read_csv(filename)
        all_rssi2.extend(data['peakRssi'].values)
        all_distances2.extend([distance_2[i]] * len(data))

    # Plot each point for obstacle 2
    plt.scatter(all_distances2, all_rssi2, c='blue', label=obstacle_2)

    if fit:
        # Plot the expected behaviour of the signal
        plt.plot(distances, expected, c='black', label='Expected Behaviour')
        plt.plot(distances, expected_fit, c='red', label='Expected Behaviour (Fitted)')
        # Format the gains to 2 decimal places
        og_antenna_text = f'Antenna Gain: {gain_tx:.2f} dBiL'
        og_tag_text = f'Tag Gain: {gain_rx:.2f} dBiL'
        gain_antenna_text = f'Fitted Antenna Gain: {gain_antenna:.2f} dBiL'
        gain_tag_text = f'Fitted Tag Gain: {gain_tag:.2f} dBiL'
        
        # Define the box properties
        box_props = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='black')
        
        # Place the text in the bottom left corner with a semi-transparent box
        plt.text(0.10, 0.80, f'{og_antenna_text}\n{og_tag_text}\n{gain_antenna_text}\n{gain_tag_text}', transform=plt.gca().transAxes, fontsize=10, verticalalignment='bottom', bbox=box_props)

    # Add vertical lines for reference
    #plt.axvline(x=two_lambda/8, color='black', linestyle='--', label=r'$\lambda/4$')
    plt.axvline(x=two_lambda/4, color='black', linestyle='--', label=r'$\lambda/2$')
    plt.axvline(x=two_lambda/2, color='blue', linestyle='--', label=r'$\lambda$')
    plt.axvline(x=two_lambda, color='red', linestyle='--', label=r'$2 \lambda$')
    plt.axvline(x=three_lambda, color='green', linestyle='--', label=r'$3 \lambda$')

    # Combine legends and place them underneath the graph
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=6, frameon=True)

    # Set the title
    plt.title(plot_title)

    # Adjust layout to add space between title and legend
    plt.tight_layout(rect=[0, 0, 1, 0.9])

    # Show the plot
    plt.show()

def rssi_distance_three_obstacle_comparison_all_points(tag_name: str, power_tx: float, gain_tx: float, gain_rx: float, 
                                                       frequency:float, gain_antenna: float, gain_tag: float,
                                                       obstacle_1:str, obstacle_2:str, obstacle_3:str, distances:list,
                                                       distance_1: list, distance_2: list, distance_3: list,
                                                       filenames_obstacle1:list, filenames_obstacle_2:list, filenames_obstacle3:list, fit:bool =False):
    # Use LaTeX for plot typography
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plot_title = f"{tag_name} @ {frequency} MHz, {power_tx} dBm ({round(dBm_to_W(power_tx), 1)}W) - Obstacle Comparison"

    # Subplot of two figures
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))

    # First Subplot - RSSI
    axs[0].set_title(f"{tag_name} @ {frequency} MHz, {power_tx} dBm ({round(dBm_to_W(power_tx), 1)}W) - RSSI Comparison")
    axs[0].set_xlabel("Distance [m]")
    axs[0].set_ylabel("RSSI [dBm]")

    # Get the expected behaviour of the signal
    two_lambda, three_lambda, expected = expected_behaviour(dBm_to_dB(power_tx), gain_tx, gain_rx, distances, frequency)
    expected_fit = expected_fitted(dBm_to_dB(power_tx), gain_antenna, gain_tag, distances, frequency)

    # Obstacle 1
    all_rssi1 = []
    all_distances1 = []
    for i, filename in enumerate(filenames_obstacle1):
        data = pd.read_csv(filename)
        all_rssi1.extend(data['peakRssi'].values)
        all_distances1.extend([distance_1[i]] * len(data))

    # Plot each point
    axs[0].scatter(all_distances1, all_rssi1, c='red', label=obstacle_1)

    # Obstacle 2
    all_rssi2 = []
    all_distances2 = []
    for i, filename in enumerate(filenames_obstacle_2):
        data = pd.read_csv(filename)
        all_rssi2.extend(data['peakRssi'].values)
        all_distances2.extend([distance_2[i]] * len(data))

    # Plot each point
    axs[0].scatter(all_distances2, all_rssi2, c='green', label=obstacle_2)

    # Obstacle 3
    all_rssi3 = []
    all_distances3 = []
    for i, filename in enumerate(filenames_obstacle3):
        data = pd.read_csv(filename)
        all_rssi3.extend(data['peakRssi'].values)
        all_distances3.extend([distance_3[i]] * len(data))

    # Plot each point
    axs[0].scatter(all_distances3, all_rssi3, c='blue', label=obstacle_3)

    if fit:
        # Plot the expected behaviour of the signal
        axs[0].plot(distances, expected, c='black', label='Expected Behaviour')
        axs[0].plot(distances, expected_fit, c='red', label='Expected Behaviour (Fitted)')
        # Format the gains to 2 decimal places
        og_antenna_text = f'Antenna Gain: {gain_tx:.2f} dBiL'
        og_tag_text = f'Tag Gain: {gain_rx:.2f} dBiL'
        gain_antenna_text = f'Fitted Antenna Gain: {gain_antenna:.2f} dBiL'
        gain_tag_text = f'Fitted Tag Gain: {gain_tag:.2f} dBiL'
        
        # Define the box properties
        box_props = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='black')
        
        # Place the text in the bottom left corner with a semi-transparent box
        axs[0].text(0.10, 0.80, f'{og_antenna_text}\n{og_tag_text}\n{gain_antenna_text}\n{gain_tag_text}', transform=axs[0].transAxes, fontsize=10, verticalalignment='bottom', bbox=box_props)

    axs[0].axvline(x=two_lambda/8, color='black', linestyle='--', label=r'$\frac{\lambda}{4}$')
    axs[0].axvline(x=two_lambda/4, color='gray', linestyle='--', label=r'$\frac{\lambda}{2}$')
    axs[0].axvline(x=two_lambda/2, color='blue', linestyle='--', label=r'$\lambda$')
    axs[0].axvline(x=two_lambda, color='red', linestyle='--', label=r'$2 \lambda$')
    axs[0].axvline(x=three_lambda, color='green', linestyle='--', label=r'$3 \lambda$')

    # Second Subplot - Phase
    axs[1].set_title(f"{tag_name} @ {frequency} MHz, {power_tx} dBm ({round(dBm_to_W(power_tx), 1)}W) - Phase Comparison")
    axs[1].set_xlabel("Distance [m]")
    axs[1].set_ylabel("Phase [degrees]")

    # Obstacle 1
    all_phase1 = []
    all_distances1 = []
    for i, filename in enumerate(filenames_obstacle1):
        data = pd.read_csv(filename)
        all_phase1.extend(data['phase'].values)
        all_distances1.extend([distance_1[i]] * len(data))

    # Plot each point
    axs[1].scatter(all_distances1, all_phase1, c='red', label=obstacle_1)

    # Obstacle 2
    all_phase2 = []
    all_distances2 = []
    for i, filename in enumerate(filenames_obstacle_2):
        data = pd.read_csv(filename)
        all_phase2.extend(data['phase'].values)
        all_distances2.extend([distance_2[i]] * len(data))

    # Plot each point
    axs[1].scatter(all_distances2, all_phase2, c='green', label=obstacle_2)

    # Obstacle 3
    all_phase3 = []
    all_distances3 = []
    for i, filename in enumerate(filenames_obstacle3):
        data = pd.read_csv(filename)
        all_phase3.extend(data['phase'].values)
        all_distances3.extend([distance_3[i]] * len(data))

    # Plot each point
    axs[1].scatter(all_distances3, all_phase3, c='blue', label=obstacle_3)

    if fit:
        # Plot the expected behaviour of the signal
        axs[1].plot(distances, expected, c='black', label='Expected Behaviour')
        axs[1].plot(distances, expected_fit, c='red', label='Expected Behaviour (Fitted)')
        # Format the gains to 2 decimal places
        og_antenna_text = f'Antenna Gain: {gain_tx:.2f} dBiL'
        og_tag_text = f'Tag Gain: {gain_rx:.2f} dBiL'
        gain_antenna_text = f'Fitted Antenna Gain: {gain_antenna:.2f} dBiL'
        gain_tag_text = f'Fitted Tag Gain: {gain_tag:.2f} dBiL'
        
        # Define the box properties
        box_props = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='black')
        
        # Place the text in the bottom left corner with a semi-transparent box
        axs[1].text(0.10, 0.80, f'{og_antenna_text}\n{og_tag_text}\n{gain_antenna_text}\n{gain_tag_text}', transform=axs[1].transAxes, fontsize=10, verticalalignment='bottom', bbox=box_props)

    axs[1].axvline(x=two_lambda/8, color='black', linestyle='--', label=r'$\frac{\lambda}{4}$')
    axs[1].axvline(x=two_lambda/4, color='gray', linestyle='--', label=r'$\frac{\lambda}{2}$')
    axs[1].axvline(x=two_lambda/2, color='blue', linestyle='--', label=r'$\lambda$')
    axs[1].axvline(x=two_lambda, color='red', linestyle='--', label=r'$2 \lambda$')
    axs[1].axvline(x=three_lambda, color='green', linestyle='--', label=r'$3 \lambda$')

    # Combine legends from both subplots
    handles, labels = [], []
    for ax in axs:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=8)

    plt.suptitle(plot_title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def rssi_distance_four_obstacle_comparison_all_points(tag_name: str, power_tx: float, gain_tx: float, gain_rx: float, 
                                                       frequency:float, gain_antenna: float, gain_tag: float,
                                                       obstacle_1:str, obstacle_2:str, obstacle_3:str, obstacle_4:str, distances:list,
                                                       distance_1: list, distance_2: list, distance_3: list, distance_4: list,
                                                       filenames_obstacle1:list, filenames_obstacle_2:list, filenames_obstacle3:list,
                                                       filenames_obstacle4:list, fit:bool =False):
    # Use LaTeX for plot typography
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plot_title = f"{tag_name} @ {frequency} MHz, {power_tx} dBm ({round(dBm_to_W(power_tx), 1)}W) - Obstacle Comparison"

    # Subplot of two figures
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))

    # First Subplot - RSSI
    axs[0].set_title(f"{tag_name} @ {frequency} MHz, {power_tx} dBm ({round(dBm_to_W(power_tx), 1)}W) - RSSI Comparison")
    axs[0].set_xlabel("Distance [m]")
    axs[0].set_ylabel("RSSI [dBm]")

    # Get the expected behaviour of the signal
    two_lambda, three_lambda, expected = expected_behaviour(dBm_to_dB(power_tx), gain_tx, gain_rx, distances, frequency)
    expected_fit = expected_fitted(dBm_to_dB(power_tx), gain_antenna, gain_tag, distances, frequency)

    # Obstacle 1
    all_rssi1 = []
    all_distances1 = []
    for i, filename in enumerate(filenames_obstacle1):
        data = pd.read_csv(filename)
        all_rssi1.extend(data['peakRssi'].values)
        all_distances1.extend([distance_1[i]] * len(data))

    # Plot each point
    axs[0].scatter(all_distances1, all_rssi1, c='red', label=obstacle_1)

    # Obstacle 2
    all_rssi2 = []
    all_distances2 = []
    for i, filename in enumerate(filenames_obstacle_2):
        data = pd.read_csv(filename)
        all_rssi2.extend(data['peakRssi'].values)
        all_distances2.extend([distance_2[i]] * len(data))

    # Plot each point
    axs[0].scatter(all_distances2, all_rssi2, c='green', label=obstacle_2)

    # Obstacle 3
    all_rssi3 = []
    all_distances3 = []
    for i, filename in enumerate(filenames_obstacle3):
        data = pd.read_csv(filename)
        all_rssi3.extend(data['peakRssi'].values)
        all_distances3.extend([distance_3[i]] * len(data))

    # Plot each point
    axs[0].scatter(all_distances3, all_rssi3, c='blue', label=obstacle_3)

    # Obstacle 4
    all_rssi4 = []
    all_distances4 = []
    for i, filename in enumerate(filenames_obstacle4):
        data = pd.read_csv(filename)
        all_rssi4.extend(data['peakRssi'].values)
        all_distances4.extend([distance_4[i]] * len(data))

    # Plot each point
    axs[0].scatter(all_distances4, all_rssi4, c='brown', label=obstacle_4)

    if fit:
        # Plot the expected behaviour of the signal
        axs[0].plot(distances, expected, c='black', label='Expected Behaviour')
        axs[0].plot(distances, expected_fit, c='red', label='Expected Behaviour (Fitted)')
        # Format the gains to 2 decimal places
        og_antenna_text = f'Antenna Gain: {gain_tx:.2f} dBiL'
        og_tag_text = f'Tag Gain: {gain_rx:.2f} dBiL'
        gain_antenna_text = f'Fitted Antenna Gain: {gain_antenna:.2f} dBiL'
        gain_tag_text = f'Fitted Tag Gain: {gain_tag:.2f} dBiL'
        
        # Define the box properties
        box_props = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='black')
        
        # Place the text in the bottom left corner with a semi-transparent box
        axs[0].text(0.10, 0.80, f'{og_antenna_text}\n{og_tag_text}\n{gain_antenna_text}\n{gain_tag_text}', transform=axs[0].transAxes, fontsize=10, verticalalignment='bottom', bbox=box_props)

    axs[0].axvline(x=two_lambda/8, color='black', linestyle='--', label=r'$\frac{\lambda}{4}$')
    axs[0].axvline(x=two_lambda/4, color='gray', linestyle='--', label=r'$\frac{\lambda}{2}$')
    axs[0].axvline(x=two_lambda/2, color='blue', linestyle='--', label=r'$\lambda$')
    axs[0].axvline(x=two_lambda, color='red', linestyle='--', label=r'$2 \lambda$')
    axs[0].axvline(x=three_lambda, color='green', linestyle='--', label=r'$3 \lambda$')

    # Second Subplot - Phase
    axs[1].set_title(f"{tag_name} @ {frequency} MHz, {power_tx} dBm ({round(dBm_to_W(power_tx), 1)}W) - Phase Comparison")
    axs[1].set_xlabel("Distance [m]")
    axs[1].set_ylabel("Phase [degrees]")

    # Obstacle 1
    all_phase1 = []
    all_distances1 = []
    for i, filename in enumerate(filenames_obstacle1):
        data = pd.read_csv(filename)
        all_phase1.extend(data['phase'].values)
        all_distances1.extend([distance_1[i]] * len(data))

    # Plot each point
    axs[1].scatter(all_distances1, all_phase1, c='red', label=obstacle_1)

    # Obstacle 2
    all_phase2 = []
    all_distances2 = []
    for i, filename in enumerate(filenames_obstacle_2):
        data = pd.read_csv(filename)
        all_phase2.extend(data['phase'].values)
        all_distances2.extend([distance_2[i]] * len(data))

    # Plot each point
    axs[1].scatter(all_distances2, all_phase2, c='green', label=obstacle_2)

    # Obstacle 3
    all_phase3 = []
    all_distances3 = []
    for i, filename in enumerate(filenames_obstacle3):
        data = pd.read_csv(filename)
        all_phase3.extend(data['phase'].values)
        all_distances3.extend([distance_3[i]] * len(data))

    # Plot each point
    axs[1].scatter(all_distances3, all_phase3, c='blue', label=obstacle_3)

    # Obstacle 4
    all_phase4 = []
    all_distances4 = []
    for i, filename in enumerate(filenames_obstacle4):
        data = pd.read_csv(filename)
        all_phase4.extend(data['phase'].values)
        all_distances4.extend([distance_4[i]] * len(data))
    
    # Plot each point
    axs[1].scatter(all_distances4, all_phase4, c='brown', label=obstacle_4)

    if fit:
        # Plot the expected behaviour of the signal
        axs[1].plot(distances, expected, c='black', label='Expected Behaviour')
        axs[1].plot(distances, expected_fit, c='red', label='Expected Behaviour (Fitted)')
        # Format the gains to 2 decimal places
        og_antenna_text = f'Antenna Gain: {gain_tx:.2f} dBiL'
        og_tag_text = f'Tag Gain: {gain_rx:.2f} dBiL'
        gain_antenna_text = f'Fitted Antenna Gain: {gain_antenna:.2f} dBiL'
        gain_tag_text = f'Fitted Tag Gain: {gain_tag:.2f} dBiL'
        
        # Define the box properties
        box_props = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='black')
        
        # Place the text in the bottom left corner with a semi-transparent box
        axs[1].text(0.10, 0.80, f'{og_antenna_text}\n{og_tag_text}\n{gain_antenna_text}\n{gain_tag_text}', transform=axs[1].transAxes, fontsize=10, verticalalignment='bottom', bbox=box_props)

    axs[1].axvline(x=two_lambda/8, color='black', linestyle='--', label=r'$\frac{\lambda}{4}$')
    axs[1].axvline(x=two_lambda/4, color='gray', linestyle='--', label=r'$\frac{\lambda}{2}$')
    axs[1].axvline(x=two_lambda/2, color='blue', linestyle='--', label=r'$\lambda$')
    axs[1].axvline(x=two_lambda, color='red', linestyle='--', label=r'$2 \lambda$')
    axs[1].axvline(x=three_lambda, color='green', linestyle='--', label=r'$3 \lambda$')

    # Combine legends from both subplots
    handles, labels = [], []
    for ax in axs:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=9)

    plt.suptitle(plot_title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def rssi_distance_three_power_comparison_all_points(tag_name: str, power_tx: float, gain_tx: float, gain_rx: float, 
                                                       frequency:float, gain_antenna: float, gain_tag: float,
                                                       power_1:float, power_2:float, power_3:float, distances:list,
                                                       distance_1: list, distance_2: list, distance_3: list,
                                                       filenames_power1:list, filenames_power2:list, filenames_power3:list):
    # Use LaTeX for plot typography
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plot_title = f"{tag_name} @ {frequency} MHz - Power Comparison"

    # Subplot of two figures
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))

    # First Subplot - RSSI
    axs[0].set_title(f"{tag_name} @ {frequency} MHz - RSSI Comparison")
    axs[0].set_xlabel("Distance [m]")
    axs[0].set_ylabel("RSSI [dBm]")

    # Get the expected behaviour of the signal
    two_lambda, three_lambda, expected = expected_behaviour(dBm_to_dB(power_tx), gain_tx, gain_rx, distances, frequency)

    # Power 1
    power_1_name = f"{power_1} dBm ({round(dBm_to_W(power_1), 1)}W)"
    all_rssi1 = []
    all_distances1 = []
    for i, filename in enumerate(filenames_power1):
        data = pd.read_csv(filename)
        all_rssi1.extend(data['peakRssi'].values)
        all_distances1.extend([distance_1[i]] * len(data))

    # Plot each point
    axs[0].scatter(all_distances1, all_rssi1, c='red', label=power_1_name)

    # Power 2
    power_2_name = f"{power_2} dBm ({round(dBm_to_W(power_2), 1)}W)"
    all_rssi2 = []
    all_distances2 = []
    for i, filename in enumerate(filenames_power2):
        data = pd.read_csv(filename)
        all_rssi2.extend(data['peakRssi'].values)
        all_distances2.extend([distance_2[i]] * len(data))

    # Plot each point
    axs[0].scatter(all_distances2, all_rssi2, c='green', label=power_2_name)

    # Power 3
    power_3_name = f"{power_3} dBm ({round(dBm_to_W(power_3), 1)}W)"
    all_rssi3 = []
    all_distances3 = []
    for i, filename in enumerate(filenames_power3):
        data = pd.read_csv(filename)
        all_rssi3.extend(data['peakRssi'].values)
        all_distances3.extend([distance_3[i]] * len(data))

    # Plot each point
    axs[0].scatter(all_distances3, all_rssi3, c='blue', label=power_3_name)

    axs[0].axvline(x=two_lambda/8, color='black', linestyle='--', label=r'$\frac{\lambda}{4}$')
    axs[0].axvline(x=two_lambda/4, color='gray', linestyle='--', label=r'$\frac{\lambda}{2}$')
    axs[0].axvline(x=two_lambda/2, color='blue', linestyle='--', label=r'$\lambda$')
    axs[0].axvline(x=two_lambda, color='red', linestyle='--', label=r'$2 \lambda$')
    axs[0].axvline(x=three_lambda, color='green', linestyle='--', label=r'$3 \lambda$')

    # Second Subplot - Phase
    axs[1].set_title(f"{tag_name} @ {frequency} MHz, {power_tx} dBm ({round(dBm_to_W(power_tx), 1)}W) - Phase Comparison")
    axs[1].set_xlabel("Distance [m]")
    axs[1].set_ylabel("Phase [degrees]")

    # Power 1
    all_phase1 = []
    all_distances1 = []
    for i, filename in enumerate(filenames_power1):
        data = pd.read_csv(filename)
        all_phase1.extend(data['phase'].values)
        all_distances1.extend([distance_1[i]] * len(data))

    # Plot each point
    axs[1].scatter(all_distances1, all_phase1, c='red', label=power_1_name)

    # Power 2
    all_phase2 = []
    all_distances2 = []
    for i, filename in enumerate(filenames_power2):
        data = pd.read_csv(filename)
        all_phase2.extend(data['phase'].values)
        all_distances2.extend([distance_2[i]] * len(data))

    # Plot each point
    axs[1].scatter(all_distances2, all_phase2, c='green', label=power_2_name)

    # Power 3
    all_phase3 = []
    all_distances3 = []
    for i, filename in enumerate(filenames_power3):
        data = pd.read_csv(filename)
        all_phase3.extend(data['phase'].values)
        all_distances3.extend([distance_3[i]] * len(data))

    # Plot each point
    axs[1].scatter(all_distances3, all_phase3, c='blue', label=power_3_name)

    axs[1].axvline(x=two_lambda/8, color='black', linestyle='--', label=r'$\frac{\lambda}{4}$')
    axs[1].axvline(x=two_lambda/4, color='gray', linestyle='--', label=r'$\frac{\lambda}{2}$')
    axs[1].axvline(x=two_lambda/2, color='blue', linestyle='--', label=r'$\lambda$')
    axs[1].axvline(x=two_lambda, color='red', linestyle='--', label=r'$2 \lambda$')
    axs[1].axvline(x=three_lambda, color='green', linestyle='--', label=r'$3 \lambda$')

    # Combine legends from both subplots
    handles, labels = [], []
    for ax in axs:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=8)

    plt.suptitle(plot_title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
# =================================================================================================================================== #


# =================================================================================================================================== #
# ----------------------------------------------------- GAUSSIAN MIXTURE MODELS ----------------------------------------------------- #
def plot_rssi_phase_obstacles(power_tx: float, frequency:float, obstacle_list:list, distances:list, filenames_list:list):
    #  Use LaTeX for plot typography
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Extract distance range
    distance_start = distances[0]
    distance_end = distances[-1]

    plot_title = f"{power_tx} dBm ({round(dBm_to_W(power_tx), 1)}W) @ {frequency} MHz ({round(distance_start, 2)}m - {round(distance_end, 2)}m)"
    plot_x_axis_title = "Phase [degrees]"
    plot_y_axis_title = "RSSI [dBm]"

    cmap = plt.get_cmap('tab10')

    # For each obstacle, plot the RSSI and Phase
    for idx, obstacle in enumerate(obstacle_list):
        all_rssi = []
        all_phase = []
        for filename in filenames_list[idx]:
            try:
                data = pd.read_csv(filename)
                all_rssi.extend(data['peakRssi'].values)
                all_phase.extend(data['phase'].values)
            except FileNotFoundError:
                print(f"File not found: {filename}")
            except pd.errors.EmptyDataError:
                print(f"Empty data in file: {filename}")
            except KeyError:
                print(f"Missing expected columns in file: {filename}")
        
        # Plot each point
        plt.scatter(all_phase, all_rssi, color=cmap(idx), label=obstacle)
        
    # Set the title and labels
    plt.title(plot_title)
    plt.xlabel(plot_x_axis_title)
    plt.ylabel(plot_y_axis_title)
    plt.legend()
    plt.show()
# =================================================================================================================================== #


# =================================================================================================================================== #
# --------------------------------------------------------- ANGLE OF ARRIVAL -------------------------------------------------------- #
def plot_aoa_phasedif(tag_name: str, frequency: float, distance: float, distance_antennas: float, power_tx: float, widths: list, filenames: list, tag_id: str):       
    # --------------------------------- PARAMETERS ------------------------------- #
    c   = sc.c                     # speed of light [m/s]                          #
    f   = MHz_to_Hz(frequency)     # carrier frequency [Hz]                        #
    lam = c / f                    # wavelength ≈ 0.3465 m                         #
    L   = distance_antennas        # antenna separation [m]                        #
    D   = distance                 # fixed tag‐to‐antenna distance [m]             #
    delta0_tx = 0.0                # Force Δϕ₀ = 0 for the symmetric antenna case  #
    delta0_tag = 0.0               # Force Δϕ₀ = 0 for the symmetric antenna case  #
    aoa_experimental = []          # Storage for experimental AoA (θ)              #
    phi_experimental_tx = []       # Storage for experimental phase diff (from Tx) #
    phi_experimental_tag = []      # Storage for experimental phase diff (from Tag)#
    theta_calc_tx = []             # Storage for calculated AoA (θ) (from Tx)      #
    theta_calc_tag = []            # Storage for calculated AoA (θ) (from Tag)     #
    mean_pwr_antenna1 = []         # Storage for mean power (from Tx) (Antenna 1)  #
    std_pwr_antenna1 = []          # Storage for std power (from Tx) (Antenna 1)   #
    mean_pwr_antenna2 = []         # Storage for mean power (from Tx) (Antenna 2)  #
    std_pwr_antenna2 = []          # Storage for std power (from Tx) (Antenna 2)   #
    # ---------------------------------------------------------------------------- #

    # ------------------------------- MEASUREMENTS ------------------------------- #
    # The measured widths are the horizontal distance between the center of the    #
    # two antennas and the tag, placed at a fixed vertical distance D.             #
    WIDTHS = widths     # Measured widths [m]                                      #
    FILES  = filenames  # CSV files                                                # 
    TAG    = tag_id     # Tag ID (hex)                                             #
    # ---------------------------------------------------------------------------- #

    # -------------------------------- HW OFFSET --------------------------------- #
    try:
        delta0_tx = dm.phi_hw_offset_tx(filenames[0], TAG)
        delta0_tag = dm.phi_hw_offset_tag(filenames[0], TAG)
        
        if np.isnan(delta0_tx) or np.isnan(delta0_tag):
            print(f"Warning: NaN hardware offset. Using defaults (0.0).")
            delta0_tx = 0.0
            delta0_tag = 0.0
    except Exception as e:
        print(f"Error calculating hardware offset: {e}. Using defaults (0.0).")
        delta0_tx = 0.0
        delta0_tag = 0.0
    # ---------------------------------------------------------------------------- #

    # ------------------------------ DATA PROCESSING ----------------------------- #
    valid_measurements = []  # To keep track of valid indices
    for i, (w, fn) in enumerate(zip(WIDTHS, FILES)):
        df = pd.read_csv(fn)

        # Filter out non‐tag measurements
        df['idHex'] = df['idHex'].astype(str).str.strip()
        tagdf = df[df['idHex'] == TAG]
        
        # Check if both antennas have readings
        antenna1_readings = tagdf[tagdf['antenna']==1]
        antenna2_readings = tagdf[tagdf['antenna']==2]
        
        if len(antenna1_readings) == 0 or len(antenna2_readings) == 0:
            print(f"Warning: Missing antenna readings in file {fn}. Skipping this measurement.")
            continue
        
        # Phase = from tag
        m1_tag = (dm.circular_mean_deg(antenna1_readings['phase']))/2
        m2_tag = (dm.circular_mean_deg(antenna2_readings['phase']))/2

        # Phase = from transceiver
        m1_tx = (dm.circular_mean_deg(antenna1_readings['phase']))
        m2_tx = (dm.circular_mean_deg(antenna2_readings['phase']))
        
        # Check for NaN values
        if np.isnan(m1_tag) or np.isnan(m2_tag) or np.isnan(m1_tx) or np.isnan(m2_tx):
            print(f"Warning: NaN phase values in file {fn}. Skipping this measurement.")
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
        mean_pwr_antenna1.append(np.mean(antenna1_readings['peakRssi'].values))
        std_pwr_antenna1.append(np.std(antenna1_readings['peakRssi'].values))
        mean_pwr_antenna2.append(np.mean(antenna2_readings['peakRssi'].values))
        std_pwr_antenna2.append(np.std(antenna2_readings['peakRssi'].values))
        
        # Keep track of valid measurement index
        valid_measurements.append(i)
    # ---------------------------------------------------------------------------- #

    # ------------------------ THEORETICAL CURVE (Δϕ₀ = 0) ----------------------- #
    θ_range = np.linspace(min(aoa_experimental)-5, max(aoa_experimental)+5, 200)   #
    phi_th_tag = np.degrees((2*np.pi * L / lam) * np.sin(np.radians(θ_range))) - delta0_tag
    phi_th_tx  = np.degrees((2*np.pi * L / lam) * np.sin(np.radians(θ_range))) - delta0_tx
    # ---------------------------------------------------------------------------- #

    # ----------------------------------- PLOTS ---------------------------------- #
    #  Use LaTeX for plot typography
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Angle of Arrival Analysis: {tag_name} @ {frequency} MHz, {power_tx} dBm", fontsize=16)

    # Subplot (1,1): Phase difference vs angle of arrival, from tx
    axs[0, 0].plot(θ_range, phi_th_tx, 'g--', label=r'Theoretical $\Delta \varphi$')
    axs[0, 0].plot(aoa_experimental, phi_experimental_tx, 'bo', label='Experimental')
    axs[0, 0].set_title(r"$\Delta \varphi$ vs. AoA (From Tx)")
    axs[0, 0].set_xlabel(r"Angle of Arrival $\theta$ [degrees]")
    axs[0, 0].set_ylabel(r"Phase difference $\Delta \varphi$ [degrees]")
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    box_props = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='black')
    axs[0, 0].text(0.05, 0.95, f"L = {L} m\nf = {frequency} MHz\nD = {D} m", 
                transform=axs[0, 0].transAxes, fontsize=10, verticalalignment='top', bbox=box_props)

    # Subplot (1,2): Phase difference vs angle of arrival, from tag
    axs[0, 1].plot(θ_range, phi_th_tag, 'g--', label=r'Theoretical $\Delta \varphi$')
    axs[0, 1].plot(aoa_experimental, phi_experimental_tag, 'bo', label='Experimental')
    axs[0, 1].set_title(r"$\Delta \varphi$ vs. AoA (From Tag)")
    axs[0, 1].set_xlabel(r"Angle of Arrival $\theta$ [degrees]")
    axs[0, 1].set_ylabel(r"Phase difference $\Delta \varphi$ [degrees]")
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    axs[0, 1].text(0.05, 0.95, f"L = {L} m\nf = {frequency} MHz\nD = {D} m", 
                transform=axs[0, 1].transAxes, fontsize=10, verticalalignment='top', bbox=box_props)

    # Subplot (2,1): Calculated theta vs experimental theta, from tx
    axs[1, 0].plot(aoa_experimental, aoa_experimental, 'g--', label='Ideal (y=x)')
    axs[1, 0].plot(aoa_experimental, theta_calc_tx, 'ro', label='Estimated')
    axs[1, 0].set_title(r"Estimated vs. Actual Angle (From Tx)")
    axs[1, 0].set_xlabel(r"Actual Angle $\theta$ [degrees]")
    axs[1, 0].set_ylabel(r"Estimated Angle $\theta$ [degrees]")
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    # Calculate RMSE for tx
    rmse_tx = np.sqrt(np.mean((np.array(theta_calc_tx) - np.array(aoa_experimental))**2))
    axs[1, 0].text(0.05, 0.95, f"RMSE = {rmse_tx:.2f}°", transform=axs[1, 0].transAxes, 
                fontsize=10, verticalalignment='top', bbox=box_props)

    # Subplot (2,2): Calculated theta vs experimental theta, from tag
    axs[1, 1].plot(aoa_experimental, aoa_experimental, 'g--', label='Ideal (y=x)')
    axs[1, 1].plot(aoa_experimental, theta_calc_tag, 'ro', label='Estimated')
    axs[1, 1].set_title(r"Estimated vs. Actual Angle (From Tag)")
    axs[1, 1].set_xlabel(r"Actual Angle $\theta$ [degrees]")
    axs[1, 1].set_ylabel(r"Estimated Angle $\theta$ [degrees]")
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    
    # Calculate RMSE for tag
    rmse_tag = np.sqrt(np.mean((np.array(theta_calc_tag) - np.array(aoa_experimental))**2))
    axs[1, 1].text(0.05, 0.95, f"RMSE = {rmse_tag:.2f}°", transform=axs[1, 1].transAxes, 
                fontsize=10, verticalalignment='top', bbox=box_props)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.show()

    # ----------------------------- DETAILED SUMMARY ---------------------------- #
    # Calculate error statistics
    tx_error = np.array(theta_calc_tx) - np.array(aoa_experimental)
    tag_error = np.array(theta_calc_tag) - np.array(aoa_experimental)

    separator = "=" * 100
    header = "\033[1;36m{}\033[0m"  # Cyan, bold
    subheader = "\033[1;33m{}\033[0m"  # Yellow, bold
    value = "\033[0;32m{}\033[0m"  # Green

    print(separator)
    print(header.format(f"AOA MEASUREMENT SUMMARY: {tag_name}"))
    print(separator)
    print(f"Tag ID: {value.format(tag_id)}")
    print(f"Frequency: {value.format(f'{frequency} MHz')}")
    print(f"Power: {value.format(f'{power_tx} dBm ({round(dBm_to_W(power_tx), 1)}W)')}")
    print(f"Tag-to-Antenna Distance (D): {value.format(f'{distance} m')}")
    print(f"Antenna Separation (L): {value.format(f'{distance_antennas} m')}")
    print(f"Wavelength (λ): {value.format(f'{lam:.4f} m')}")
    print(separator)
    
    print(subheader.format("HARDWARE PHASE OFFSET"))
    print(f"Tx Offset (Δϕ₀): {value.format(f'{delta0_tx:.2f}°')}")
    print(f"Tag Offset (Δϕ₀): {value.format(f'{delta0_tag:.2f}°')}")
    print(separator)
    
    print(subheader.format("ANTENNA 1 RSSI"))
    print(f"Mean Power: {value.format(f'{np.mean(mean_pwr_antenna1):.2f} dBm')}")
    print(f"Std Power: {value.format(f'{np.mean(std_pwr_antenna1):.2f} dBm')}")
    print(separator)
    
    print(subheader.format("ANTENNA 2 RSSI"))
    print(f"Mean Power: {value.format(f'{np.mean(mean_pwr_antenna2):.2f} dBm')}")
    print(f"Std Power: {value.format(f'{np.mean(std_pwr_antenna2):.2f} dBm')}")
    print(separator)
    
    print(subheader.format("PHASE DIFFERENCE"))
    print(f"Mean Phase Difference (From Tx): {value.format(f'{np.mean(phi_experimental_tx):.2f}°')}")
    print(f"Mean Phase Difference (From Tag): {value.format(f'{np.mean(phi_experimental_tag):.2f}°')}")
    print(separator)
    
    print(subheader.format("ANGLE OF ARRIVAL ESTIMATION"))
    print(f"Mean Estimated AoA (From Tx): {value.format(f'{np.mean(theta_calc_tx):.2f}°')}")
    print(f"Mean Estimated AoA (From Tag): {value.format(f'{np.mean(theta_calc_tag):.2f}°')}")
    print(separator)
    
    print(subheader.format("ESTIMATION ERROR STATISTICS"))
    print(f"RMSE (From Tx): {value.format(f'{rmse_tx:.2f}°')}")
    print(f"RMSE (From Tag): {value.format(f'{rmse_tag:.2f}°')}")
    print(f"Mean Error (From Tx): {value.format(f'{np.mean(tx_error):.2f}°')}")
    print(f"Mean Error (From Tag): {value.format(f'{np.mean(tag_error):.2f}°')}")
    print(f"Max Error (From Tx): {value.format(f'{np.max(np.abs(tx_error)):.2f}°')}")
    print(f"Max Error (From Tag): {value.format(f'{np.max(np.abs(tag_error)):.2f}°')}")
    print(separator)
    
    # Print detailed measurements in a table format
    print(subheader.format("DETAILED MEASUREMENTS"))

    # Report which measurements were skipped
    if len(valid_measurements) < len(WIDTHS):
        skipped_indices = [i for i in range(len(WIDTHS)) if i not in valid_measurements]
        print(f"Warning: Skipped measurements at indices: {skipped_indices}")
        print(f"Corresponding widths: {[WIDTHS[i] for i in skipped_indices]}")
        print(f"Corresponding files: {[FILES[i] for i in skipped_indices]}")

    # Create a table with only the valid measurements
    measurements = []
    for i, valid_idx in enumerate(valid_measurements):
        measurements.append({
            'Width (m)': f"{WIDTHS[valid_idx]:.3f}",
            'AoA Exp (°)': f"{aoa_experimental[i]:.2f}",
            'Phase Tx (°)': f"{phi_experimental_tx[i]:.2f}",
            'Phase Tag (°)': f"{phi_experimental_tag[i]:.2f}",
            'AoA Tx (°)': f"{theta_calc_tx[i]:.2f}",
            'AoA Tag (°)': f"{theta_calc_tag[i]:.2f}",
            'Error Tx (°)': f"{tx_error[i]:.2f}",
            'Error Tag (°)': f"{tag_error[i]:.2f}"
        })
    
    # Convert to DataFrame for nice table formatting
    df_measurements = pd.DataFrame(measurements)
    print(df_measurements.to_string(index=False))
    print(separator)
    
    # ----------------------- ADDITIONAL STANDALONE PLOT ----------------------- #
    # Create a standalone figure with just the tag angle estimation plot
    plt.figure(figsize=(10, 8))
    plt.plot(aoa_experimental, aoa_experimental, 'g--', label='Ideal (y=x)')
    plt.plot(aoa_experimental, theta_calc_tag, 'ro', label='Estimated')
    plt.title(f"Estimated vs. Actual Angle (From Tag)\n{tag_name} @ {frequency} MHz", fontsize=14)
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
    plt.show()
    # ------------------------------------------------------------------------- #
    
    return {
        'aoa_experimental': aoa_experimental,
        'phi_experimental_tx': phi_experimental_tx,
        'phi_experimental_tag': phi_experimental_tag,
        'theta_calc_tx': theta_calc_tx,
        'theta_calc_tag': theta_calc_tag,
        'rmse_tx': rmse_tx,
        'rmse_tag': rmse_tag
    }
# =================================================================================================================================== #

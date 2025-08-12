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
import seaborn as sns                            # Statistical data visualization based on matplotlib.                                #
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting.                                                                               #
import scipy.constants as sc  # Physical and mathematical constants.                                                                  #
import numpy as np  # Mathematical functions.                                                                                         #
import src.data_management as dm # Data management functions.                                                                         #
# =================================================================================================================================== #


# =================================================================================================================================== #
# ---------------------------------------------------------- PLOTTING SETTINGS ------------------------------------------------------ #
plt.style.use("seaborn-v0_8-whitegrid")                                                                                               #
plt.rcParams.update({                                                                                                                 #
    "font.size"       : 10,        # Base font size for all text in the plot.                                                         #
    "axes.titlesize"  : 12,        # Title size for axes.                                                                             #
    "axes.labelsize"  : 12,        # Axis labels size.                                                                                #
    "xtick.labelsize" : 12,        # X-axis tick labels size.                                                                         #
    "ytick.labelsize" : 12,        # Y-axis tick labels size.                                                                         #
    "legend.fontsize" : 14,        # Legend font size for all text in the legend.                                                     #
    "figure.titlesize": 14,        # Overall figure title size for all text in the figure.                                            #
})                                                                                                                                    #
sns.set_theme(style="whitegrid", context="paper") # Set seaborn theme for additional aesthetics and context.                          #
plt.rcParams["figure.figsize"] = (6, 4)  # Set default figure size for all plots to 6x4.                                              #
TAG_NAME = "Belt DEE"             # Default tag name for the analysis.                                                                #
# =================================================================================================================================== #
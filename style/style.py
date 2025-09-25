# =================================================================================================================================== #
# ----------------------------------------------------------- DESCRIPTION ----------------------------------------------------------- #
# Contains global plotting styles and configurations for consistent aesthetics across all visualizations.                             #
# =================================================================================================================================== #


# =================================================================================================================================== #
# --------------------------------------------------------- EXTERNAL IMPORTS -------------------------------------------------------- #
import seaborn                 as sns                     # For enhanced plotting aesthetics.                                         #
import matplotlib              as mpl                     # For plotting settings.                                                    #
import matplotlib.pyplot       as plt                     # For plotting.                                                             #
from   cycler                  import cycler              # For custom matplotlib color cycles.                                       # 
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
    "figure.dpi": 300,                                                                                                                #
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
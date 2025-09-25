# =================================================================================================================================== #
# ----------------------------------------------------------- DESCRIPTION ----------------------------------------------------------- #
# This script provides functions to visualize RFID Angle of Arrival (AoA) analysis results.                                           #               
# =================================================================================================================================== #


# =================================================================================================================================== #
# --------------------------------------------------------- EXTERNAL IMPORTS -------------------------------------------------------- #
import pandas as pd                       # Data manipulation and analysis.                                                           #
import matplotlib.pyplot as plt           # Data visualization.                                                                       #
import seaborn as sns                     # Statistical data visualization based on matplotlib.                                       #
import numpy as np                        # Mathematical functions.                                                                   #
import os                                 # Operating system dependent functionality.                                                 #
import main                               # Main module containing data manager and analysis functions.                               #
from matplotlib.gridspec import GridSpec  # Flexible grid layout for subplots.                                                        #
import style.style as style               # Custom plotting styles and configurations.                                                #
# =================================================================================================================================== #


# =================================================================================================================================== #
# ---------------------------------------------------------- PLOTTING FUNCTIONS ----------------------------------------------------- #
def visualize_aoa_results(results, aoa_scan, title=None):
    """
    Visualize AoA estimation results with multiple subplots.
    
    Parameters:
        - results  [dict]           : Results from analyze_aoa function
        - aoa_scan [np.ndarray]     : Array of angles used for scanning
        - title    [str] (optional) : Main figure title
    """
    # Increase default font sizes for paper readability
    plt.rcParams['font.size'] = 20
    plt.rcParams['axes.titlesize'] = 22
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['xtick.labelsize'] = 19
    plt.rcParams['ytick.labelsize'] = 19
    plt.rcParams['legend.fontsize'] = 16

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
    
    # Plot 2: Method error comparison (simplified)
    ax2 = axes[1]
    methods = ['phase', 'ds', 'weighted', 'music']
    method_names = ['Phase', 'DS', 'DS+RSSI', 'MUSIC']
    colors = ['b', 'r', 'm', 'g']
    
    if results['angles']['true'] is not None:
        # Plot only the absolute errors as bars
        errors = [results['errors'][m] for m in methods]
        y_pos = np.arange(len(methods))
        
        # Create error bar chart
        bars = ax2.bar(y_pos, errors, color=colors, alpha=0.7)
        
        # Add text labels on top of bars
        for i, v in enumerate(errors):
            ax2.text(i, v + 0.1, f'{v:.1f}°', ha='center', fontsize=14)
            
        # Add estimated angles as text at the bottom
        for i, method in enumerate(methods):
            est_angle = results['angles'][method]
            ax2.text(i, -0.3, f'{est_angle:.1f}°', ha='center', fontsize=12, rotation=0)
            
        # Set axis labels and limits
        ax2.set_ylim(0, max(errors) * 1.3)  # Add headroom for text labels
        ax2.set_xticks(y_pos)
        ax2.set_xticklabels(method_names)
        ax2.set_ylabel('Absolute Error (degrees)')
        ax2.set_title('Method Error Comparison')
        
        # Add a text annotation for the true angle
        true_angle = results['angles']['true']
        ax2.annotate(f'True Angle: {true_angle:.1f}°', 
                    xy=(0.5, 0.95), xycoords='axes fraction',
                    ha='center', fontsize=14, 
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
    else:
        # If no true angle is available, show a message
        ax2.text(0.5, 0.5, 'No true angle available\nfor error comparison', 
                ha='center', va='center', fontsize=16,
                transform=ax2.transAxes)
        ax2.set_xticks([])
        ax2.set_yticks([])
    
    ax2.grid(True, alpha=0.3)
    
    # Main title
    if title:
        fig.suptitle(title, fontsize=22)
    plt.tight_layout()
    return fig
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
    # Increase default font sizes for paper readability
    plt.rcParams['font.size'] = 18
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['legend.fontsize'] = 16

    print("Starting RFID AoA Analysis Dashboard Creation...")
    # Step 1: Create DataManager and import data
    rfid_data = main.DataManager(data_dir=main.DATA_DIRECTORY, tag_id=main.TAG_ID, aoa_range=main.AoA_m)
    rfid_data.import_data()
    # Step 2: Create output directories
    dashboard_dir = os.path.join(main.RESULTS_DIRECTORY, "dashboard")
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
            analysis_aoa = np.arange(main.MIN_ANGLE, main.MAX_ANGLE + analysis_step, analysis_step)
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
                aoa_results = main.analyze_aoa(
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
        ax_aoa.plot(widths, theta_true_d, 'k--o', linewidth=2, label='True')
        ax_aoa.plot(widths, theta_phase_d, 'b-s', linewidth=2, label='Phase')
        ax_aoa.plot(widths, theta_ds_d, 'r-^', linewidth=2, label='DS')
        ax_aoa.plot(widths, theta_w_d, 'm-d', linewidth=2, label='DS+RSSI')
        ax_aoa.plot(widths, theta_music_d, 'g-o', linewidth=2, label='MUSIC')
        ax_aoa.set_xlabel('Width (m)')
        ax_aoa.set_ylabel('AoA (degrees)')
        ax_aoa.set_title(f'AoA Estimation vs Width (D = {distance:.2f}m)')
        ax_aoa.grid(True, alpha=0.3)
        ax_aoa.legend()
        # Create and save standalone AoA vs Width plot
        aoa_width_fig = plt.figure(figsize=(10, 4))
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.plot(widths, theta_true_d, 'k--o', linewidth=2, label='True')
        plt.plot(widths, theta_phase_d, 'b-s', linewidth=2, label='Phase')
        plt.plot(widths, theta_ds_d, 'r-^', linewidth=2, label='DS')
        plt.plot(widths, theta_w_d, 'm-d', linewidth=2, label='DS+RSSI')
        plt.plot(widths, theta_music_d, 'g-o', linewidth=2, label='MUSIC')
        plt.xlabel('Width (m)')
        plt.ylabel('AoA (degrees)')
        plt.title(f'AoA Estimation vs Width (D = {distance:.2f}m)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        # Calculate MAE for subtitle
        mae_phase = np.mean(np.abs(theta_phase_d - theta_true_d))
        mae_ds = np.mean(np.abs(theta_ds_d - theta_true_d))
        mae_w = np.mean(np.abs(theta_w_d - theta_true_d))
        mae_music = np.mean(np.abs(theta_music_d - theta_true_d))
        plt.figtext(0.5, 0.01, 
                    f'MAE: Phase={mae_phase:.1f}°, DS={mae_ds:.1f}°, DS+RSSI={mae_w:.1f}°, MUSIC={mae_music:.1f}°', 
                    ha='center', fontsize=14)
        plt.tight_layout()
        aoa_width_fig.savefig(os.path.join(main.RESULTS_BASE_DIR, f'aoa_vs_width_D{distance:.2f}.png'), 
                             dpi=300, bbox_inches='tight')
        plt.close(aoa_width_fig)
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
                                    extent=[main.MIN_ANGLE, main.MAX_ANGLE, widths[-1], widths[0]],
                                    aspect='auto', cmap='jet')
            plt.colorbar(im_ds, ax=ax_heat_ds)
            ax_heat_ds.set_xlabel('AoA (degrees)')
            ax_heat_ds.set_ylabel('Width (m)')
            ax_heat_ds.set_title('DS Beamforming')
            # Plot true angles on heatmap
            ax_heat_ds.plot(theta_true_d, widths, 'w--', linewidth=1.5)
            im_w = ax_heat_w.imshow(heatmap_data_w, 
                                  extent=[main.MIN_ANGLE, main.MAX_ANGLE, widths[-1], widths[0]],
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
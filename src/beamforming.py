# =================================================================================================================================== #
# ----------------------------------------------------------- DESCRIPTION ----------------------------------------------------------- #
# This script contains the functions for analyzing Angle of Arrival (AoA) measurements from RFID systems, using the classical         #
# antenna-array method "Beamforming". It includes the classic delay-and-sum (DS) beamforming and the weighted DS beamforming, using   #
# RSSI data for the weights.                                                                                                          #
# =================================================================================================================================== #


# =================================================================================================================================== #
# --------------------------------------------------------- EXTERNAL IMPORTS -------------------------------------------------------- #
import numpy as np  # Mathematical functions.                                                                                         #
# =================================================================================================================================== #


# =================================================================================================================================== #
# ------------------------------------------------------- BEAMFORMING FUNCTION ------------------------------------------------------ #
def beamforming_spectrum_calculation(phasor1, phasor2, rssi1, rssi2, L, wavelength, aoa_scan):
    """
    Calculate the beamforming spectrum (both standard and RSSI-weighted) for RFID AoA estimation.
    
    Parameters:
        - phasor1    [np.ndarray] : Complex phasors from antenna 1
        - phasor2    [np.ndarray] : Complex phasors from antenna 2
        - rssi1      [np.ndarray] : RSSI values from antenna 1 (in dBm)
        - rssi2      [np.ndarray] : RSSI values from antenna 2 (in dBm)
        - L          [float]      : Antenna separation distance (in meters)
        - wavelength [float]      : Signal wavelength (in meters)
        - aoa_scan   [np.ndarray] : Array of angles to scan (in degrees)
        
    Returns:
        - tuple    : (B_ds, B_w, theta_ds, theta_w)
            - B_ds     : Standard delay-and-sum beamforming spectrum
            - B_w      : RSSI-weighted beamforming spectrum
            - theta_ds : Estimated angle using standard beamforming
            - theta_w  : Estimated angle using RSSI-weighted beamforming
    """
    # STEP 1: Align phasors to the same length
    N       = min(len(phasor1), len(phasor2))
    phasor1 = phasor1[:N]
    phasor2 = phasor2[:N]
    # STEP 2: Transform RSSI from dBm to linear scale, and normalize by minimum value
    all_rssi = [rssi1, rssi2]
    rssi_min = min([np.min(r) for r in all_rssi])
    w1       = np.sqrt(10**((rssi1 - rssi_min) / 10))  # Weight for antenna 1
    w2       = np.sqrt(10**((rssi2 - rssi_min) / 10))  # Weight for antenna 2
    # STEP 3: Calculate the beamforming spectrum
    k    = 2 * np.pi / wavelength  # Wave number
    M    = len(aoa_scan)
    B_ds = np.zeros(M)  # Standard delay-and-sum spectrum
    B_w  = np.zeros(M)   # RSSI-weighted spectrum
    for m in range(M):
        # Phase shift based on angle
        dphi = k * L * np.sin(np.deg2rad(aoa_scan[m]))
        # Standard delay-and-sum beamforming
        y_ds    = phasor1 + np.exp(-1j * dphi) * phasor2
        B_ds[m] = np.mean(np.abs(y_ds)**2)
        # RSSI-weighted beamforming
        y_w    = w1 * phasor1 + np.exp(-1j * dphi) * (w2 * phasor2)
        B_w[m] = np.mean(np.abs(y_w)**2)
    # STEP 4: Normalize spectra
    B_ds = B_ds / np.max(B_ds) if np.max(B_ds) > 0 else B_ds
    B_w  = B_w / np.max(B_w) if np.max(B_w) > 0 else B_w
    # STEP 5: Find peak angles
    theta_ds = aoa_scan[np.argmax(B_ds)]
    theta_w  = aoa_scan[np.argmax(B_w)]
    return B_ds, B_w, theta_ds, theta_w
# =================================================================================================================================== #
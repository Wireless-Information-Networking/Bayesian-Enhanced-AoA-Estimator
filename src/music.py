# =================================================================================================================================== #
# ----------------------------------------------------------- DESCRIPTION ----------------------------------------------------------- #
# This script contains the functions for analyzing Angle of Arrival (AoA) measurements from RFID systems, using the classical         #
# antenna-array method "MUSIC Algorithm".                                                                                             #
# =================================================================================================================================== #


# =================================================================================================================================== #
# --------------------------------------------------------- EXTERNAL IMPORTS -------------------------------------------------------- #
import numpy as np  # Mathematical functions.                                                                                         #
# =================================================================================================================================== #


# =================================================================================================================================== #
# ---------------------------------------------------------- MUSIC FUNCTION --------------------------------------------------------- #
def music_algorithm(phasor1, phasor2, L, wavelength, aoa_scan):
    """
    Implement the MUSIC (MUltiple SIgnal Classification) algorithm for RFID AoA estimation.
    
    Parameters:
        phasor1    [np.ndarray] : Complex phasors from antenna 1
        phasor2    [np.ndarray] : Complex phasors from antenna 2
        L          [float]      : Antenna separation distance (in meters)
        wavelength [float]      : Signal wavelength (in meters)
        aoa_scan   [np.ndarray] : Array of angles to scan (in degrees)
        
    Returns:
        - tuple : (theta_music, P_music)
            - theta_music : Estimated angle using MUSIC algorithm
            - P_music     : MUSIC spectrum
    """
    # STEP 1: Form spatial covariance matrix
    # Reshape phasors to column vectors and combine
    x1 = phasor1.reshape(-1, 1)
    x2 = phasor2.reshape(-1, 1)
    X = np.hstack((x1, x2))  # Combine signals from both antennas
    # Calculate spatial covariance matrix
    R = X.conj().T @ X / X.shape[0]
    # STEP 2: Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eig(R)
    # Sort eigenvectors by eigenvalues in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    # STEP 3: Noise subspace (assuming 1 signal, so 1 eigenvector for signal)
    # Take all eigenvectors except the first one
    En = eigenvectors[:, 1:]
    # STEP 4: MUSIC spectrum calculation
    k = 2 * np.pi / wavelength
    P_music = np.zeros(len(aoa_scan))
    for i in range(len(aoa_scan)):
        # Array steering vector
        a = np.array([1, np.exp(-1j * k * L * np.sin(np.deg2rad(aoa_scan[i])))])
        a = a.reshape(-1, 1) 
        # MUSIC pseudospectrum - FIX: Extract scalar value using .item()
        denominator = a.conj().T @ (En @ En.conj().T) @ a
        P_music[i] = 1 / np.abs(denominator.item())  # Use .item() to extract scalar value
    # STEP 5: Normalize spectrum
    P_music = P_music / np.max(P_music) if np.max(P_music) > 0 else P_music
    # STEP 6: Find peak
    theta_music = aoa_scan[np.argmax(P_music)]
    
    return theta_music, P_music
# =================================================================================================================================== #
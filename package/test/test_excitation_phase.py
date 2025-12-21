# package/test/test_excitation_phase.py
import pytest
import pandas as pd
import numpy as np
import os
import warnings
from openflash_utils import run_openflash_case
from openflash.multi_equations import wavenumber

# Define constants
H = 300
A = [3, 10]
D = [35, 2]
RHO = 1023
HEAVING = [1, 1]
NMK = [50, 50, 50]

# Path to data
DATA_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), 
        "../../dev/python/test/data/WAMIT_exc_phase.csv"
    )
)

def load_wamit_data():
    if not os.path.exists(DATA_PATH):
        pytest.skip(f"WAMIT data file not found at {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH)
    omegas = 0.02 * np.arange(1, 261)
    # Negate the Series first to avoid Pylance/Type errors
    wamit_phases = (-df["excitation phase (rad)"]).values
    return omegas, wamit_phases

def calculate_angular_difference(angle1, angle2):
    diff = angle1 - angle2
    return np.arctan2(np.sin(diff), np.cos(diff))

def test_excitation_phase_match():
    """
    Compare OpenFLASH excitation phase against WAMIT data.
    """
    omegas, wamit_phases = load_wamit_data()
    
    max_error = 0.0
    failures = []
    
    # Filter warnings about sinh overflow (harmless in deep water approx)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in sinh")
        
        for i, omega in enumerate(omegas):
            # FIX: Skip very high frequencies where force -> 0 and phase is unstable
            if omega > 5.1:
                continue

            # FIX 1: Convert Omega (rad/s) to Wavenumber (rad/m)
            # OpenFLASH core requires m0, not omega.
            m0 = wavenumber(omega, H)

            # FIX 2: Unpack the tuple return values
            # run_openflash_case returns (AddedMass, Damping, Phase)
            _, _, phase = run_openflash_case(H, D, A, HEAVING, NMK, m0, RHO)
            
            # Compare
            wamit_val = wamit_phases[i]
            error = abs(calculate_angular_difference(phase, wamit_val))
            
            if error > max_error:
                max_error = error

            # Tolerance: 0.1 rad (~5.7 degrees) is acceptable for BEM vs Analytical
            if error > 0.1:
                failures.append(
                    f"Omega={omega:.2f}: MEEM={phase:.4f}, WAMIT={wamit_val:.4f}, Diff={error:.4f}"
                )

    # Print summary
    print(f"\nMax Angular Error (omega < 5.1): {max_error:.6f} rad")
    
    # Assert
    if failures:
        print("\n".join(failures))
        pytest.fail(f"Test failed with {len(failures)} mismatches. See stdout for details.")
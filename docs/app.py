import streamlit as st
import sys
import os
import numpy as np
import pandas as pd
from scipy import linalg
import matplotlib.pyplot as plt
from math import sqrt, cosh, cos, sinh, sin, pi
from scipy.special import hankel1 as besselh, iv as besseli, kv as besselk
from scipy.optimize import newton, minimize_scalar
from scipy.integrate import quad

# Ensure the correct path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../package/src')))
from equations import (
    phi_p_i1, phi_p_i2, diff_phi_i1, diff_phi_i2, Z_n_i1, Z_n_i2, Z_n_e,
    m_k, Lambda_k_r, diff_Lambda_k_a2, R_1n_1, R_1n_2, R_2n_2,
    diff_R_1n_1, diff_R_1n_2, diff_R_2n_2, R_2n_1
)
from meem_engine import MEEMEngine
from meem_problem import MEEMProblem
from geometry import Geometry

# Helper function to display the A matrix visualization
def visualize_A_matrix(A, title="Matrix Visualization"):
    rows, cols = np.nonzero(A)
    plt.figure(figsize=(6, 6))
    plt.scatter(cols, rows, color='blue', marker='o', s=100)
    plt.gca().invert_yaxis()
    plt.xticks(range(A.shape[1]))
    plt.yticks(range(A.shape[0]))
    N, M = 4, 4
    block_dividers = [N, N + M, N + 2 * M]
    for val in block_dividers:
        plt.axvline(val - 0.5, color='black', linestyle='-', linewidth=1)
        plt.axhline(val - 0.5, color='black', linestyle='-', linewidth=1)
    plt.grid(True)
    plt.title(title)
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    st.pyplot(plt)
    plt.close()

# Helper function to plot the potential field
def plot_potential_field(R, Z, phi, title="Potential Field"):
    plt.figure(figsize=(8, 6))
    plt.contourf(R, Z, phi.real, levels=50, cmap='viridis')
    plt.colorbar(label='Potential')
    plt.title(title)
    plt.xlabel('Radial Coordinate (r)')
    plt.ylabel('Vertical Coordinate (z)')
    st.pyplot(plt)
    plt.close()

def plot_potential(phi, R, Z, title):
    """Plot the real and imaginary parts of a potential function."""
    plt.figure(figsize=(12, 6))

    # Real part
    plt.subplot(1, 2, 1)
    contour_real = plt.contourf(R, Z, np.real(phi), levels=50, cmap='viridis')
    plt.colorbar(contour_real)
    plt.title(f'{title} - Real Part')
    plt.xlabel('Radial Distance (R)')
    plt.ylabel('Axial Distance (Z)')

    # Imaginary part
    plt.subplot(1, 2, 2)
    contour_imag = plt.contourf(R, Z, np.imag(phi), levels=50, cmap='viridis')
    plt.colorbar(contour_imag)
    plt.title(f'{title} - Imaginary Part')
    plt.xlabel('Radial Distance (R)')
    plt.ylabel('Axial Distance (Z)')

    plt.tight_layout()
    
    # Use st.pyplot to display the plot in Streamlit
    st.pyplot(plt)
    
    # Clear the figure to free up memory
    plt.close()

def plot_matching(phi1, phi2, phi3, a1, a2, R, Z, title):
    plt.figure(figsize=(8, 6))
    mask1 = R <= a1
    mask2 = (R > a1) & (R <= a2)
    mask3 = R > a2

    # Create a combined field based on regions
    combined_field = np.zeros_like(R)
    combined_field[mask1] = phi1[mask1]
    combined_field[mask2] = phi2[mask2]
    combined_field[mask3] = phi3[mask3]

    plt.contourf(R, Z, combined_field, levels=50, cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('Radial Distance (R)')
    plt.ylabel('Axial Distance (Z)')
    st.pyplot(plt)
    plt.close()

# Main function for the Streamlit app
def main():
    st.title("MEEM Engine Simulation")

    # Sidebar for simulation parameters
    st.sidebar.header("Geometry configuration and problem instance")
    # User can modify N, M, K
    N = st.sidebar.number_input("Enter value for N:", value=4, min_value=1, step=1)
    M = st.sidebar.number_input("Enter value for M:", value=4, min_value=1, step=1)
    K = st.sidebar.number_input("Enter value for K:", value=4, min_value=1, step=1)
    a1 = st.sidebar.number_input("Enter value for a1:", value=0.5, step=0.01)
    a2 = st.sidebar.number_input("Enter value for a2:", value=1.0, step=0.01)
    h = st.sidebar.number_input("Enter value for h:", value=1.001, step=0.01)
    d1 = st.sidebar.number_input("Enter value for d1:", value=0.5, step=0.01)
    d2 = st.sidebar.number_input("Enter value for d2:", value=0.25, step=0.01)

    # Sidebar for tolerance and threshold
    st.sidebar.header("Tolerance and Threshold Settings")
    tolerance = st.sidebar.number_input("Set tolerance level:", value=1e-3, format="%.5f", step=1e-5)
    threshold = st.sidebar.number_input("Set threshold level:", value=0.01, step=0.001)

    # Sidebar for expected_b values
    st.sidebar.header("Expected b Values")
    expected_b = np.array([
        st.sidebar.number_input(f"expected_b[{i}]", value=value) 
        for i, value in enumerate([
            0.0069, 0.0120, -0.0030, 0.0013, 
            0.1560, 0.0808, -0.0202, 0.0090, 
            0, -0.1460, 0.0732, -0.0002, 
            -0.4622, -0.2837, 0.1539, -0.0673
        ])
    ], dtype=np.complex128)

    # Define functions for the potentials
    def phi_h_n_i1_func(n, r, z):
        return (C_1n_1s[n] * R_1n_1(n, r) + C_2n_1s[n] * R_2n_1(n)) * Z_n_i1(n, z)

    def phi_h_m_i2_func(m, r, z):
        return (C_1n_2s[m] * R_1n_2(m, r) + C_2n_2s[m] * R_2n_2(m, r)) * Z_n_i2(m, z)

    def phi_e_k_func(k, r, z):
        return B_ks[k] * Lambda_k_r(k, r) * Z_n_e(k, z)


    # Geometry and domain configuration
    domain_params = [
        {'number_harmonics': N, 'height': h, 'radial_width': a1, 'top_BC': None, 'bottom_BC': None, 'category': 'inner', 'di': d1},
        {'number_harmonics': M, 'height': h, 'radial_width': a2, 'top_BC': None, 'bottom_BC': None, 'category': 'outer', 'di': d2},
        {'number_harmonics': K, 'height': h, 'radial_width': 1.5, 'top_BC': None, 'bottom_BC': None, 'category': 'exterior'}
    ]
    
    if st.sidebar.button("Show Domain Parameters"):
        # Display the domain configuration
        st.write("Domain Parameters:", domain_params)

    r_coordinates = {'a1': a1, 'a2': a2}
    z_coordinates = {'h': h}
    geometry = Geometry(r_coordinates, z_coordinates, domain_params)
    problem = MEEMProblem(geometry)
    engine = MEEMEngine([problem])

    generated_A = engine.assemble_A(problem)
    if st.sidebar.button("Generate A Matrix"):
        st.header("Matrix and Vector Assembly")
        visualize_A_matrix(generated_A, title="Generated A Matrix")

    cols = [4, 8, 12]

    for val in cols:
        plt.axvline(val - 0.5, color='black', linestyle='-', linewidth=1)
        plt.axhline(val - 0.5, color='black', linestyle='-', linewidth=1)

    plt.grid(True)
    plt.title('Non-Zero Entries of the Matrix Not Matching the Threshold')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    # st.pyplot(plt)
    plt.close()

    # Generate and verify b vector
    generated_b = engine.assemble_b(problem)
    # if np.allclose(generated_b, expected_b, atol=1e-3):
    #     st.success("b vector matches expected values.")
    # else:
    #     st.error("b vector does not match expected values.")

    # Save matching results of b vector
    is_within_threshold_b = np.isclose(expected_b, generated_b, atol=threshold)

    # Solve the system A x = b
    X = linalg.solve(generated_A, generated_b)

    # Extract coefficients from X (assuming X is already defined)
    C_1n_1s = X[:N]
    C_1n_2s = X[N:N+M]
    C_2n_2s = X[N+M:N+2*M]
    C_2n_1s = np.zeros(N, dtype=complex)  # Assuming C_2n_1s are zeros
    B_ks = X[N+2*M:]

    # Spatial grid and potential field plotting
    spatial_res = 50
    r_vec = np.linspace(0, 2 * a2, spatial_res)
    z_vec = np.linspace(-h, 0, spatial_res)
    R, Z = np.meshgrid(r_vec, z_vec)
    
    # Define regions
    region1 = (R <= a1) & (Z < -d1)
    region2 = (R > a1) & (R <= a2) & (Z < -d2)
    regione = R > a2
    region_body = ~region1 & ~region2 & ~regione  # The body of the cylinder

    # Initialize potential arrays
    phiH = np.zeros_like(R, dtype=complex)
    phiP = np.zeros_like(R, dtype=complex)

    # Compute the homogeneous potential in each region
    # Region 1 (using phi_h_n_i1_func)
    for n in range(4):
        phiH[region1] += phi_h_n_i1_func(n, R[region1], Z[region1])

    # Region 2 (using phi_h_m_i2_func)
    for m in range(4):
        phiH[region2] += phi_h_m_i2_func(m, R[region2], Z[region2])

    # Exterior region (using phi_e_k_func)
    for k in range(4):
        phiH[regione] += phi_e_k_func(k, R[regione], Z[regione])

    # Compute the particular potential in each region (using phi_p_i1 and phi_p_i2)
    phiP[region1] = phi_p_i1(R[region1], Z[region1])
    phiP[region2] = phi_p_i2(R[region2], Z[region2])
    phiP[regione] = 0  # No particular potential in the exterior region

    # Compute the total potential
    phi = phiH + phiP

    # Now you have the actual potential data in the variable 'phi'
    
    # Extract potentials in different regions for matching
    phi1 = np.where(region1, phi, np.nan)
    phi2 = np.where(region2, phi, np.nan)
    phie = np.where(regione, phi, np.nan)
    if st.sidebar.button("Plot matching at interfaces"):
        # Plot matching at interfaces
        plot_matching(phi1, phi2, phie, a1, a2, R, Z, 'Potential')
    
    # Additional verification: Continuity at interfaces
    def verify_continuity_at_interfaces():
        # Potential at r = a1 from region1 and region2
        idx_a1 = np.argmin(np.abs(R[0, :] - a1))
        phi_a1_region1 = phi1[:, idx_a1]
        phi_a1_region2 = phi2[:, idx_a1]

        # Compute the difference
        difference_a1 = np.abs(phi_a1_region1 - phi_a1_region2)

        # Potential at r = a2 from region2 and exterior region
        idx_a2 = np.argmin(np.abs(R[0, :] - a2))
        phi_a2_region2 = phi2[:, idx_a2]
        phi_a2_exterior = phie[:, idx_a2]

        # Compute the difference
        difference_a2 = np.abs(phi_a2_region2 - phi_a2_exterior)

        # Plot the differences
        plt.figure(figsize=(8, 6))
        plt.plot(Z[:, 0], difference_a1, label='Difference at r = a1')
        plt.plot(Z[:, 0], difference_a2, label='Difference at r = a2')
        plt.legend()
        plt.xlabel('Z')
        plt.ylabel('Potential Difference')
        plt.title('Continuity of Potential at Interfaces')
        st.pyplot(plt)
        plt.close()
    
    if st.sidebar.button("Plot the Potentials"):
        # Assume phi, R, and Z have been calculated/generated elsewhere in the app
        title = "Potentials"
        # Plot the potentials
        plot_potential(phiH, R, Z, 'Homogeneous Potential')
        plot_potential(phiP, R, Z, 'Particular Potential')
        plot_potential(phi, R, Z, 'Total Potential')

    if st.sidebar.button("Generate Potential Field Plots"):
        # Assume phi, R, and Z have been calculated/generated elsewhere in the app
        title = "Potentials"
        # Plot the potentials
        # Plot Homogeneous Potential
        plot_potential_field(R, Z, phiH, title="Homogeneous Potential")

        # Plot Particular Potential
        plot_potential_field(R, Z, phiP, title="Particular Potential")

        # Plot Total Potential
        plot_potential_field(R, Z, phi, title="Total Potential")



    if st.sidebar.button("Additional verification: Continuity at interfaces"):
        verify_continuity_at_interfaces()

if __name__ == "__main__":
    main()

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import os
import sys
import pandas as pd

# Import your custom modules
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../package/src'))
sys.path.append(src_path)

from equations import *
from constants import *
from multi_equations import *
from meem_engine import MEEMEngine
from meem_problem import MEEMProblem
from geometry import Geometry

np.set_printoptions(threshold=np.inf, linewidth=np.inf, precision=8, suppress=True)

# Streamlit App
st.title("MEEM Simulation")

# Sidebar inputs
st.sidebar.header("Simulation Parameters")
NMK = st.sidebar.text_input("Number of harmonics in each region (comma-separated)", "20,20,20,20")
NMK = list(map(int, NMK.split(",")))
boundary_count = len(NMK) - 1

h = st.sidebar.number_input("Height of the domain (h)", value=10.0, step=0.1)
a = st.sidebar.text_input("List of radii (comma-separated)", "1.0,1.5,2.0,3.0")
a = list(map(float, a.split(",")))
d = st.sidebar.text_input("List of depths (comma-separated)", "0.1,0.2,0.3,0.4")
d = list(map(float, d.split(",")))
heaving = st.sidebar.text_input("List of heaving amplitudes (comma-separated)", "1.0,1.0,1.0,1.0")
heaving = list(map(float, heaving.split(",")))

omega = st.sidebar.number_input("Frequency (omega)", value=1.0, step=0.1)

# Create domain parameters
# Create domain parameters dynamically
domain_params_multi = []
for idx in range(len(NMK)):
    params = {
        'number_harmonics': NMK[idx],
        'height': h - d[idx] if idx < len(d) else h,
        'radial_width': a[idx] if idx < len(a) else a[-1] * 1.5,
        'top_BC': None,
        'bottom_BC': None,
        'category': 'multi',
        'di': d[idx] if idx < len(d) else 0,
        'a': a[idx] if idx < len(a) else a[-1] * 1.5,
        'heaving': heaving[idx] if idx < len(heaving) else False,
        'slant': [0, 0, 1]
    }
    domain_params_multi.append(params)

# Display the domain parameters to the user
st.subheader("Domain Parameters")
domain_table = []
for idx, params in enumerate(domain_params_multi):
    domain_table.append({
        "Region": idx + 1,
        "Harmonics": params['number_harmonics'],
        "Height": params['height'],
        "Radial Width": params['radial_width'],
        "Depth (di)": params['di'],
        "Heaving": params['heaving'],
        "Slant": params['slant'],
    })

st.table(domain_table)

# Create Geometry object
r_coordinates = {'a': a}
z_coordinates = {'h': h}
geometry = Geometry(r_coordinates, z_coordinates, domain_params_multi)

# Create MEEMProblem object
problem = MEEMProblem(geometry)

# Create MEEMEngine object
engine = MEEMEngine([problem])

# Assemble A matrix and b vector using multi-region methods
A = engine.assemble_A_multi(problem)
b = engine.assemble_b_multi(problem)


# Solve the linear system A x = b
X = linalg.solve(A, b)


# Simulation code

size = NMK[0] + NMK[-1] + 2 * sum(NMK[1:len(NMK) - 1])
hydro_terms = np.zeros((size - NMK[-1]), dtype=complex)

col = 0
for n in range(NMK[0]):
    hydro_terms[n] = int_R_1n(0, n)* X[n] * z_n_d(n)
col += NMK[0]
for i in range(1, boundary_count):
    M = NMK[i]
    for m in range(M):
        hydro_terms[col + m] = int_R_1n(i, m)* X[col + m] * z_n_d(m)
        hydro_terms[col + M + m] = int_R_2n(i, m)* X[col + M + m] * z_n_d(m)
    col += 2 * M

hydro_p_terms = np.zeros(boundary_count, dtype=complex)
for i in range(boundary_count):
    hydro_p_terms[i] = heaving[i] * int_phi_p_i_no_coef(i)

hydro_coef = 2 * pi * (sum(hydro_terms) + sum(hydro_p_terms))
hydro_coef_real = hydro_coef.real
hydro_coef_imag = hydro_coef.imag / omega

# Find maximum heaving radius
max_rad = a[0]
for i in range(boundary_count - 1, 0, -1):
    if heaving[i]:
        max_rad = a[i]
        break

hydro_coef_nondim = h**3 / (max_rad**3 * pi) * hydro_coef

st.write(f"Hydrodynamic coefficient (real part): {hydro_coef_real}")
st.write(f"Hydrodynamic coefficient (imaginary part): {hydro_coef_imag}")
st.write(f"Non-dimensional hydrodynamic coefficient: {hydro_coef_nondim}")

# Split up the Cs into groups depending on which equation they belong to.
Cs = []
row = 0
Cs.append(X[:NMK[0]])
row += NMK[0]
for i in range(1, boundary_count):
    Cs.append(X[row: row + NMK[i] * 2])
    row += NMK[i] * 2
Cs.append(X[row:])

def phi_h_n_inner_func(n, r, z):
    return (Cs[0][n] * R_1n(n, r, 0)) * Z_n_i(n, z, 0)

def phi_h_m_i_func(i, m, r, z):
    return (Cs[i][m] * R_1n(m, r, i) + Cs[i][NMK[i] + m] * R_2n(m, r, i)) * Z_n_i(m, z, i)

def phi_e_k_func(k, r, z):
    return Cs[-1][k] * Lambda_k(k, r) * Z_n_e(k, z)

# Create radial and vertical grids
spatial_res = st.sidebar.slider("Spatial Resolution", min_value=10, max_value=200, value=50, step=10)
r_vec = np.linspace(2 * a[-1] / spatial_res, 2 * a[-1], spatial_res)
z_vec = np.linspace(-h, 0, spatial_res)

# Add values at the radii
a_eps = 1.0e-4
for i in range(len(a)):
    r_vec = np.append(r_vec, a[i] * (1 - a_eps))
    r_vec = np.append(r_vec, a[i] * (1 + a_eps))
r_vec = np.unique(r_vec)

for i in range(len(d)):
    z_vec = np.append(z_vec, -d[i])
z_vec = np.unique(z_vec)

R, Z = np.meshgrid(r_vec, z_vec)

regions = []
regions.append((R <= a[0]) & (Z < -d[0]))
for i in range(1, boundary_count):
    regions.append((R > a[i-1]) & (R <= a[i]) & (Z < -d[i]))
regions.append(R > a[-1])

phi = np.full_like(R, np.nan + np.nan * 1j, dtype=complex)
phiH = np.full_like(R, np.nan + np.nan * 1j, dtype=complex)
phiP = np.full_like(R, np.nan + np.nan * 1j, dtype=complex)

# Calculate phiH (Homogeneous Potential)
for n in range(NMK[0]):
    temp_phiH = phi_h_n_inner_func(n, R[regions[0]], Z[regions[0]])
    phiH[regions[0]] = temp_phiH if n == 0 else phiH[regions[0]] + temp_phiH

for i in range(1, boundary_count):
    for m in range(NMK[i]):
        temp_phiH = phi_h_m_i_func(i, m, R[regions[i]], Z[regions[i]])
        phiH[regions[i]] = temp_phiH if m == 0 else phiH[regions[i]] + temp_phiH

for k in range(NMK[-1]):
    temp_phiH = phi_e_k_func(k, R[regions[-1]], Z[regions[-1]])
    phiH[regions[-1]] = temp_phiH if k == 0 else phiH[regions[-1]] + temp_phiH

# Calculate phiP (Particular Potential)
phi_p_i_vec = np.vectorize(phi_p_i)
phiP[regions[0]] = heaving[0] * phi_p_i_vec(d[0], R[regions[0]], Z[regions[0]])
for i in range(1, boundary_count):
    phiP[regions[i]] = heaving[i] * phi_p_i_vec(d[i], R[regions[i]], Z[regions[i]])
phiP[regions[-1]] = 0

phi = phiH + phiP

# Plotting the potentials
def plot_potential(field, R, Z, title):
     # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create the contour plot
    contour = ax.contourf(R, Z, field, levels=50, cmap='viridis')

    # Add a colorbar
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label('Potential')

    # Add title and labels
    ax.set_title(title)
    ax.set_xlabel('Radial Distance (R)')
    ax.set_ylabel('Axial Distance (Z)')

    # Display the plot in Streamlit
    st.pyplot(fig)

plot_potential(np.real(phiH), R, Z, 'Homogeneous Potential')
plot_potential(np.imag(phiH), R, Z, 'Homogeneous Potential Imaginary')
plot_potential(np.real(phiP), R, Z, 'Particular Potential')
plot_potential(np.imag(phiP), R, Z, 'Particular Potential Imaginary')
plot_potential(np.real(phi), R, Z, 'Total Potential')
plot_potential(np.imag(phi), R, Z, 'Total Potential Imaginary')

#  Velocity calculations and plots
def v_r_inner_func(n, r, z):
    return (Cs[0][n] * diff_R_1n(n, r, 0)) * Z_n_i(n, z, 0)

def v_r_m_i_func(i, m, r, z):
    return (Cs[i][m] * diff_R_1n(m, r, i) + Cs[i][NMK[i] + m] * diff_R_2n(m, r, i)) * Z_n_i(m, z, i)

def v_r_e_k_func(k, r, z):
    return Cs[-1][k] * diff_Lambda_k(k, r) * Z_n_e(k, z)

def v_z_inner_func(n, r, z):
    return (Cs[0][n] * R_1n(n, r, 0)) * diff_Z_n_i(n, z, 0)

def v_z_m_i_func(i, m, r, z):
    return (Cs[i][m] * R_1n(m, r, i) + Cs[i][NMK[i] + m] * R_2n(m, r, i)) * diff_Z_n_i(m, z, i)

def v_z_e_k_func(k, r, z):
    return Cs[-1][k] * Lambda_k(k, r) * diff_Z_n_e(k, z)

vr = np.full_like(R, np.nan + np.nan*1j, dtype=complex) 
vrH = np.full_like(R, np.nan + np.nan*1j, dtype=complex) 
vrP = np.full_like(R, np.nan + np.nan*1j, dtype=complex) 

vz = np.full_like(R, np.nan + np.nan*1j, dtype=complex) 
vzH = np.full_like(R, np.nan + np.nan*1j, dtype=complex) 
vzP = np.full_like(R, np.nan + np.nan*1j, dtype=complex)

for n in range(NMK[0]):
    temp_vrH = v_r_inner_func(n, R[regions[0]], Z[regions[0]])
    temp_vzH = v_z_inner_func(n, R[regions[0]], Z[regions[0]])
    if n == 0:
        vrH[regions[0]] = temp_vrH
        vzH[regions[0]] = temp_vzH
    else:
        vrH[regions[0]] = vrH[regions[0]] + temp_vrH
        vzH[regions[0]] = vzH[regions[0]] + temp_vzH

for i in range(1, boundary_count):
    for m in range(NMK[i]):
        temp_vrH = v_r_m_i_func(i, m, R[regions[i]], Z[regions[i]])
        temp_vzH = v_z_m_i_func(i, m, R[regions[i]], Z[regions[i]])
        if m == 0:
            vrH[regions[i]] = temp_vrH
            vzH[regions[i]] = temp_vzH
        else:
            vrH[regions[i]] = vrH[regions[i]] + temp_vrH
            vzH[regions[i]] = vzH[regions[i]] + temp_vzH

for k in range(NMK[-1]):
    temp_vrH = v_r_e_k_func(k, R[regions[-1]], Z[regions[-1]])
    temp_vzH = v_z_e_k_func(k, R[regions[-1]], Z[regions[-1]])
    if k == 0:
        vrH[regions[-1]] = temp_vrH
        vzH[regions[-1]] = temp_vzH
    else:
        vrH[regions[-1]] = vrH[regions[-1]] + temp_vrH
        vzH[regions[-1]] = vzH[regions[-1]] + temp_vzH

vr_p_i_vec = np.vectorize(diff_r_phi_p_i)
vz_p_i_vec = np.vectorize(diff_z_phi_p_i)

vrP[regions[0]] = heaving[0] * vr_p_i_vec(d[0], R[regions[0]], Z[regions[0]])
vzP[regions[0]] = heaving[0] * vz_p_i_vec(d[0], R[regions[0]], Z[regions[0]])
for i in range(1, boundary_count):
    vrP[regions[i]] = heaving[i] * vr_p_i_vec(d[i], R[regions[i]], Z[regions[i]])
    vzP[regions[i]] = heaving[i] * vz_p_i_vec(d[i], R[regions[i]], Z[regions[i]])
vrP[regions[-1]] = 0
vzP[regions[-1]] = 0

vr = vrH + vrP
vz = vzH + vzP
plot_potential(np.real(vr), R, Z, 'Radial Velocity - Real')
plot_potential(np.imag(vr), R, Z, 'Radial Velocity - Imaginary')
plot_potential(np.real(vz), R, Z, 'Vertical Velocity - Real')
plot_potential(np.imag(vz), R, Z, 'Vertical Velocity - Imaginary')

# Helper function to display the A matrix visualization
# keep
def visualize_A_matrix(A, title="Matrix Visualization"):
    rows, cols = np.nonzero(A)
    plt.figure(figsize=(6, 6))
    plt.scatter(cols, rows, color='blue', marker='o', s=100)
    plt.gca().invert_yaxis()
    plt.xticks(range(A.shape[1]))
    plt.yticks(range(A.shape[0]))
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

def plot_matching(phi1, phi2, phie, a1, a2, R, Z, title):
    """
    Visualize the potential matching at specified radial positions.

    Parameters:
        phi1: np.ndarray - Potential array for region 1.
        phi2: np.ndarray - Potential array for region 2.
        phie: np.ndarray - Potential array for the exterior region.
        a1: float - Radial coordinate of the first interface.
        a2: float - Radial coordinate of the second interface.
        R: np.ndarray - Radial coordinate grid.
        Z: np.ndarray - Vertical coordinate grid.
        title: str - Title for the plot.
    """
    idx_a1 = np.argmin(np.abs(R[0, :] - a1))
    idx_a2 = np.argmin(np.abs(R[0, :] - a2))

    Z_line = Z[:, 0]

    # Potential at r = a1
    phi1_a1 = phi1[:, idx_a1]
    phi2_a1 = phi2[:, idx_a1]

    # Potential at r = a2
    phi2_a2 = phi2[:, idx_a2]
    phie_a2 = phie[:, idx_a2]

    plt.figure(figsize=(8, 6))
    plt.plot(Z_line, np.abs(phi1_a1), 'r--', label=f'|{title}_1| at r = a1')
    plt.plot(Z_line, np.abs(phi2_a1), 'b-', label=f'|{title}_2| at r = a1')
    plt.plot(Z_line, np.abs(phi2_a2), 'g--', label=f'|{title}_2| at r = a2')
    plt.plot(Z_line, np.abs(phie_a2), 'k-', label=f'|{title}_e| at r = a2')
    plt.legend()
    plt.xlabel('Z')
    plt.ylabel(f'|{title}|')
    plt.title(f'{title} Matching at Interfaces')
    plt.grid(True)
    plt.show()


# Set expected values
#keep
# expected_b = np.array([
#     0.0069, 0.0120, -0.0030, 0.0013, 
#     0.1560, 0.0808, -0.0202, 0.0090, 
#     0, -0.1460, 0.0732, -0.0002, 
#     -0.4622, -0.2837, 0.1539, -0.0673
# ], dtype=np.complex128)

# # Convert to complex
# #keep
# def to_complex(val):
#     try:
#         return np.complex128(val)
#     except ValueError:
#         return np.nan + 1j * np.nan

# # Define functions for the potentials
# # keep
# def phi_h_n_i1_func(n, r, z):
#     return (C_1n_1s[n] * R_1n_1(n, r) + C_2n_1s[n] * R_2n_1(n)) * Z_n_i1(n, z)

# def phi_h_m_i2_func(m, r, z):
#     return (C_1n_2s[m] * R_1n_2(m, r) + C_2n_2s[m] * R_2n_2(m, r)) * Z_n_i2(m, z)

# def phi_e_k_func(k, r, z):
#     return B_ks[k] * Lambda_k_r(k, r) * Z_n_e(k, z)


# # Geometry and domain configuration
# #keep
# N = NMK[0]
# M = NMK[1]
# K = NMK[2]
# a1 = a[0]
# a2 = a[1]
# a3 = a[2]
# d1 = d[0]
# d2 = d[1]
# r_coordinates = {'a': a}
# z_coordinates = {'h': h}
# domain_params = [
#     {'number_harmonics': N, 'height': h, 'radial_width': a1, 'top_BC': None, 'bottom_BC': None, 'category': 'inner', 'di': d1},
#     {'number_harmonics': M, 'height': h, 'radial_width': a2, 'top_BC': None, 'bottom_BC': None, 'category': 'outer', 'di': d2},
#     {'number_harmonics': K, 'height': h, 'radial_width': a3, 'top_BC': None, 'bottom_BC': None, 'category': 'exterior'}
# ]

# geometry = Geometry(r_coordinates, z_coordinates, domain_params)
# problem = MEEMProblem(geometry)
# engine = MEEMEngine([problem])

# if st.sidebar.button("Show non-multi Domain Parameters"):
#     # Display the domain configuration
#     st.write("Domain Parameters:", domain_params)


# generated_A = engine.assemble_A(problem)
# if st.sidebar.button("Generate A Matrix"):
#     st.header("Matrix and Vector Assembly")
#     visualize_A_matrix(generated_A, title="Generated A Matrix")

# # Generate and verify b vector
# # keep
# generated_b = engine.assemble_b(problem)
# # if np.allclose(generated_b, expected_b, atol=1e-3):
# #     st.success("b vector matches expected values.")
# # else:
# #     st.error("b vector does not match expected values.")

# # Save matching results of b vector
# is_within_threshold_b = np.isclose(expected_b, generated_b, atol=threshold)

# # 2.8801
# #print(m_k(1))
# # 6.1538
# #print(m_k(2)) 
# # 9.3340
# #print(m_k(3)) 

# # Solve the system A x = b
# X = linalg.solve(generated_A, generated_b)

# # Extract coefficients from X (assuming X is already defined)
# # keep
# C_1n_1s = X[:N]
# C_1n_2s = X[N:N+M]
# C_2n_2s = X[N+M:N+2*M]
# C_2n_1s = np.zeros(N, dtype=complex)  # Assuming C_2n_1s are zeros
# B_ks = X[N+2*M:]

# # Spatial grid and potential field plotting
# #keep
# r_vec = np.linspace(0, 2 * a2, spatial_res)
# z_vec = np.linspace(-h, 0, spatial_res)
# R, Z = np.meshgrid(r_vec, z_vec)
    
# # Define regions
# # keep
# region1 = (R <= a1) & (Z < -d1)
# region2 = (R > a1) & (R <= a2) & (Z < -d2)
# regione = R > a2
# region_body = ~region1 & ~region2 & ~regione  # The body of the cylinder

# # Initialize potential arrays
# # keep
# phiH = np.zeros_like(R, dtype=complex)
# phiP = np.zeros_like(R, dtype=complex)

# # Compute the homogeneous potential in each region
# # Region 1 (using phi_h_n_i1_func)
# #keep
# for n in range(N):
#     phiH[region1] += phi_h_n_i1_func(n, R[region1], Z[region1])

# # Region 2 (using phi_h_m_i2_func)
# # keep
# for m in range(M):
#     phiH[region2] += phi_h_m_i2_func(m, R[region2], Z[region2])

# # Exterior region (using phi_e_k_func)
# for k in range(K):
#     phiH[regione] += phi_e_k_func(k, R[regione], Z[regione])

# # Compute the particular potential in each region (using phi_p_i1 and phi_p_i2)
# # keep
# phiP[region1] = phi_p_i1(R[region1], Z[region1])
# phiP[region2] = phi_p_i2(R[region2], Z[region2])
# phiP[regione] = 0  # No particular potential in the exterior region

# # Compute the total potential
# #keep
# phi = phiH + phiP

# # Now you have the actual potential data in the variable 'phi'
    
# # Extract potentials in different regions for matching
# #keep
# phi1 = np.where(region1, phi, np.nan)
# phi2 = np.where(region2, phi, np.nan)
# phie = np.where(regione, phi, np.nan)
# if st.sidebar.button("Plot matching at interfaces"):
#     # Plot matching at interfaces
#     plot_matching(phi1, phi2, phie, a1, a2, R, Z, 'Potential')
    
# # Additional verification: Continuity at interfaces
# def verify_continuity_at_interfaces():
#     # Potential at r = a1 from region1 and region2
#     idx_a1 = np.argmin(np.abs(R[0, :] - a1))
#     phi_a1_region1 = phi1[:, idx_a1]
#     phi_a1_region2 = phi2[:, idx_a1]

#     # Compute the difference
#     difference_a1 = np.abs(phi_a1_region1 - phi_a1_region2)

#     # Potential at r = a2 from region2 and exterior region
#     idx_a2 = np.argmin(np.abs(R[0, :] - a2))
#     phi_a2_region2 = phi2[:, idx_a2]
#     phi_a2_exterior = phie[:, idx_a2]

#     # Compute the difference
#     difference_a2 = np.abs(phi_a2_region2 - phi_a2_exterior)

#     # Plot the differences
#     plt.figure(figsize=(8, 6))
#     plt.plot(Z[:, 0], difference_a1, label='Difference at r = a1')
#     plt.plot(Z[:, 0], difference_a2, label='Difference at r = a2')
#     plt.legend()
#     plt.xlabel('Z')
#     plt.ylabel('Potential Difference')
#     plt.title('Continuity of Potential at Interfaces')
#     st.pyplot(plt)
#     plt.close()
    
# if st.sidebar.button("Plot the Potentials"):
#     # Assume phi, R, and Z have been calculated/generated elsewhere in the app
#     title = "Potentials"
#     # Plot the potentials
#     plot_potential(phiH, R, Z, 'Homogeneous Potential')
#     plot_potential(phiP, R, Z, 'Particular Potential')
#     plot_potential(phi, R, Z, 'Total Potential')

# if st.sidebar.button("Generate Potential Field Plots"):
#     # Assume phi, R, and Z have been calculated/generated elsewhere in the app
#     title = "Potentials"
#     # Plot the potentials
#     # Plot Homogeneous Potential
#     plot_potential_field(R, Z, phiH, title="Homogeneous Potential")

#     # Plot Particular Potential
#     plot_potential_field(R, Z, phiP, title="Particular Potential")

#     # Plot Total Potential
#     plot_potential_field(R, Z, phi, title="Total Potential")



# if st.sidebar.button("Additional verification: Continuity at interfaces"):
#     verify_continuity_at_interfaces()
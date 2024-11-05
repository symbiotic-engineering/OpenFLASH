import sys
import os
import numpy as np
import pandas as pd
from scipy import linalg
from scipy.integrate import quad
import matplotlib.pyplot as plt

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.append(src_path)
from equations import (
    phi_p_i1, phi_p_i2, diff_phi_i1, diff_phi_i2, Z_n_i1, Z_n_i2, Z_n_e,
    m_k, Lambda_k_r, diff_Lambda_k_a2, R_1n_1, R_1n_2, R_2n_2,
    diff_R_1n_1, diff_R_1n_2, diff_R_2n_2,R_2n_1
)

from multi_constants import h,d,a
from multi_equations import *
from math import sqrt, cosh, cos, sinh, sin, pi
from scipy.optimize import newton, minimize_scalar
from scipy.special import hankel1 as besselh
from scipy.special import iv as besseli
from scipy.special import kv as besselk
import scipy.integrate as integrate


# Set path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.append(src_path)

from meem_engine import MEEMEngine
from meem_problem import MEEMProblem
from geometry import Geometry

np.set_printoptions(threshold=np.inf, linewidth=np.inf, precision=8, suppress=True)

# Define visualization function for A matrix
def visualize_A_matrix(A, title="Matrix Visualization"):
    rows, cols = np.nonzero(A)
    plt.figure(figsize=(6, 6))
    plt.scatter(cols, rows, color='blue', marker='o', s=100)
    plt.gca().invert_yaxis()
    plt.xticks(range(A.shape[1]))
    plt.yticks(range(A.shape[0]))

    # Draw dividing lines to visualize block structure in A matrix
    N, M = 4, 4
    block_dividers = [N, N + M, N + 2 * M]
    for val in block_dividers:
        plt.axvline(val - 0.5, color='black', linestyle='-', linewidth=1)
        plt.axhline(val - 0.5, color='black', linestyle='-', linewidth=1)

    plt.grid(True)
    plt.title(title)
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    plt.show()


# Set expected values
expected_b = np.array([
    0.0069, 0.0120, -0.0030, 0.0013, 
    0.1560, 0.0808, -0.0202, 0.0090, 
    0, -0.1460, 0.0732, -0.0002, 
    -0.4622, -0.2837, 0.1539, -0.0673
], dtype=np.complex128)

expected_A_path = "../value/A_values.csv"
df = pd.read_csv(expected_A_path, header=None)

# Convert to complex
def to_complex(val):
    try:
        return np.complex128(val)
    except ValueError:
        return np.nan + 1j * np.nan
    
df_complex = df.applymap(to_complex)
expected_A = df_complex.to_numpy()
expected_A[-4][-4] = np.complex128(-0.45178 + 1.0741j)

# Set tolerance levels
tolerance = 1e-3
threshold = 0.01

# Geometry configuration and problem instance
a2=1.0
a1=0.5
h=1.001
d1=0.5
d2=0.25
r_coordinates = {'a1': 0.5, 'a2': 1.0}
z_coordinates = {'h': 1.001}
domain_params = [
    {'number_harmonics': 4, 'height': 1, 'radial_width': 0.5, 'top_BC': None, 'bottom_BC': None, 'category': 'inner', 'di': 0.5},
    {'number_harmonics': 4, 'height': 1, 'radial_width': 1.0, 'top_BC': None, 'bottom_BC': None, 'category': 'outer', 'di': 0.25},
    {'number_harmonics': 4, 'height': 1, 'radial_width': 1.5, 'top_BC': None, 'bottom_BC': None, 'category': 'exterior'}
]
geometry = Geometry(r_coordinates, z_coordinates, domain_params)
problem = MEEMProblem(geometry)
engine = MEEMEngine([problem])

# Generate and verify A matrix
generated_A = engine.assemble_A(problem)
visualize_A_matrix(generated_A, title="Generated A Matrix")

# Save A matrix to file
np.savetxt("../value/A.txt", generated_A)

# Set threshold and check matches
threshold = 0.001
is_within_threshold = np.isclose(expected_A, generated_A, rtol=threshold)

# Save matching results of A matrix to file
np.savetxt("../value/A_match.txt", is_within_threshold)

# Display indices and values of mismatches and plot mismatched positions
rows, cols = np.nonzero(~is_within_threshold)
# Plotting
plt.figure(figsize=(6, 6))
plt.scatter(cols, rows, color='blue', marker='o', s=100) 
plt.gca().invert_yaxis()
plt.xticks(range(is_within_threshold.shape[1]))
plt.yticks(range(is_within_threshold.shape[0]))

# N = is_within_threshold.shape[1] // 3
# M = is_within_threshold.shape[0] // 3
cols = [4, 8, 12]

for val in cols:
    plt.axvline(val - 0.5, color='black', linestyle='-', linewidth=1)
    plt.axhline(val - 0.5, color='black', linestyle='-', linewidth=1)

plt.grid(True)
plt.title('Non-Zero Entries of the Matrix Not Matching the Threshold')
plt.xlabel('Column Index')
plt.ylabel('Row Index')
plt.show()

# Generate and verify b vector
generated_b = engine.assemble_b(problem)
try:
    np.testing.assert_allclose(generated_b, expected_b, atol=tolerance, err_msg="b vector does not match expected values")
    print("b vector matches successfully.")
except AssertionError as e:
    print("b vector does not match expected values. Details:")
    print(e)

# Save matching results of b vector
is_within_threshold_b = np.isclose(expected_b, generated_b, atol=threshold)
np.savetxt("b_match.txt", is_within_threshold_b, fmt='%d')
print("b vector matching results saved to b_match.txt")

# Solve the system A x = b
X = linalg.solve(generated_A, generated_b)

# Extract coefficients
C_1n_1s = X[:4]
C_1n_2s = X[4:4+4]
C_2n_2s = X[4+4:4+2*4]
C_2n_1s = np.zeros(4, dtype=complex)  # Assuming C_2n_1s are zeros
B_ks = X[4+2*4:]

# Define functions for the potentials
def phi_h_n_i1_func(n, r, z):
    return (C_1n_1s[n] * R_1n_1(n, r) + C_2n_1s[n] * R_2n_1(n)) * Z_n_i1(n, z)

def phi_h_m_i2_func(m, r, z):
    return (C_1n_2s[m] * R_1n_2(m, r) + C_2n_2s[m] * R_2n_2(m, r)) * Z_n_i2(m, z)

def phi_e_k_func(k, r, z):
    return B_ks[k] * Lambda_k_r(k, r) * Z_n_e(k, z)

# Create spatial grid
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
# Region 1
for n in range(4):
    phiH[region1] += phi_h_n_i1_func(n, R[region1], Z[region1])

# Region 2
for m in range(4):
    phiH[region2] += phi_h_m_i2_func(m, R[region2], Z[region2])

# Exterior region
for k in range(4):
    phiH[regione] += phi_e_k_func(k, R[regione], Z[regione])

# Compute the particular potential in each region
phiP[region1] = phi_p_i1(R[region1], Z[region1])
phiP[region2] = phi_p_i2(R[region2], Z[region2])
phiP[regione] = 0  # No particular potential in the exterior region

# Compute the total potential
phi = phiH + phiP

# Plotting functions
def plot_potential(phi, R, Z, title):
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
    plt.show()

def plot_matching(phi1, phi2, phie, a1, a2, R, Z, title):
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
    plt.show()

# Plot the potentials
plot_potential(phiH, R, Z, 'Homogeneous Potential')
plot_potential(phiP, R, Z, 'Particular Potential')
plot_potential(phi, R, Z, 'Total Potential')

# Extract potentials in different regions for matching
phi1 = np.where(region1, phi, np.nan)
phi2 = np.where(region2, phi, np.nan)
phie = np.where(regione, phi, np.nan)

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
    plt.show()

verify_continuity_at_interfaces()



###############################################
#multi meem with slant
# Multi-region case
# Update the engine to use multi-region functionality
NMK = [6, 10, 20, 20]  
boundary_count = len(NMK) - 1

a = [0.5, 1.0, 1.5] 
d = [0.5, 0.25, 0.1]  

# construct domain_params_multi
domain_params_multi = []
for idx, n_harmonics in enumerate(NMK):
    if idx == 0:
        category = 'inner'
        di = d[idx]
        radial_width = a[idx]
    elif idx == len(NMK) - 1:
        category = 'exterior'
        di = None  # exterior region doesn't have di
        radial_width = a[-1] + 0.5  
    else:
        category = 'outer'
        di = d[idx]
        radial_width = a[idx]
    domain_params_multi.append({
        'number_harmonics': n_harmonics,
        'height': h,
        'radial_width': radial_width,
        'top_BC': None,
        'bottom_BC': None,
        'category': category,
        'di': di
    })

r_coordinates_multi = {'a' + str(idx): a_val for idx, a_val in enumerate(a)}
z_coordinates_multi = {'h': 100}

geometry_multi = Geometry(r_coordinates_multi, z_coordinates_multi, domain_params_multi)
problem_multi = MEEMProblem(geometry_multi)
problem_multi.multi_region = True
engine_multi = MEEMEngine([problem_multi], multi_region=True)

A = engine_multi.assemble_A(problem_multi)
b = engine_multi.assemble_b(problem_multi)

print(f"A shape: {A.shape}")
print(f"b shape: {b.shape}")
rank = np.linalg.matrix_rank(A)
print(f"Rank of A: {rank}, Expected: {A.shape[0]}")

if rank < A.shape[0]:
    print("Matrix A is singular or rank deficient.")
else:
    X = linalg.solve(A, b)
    print("Linear system solved successfully.")

    NMK = [domain.number_harmonics for domain in problem_multi.domain_list.values()]
    boundary_count = len(NMK) - 1
    # Extract coefficients from the solution vector X
    Cs = []
    row = 0
    Cs.append(X[:NMK[0]])
    row += NMK[0]
    for i in range(1, boundary_count):
        Cs.append(X[row: row + NMK[i] * 2])
        row += NMK[i] * 2
    Cs.append(X[row:])

    # Define potential functions
    def phi_h_n_inner_func(n, r, z):
        return (Cs[0][n] * R_1n(n, r, 0)) * Z_n_i(n, z, 0)

    def phi_h_m_i_func(i, m, r, z):
        C_1n = Cs[i][:NMK[i]]
        C_2n = Cs[i][NMK[i]:]
        return (C_1n[m] * R_1n(m, r, i) + C_2n[m] * R_2n(m, r, i)) * Z_n_i(m, z, i)

    def phi_e_k_func(k, r, z):
        return Cs[-1][k] * Lambda_k(k, r) * Z_n_e(k, z)

    r_vec = lambda spatial_res: np.linspace(2 * a[-1] / spatial_res, 2*a[-1], spatial_res)
    z_vec = lambda spatial_res: np.linspace(-h, 0, spatial_res)
    R, Z = np.meshgrid(r_vec(spatial_res=50), z_vec(spatial_res=50))

    # Define regions based on the grid
    regions = []
    regions.append((R <= a[0]) & (Z < -d[0]))
    for i in range(1, boundary_count):
        regions.append((R > a[i-1]) & (R <= a[i]) & (Z < -d[i]))
    regions.append(R > a[-1])

    # Initialize potential arrays
    phi = np.full_like(R, np.nan + np.nan*1j, dtype=complex) 
    phiH = np.full_like(R, np.nan + np.nan*1j, dtype=complex) 
    phiP = np.full_like(R, np.nan + np.nan*1j, dtype=complex) 

    # Calculate homogeneous potential in each region
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

    phi_p_i_vec = np.vectorize(phi_p_i)

    phiP[regions[0]] = phi_p_i_vec(d[0], R[regions[0]], Z[regions[0]])
    for i in range(1, boundary_count):
        phiP[regions[i]] = phi_p_i_vec(d[i], R[regions[i]], Z[regions[i]])
    phiP[regions[-1]] = 0

    phi = phiH + phiP

    #indicate if the region is slanted or not
    slant = [False, False, True]
    #indicate if the region is heaving or not
    heave = [False, True, True]
    size = NMK[0] + NMK[-1] + 2 * sum(NMK[1:len(NMK) - 1])
    hydro_terms = np.zeros((size - NMK[-1]), dtype=complex)
    vel_z = np.zeros(len(NMK)-1, dtype=complex)

    col = 0
    region_indx = 0
    for n in range(NMK[0]):
        if slant[0]:
            #need definition
            pass 
        else:
            hydro_terms[n] = int_R_1n(0, n)* X[n] * z_n_d(n)
    col += NMK[0]
    region_indx += 1
    for i in range(1, boundary_count):
        M = NMK[i]
        if slant[i]:
            #bottom_corner
            phi_corner_H = sum(phi_h_m_i_func(i, n, a[i-1], -d[i-1]) for n in range(NMK[1]))
            phi_corner_P = phi_p_i(d[i], a[i-1], -d[i-1])
            phi_bttm_corner = phi_corner_H + phi_corner_P

            #top_corner
            phi_corner_H = sum(phi_h_m_i_func(i, n, a[i], -d[i]) for n in range(NMK[1]))
            phi_corner_P = phi_p_i(d[i], a[i], -d[i])
            phi_top_corner = phi_corner_H + phi_corner_P
            #slant velocity z component approximation
            vel_z[i] = (phi_top_corner - phi_bttm_corner)*(d[i-1]-d[i])/(abs(d[i-1]-d[i])**2+(a[i]-a[i-1])**2)
            for m in range(M):
                hydro_terms[col + m] = vel_z[i]*int_R_1n(i, m)* X[col + m] * z_n_d(m)
                hydro_terms[col + M + m] = vel_z[i]*int_R_2n(i, m)* X[col + M + m] * z_n_d(m)
        else:
            for m in range(M):
                hydro_terms[col + m] = int_R_1n(i, m)* X[col + m] * z_n_d(m)
                hydro_terms[col + M + m] = int_R_2n(i, m)* X[col + M + m] * z_n_d(m)
        col += 2 * M
        region_indx += 1

    hydro_p_terms = np.zeros(boundary_count, dtype=complex)
    for i in range(boundary_count):
        if not heave[i]:
            hydro_p_terms[i] = 0
        elif slant[i]:
            hydro_p_terms[i] = vel_z[i]*int_phi_p_i_no_coef(i)
        else:
            hydro_p_terms[i] = int_phi_p_i_no_coef(i)

    #when i2 is heaving
    hydro_coef =2*pi*(sum(hydro_terms) + sum(hydro_p_terms))
    hydro_coef_real = hydro_coef.real
    hydro_coef_imag = hydro_coef.imag/omega


    hydro_coef_nondim = h**3/(a[-1]**3 * pi)*hydro_coef

    print("real", hydro_coef_real)
    print("imag", hydro_coef_imag)
    print(hydro_coef_nondim)

    def plot_potential(field, R, Z, title):
        plt.figure(figsize=(8, 6))
        plt.contourf(R, Z, field, levels=50, cmap='viridis')
        plt.colorbar()
        plt.title(title)
        plt.xlabel('Radial Distance (R)')
        plt.ylabel('Axial Distance (Z)')
        plt.show()

    def plot_velocity(v_r, v_z, R, Z):
        plt.figure(figsize=(8, 6))
        plt.streamplot(R, Z, v_r, v_z, color='magenta', density=2)
        plt.title('Velocity Field')
        plt.xlabel('Radial Distance (R)')
        plt.ylabel('Axial Distance (Z)')
        plt.show()

    plot_potential(np.real(phiH), R, Z, 'Homogeneous Potential')
    plot_potential(np.imag(phiH), R, Z, 'Homogeneous Potential Imaginary')

    plot_potential(np.real(phi), R, Z, 'Total Potential')
    plot_potential(np.imag(phi), R, Z, 'Total Potential Imaginary')

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
        plt.show()
    plot_potential(phiH, R, Z, 'Homogeneous Potential')
    plot_potential(phiP, R, Z, 'Particular Potential')
    plot_potential(phi, R, Z, 'Total Potential')















    # # Define plotting function
    # def plot_potential(field, R, Z, region_body, title):
    #     plt.figure(figsize=(8, 6))
    #     masked_field = np.ma.array(np.real(field), mask=region_body)
    #     contour_real = plt.contourf(R, Z, masked_field, levels=50, cmap='viridis')
    #     plt.colorbar()
    #     plt.title(f'{title} - Real Part')
    #     plt.xlabel('Radial Distance (R)')
    #     plt.ylabel('Axial Distance (Z)')
    #     plt.show()

    # # Plot the potentials
    # plot_potential(phiH, R, Z, region_body, 'Homogeneous Potential (Multi-Region)')
    # plot_potential(phiP, R, Z, region_body, 'Particular Potential (Multi-Region)')
    # plot_potential(phi, R, Z, region_body, 'Total Potential (Multi-Region)')

    # # Verify continuity at interfaces
    # def verify_continuity_at_interfaces_multi():
    #     Z_line = Z[:, 0]
    #     for i in range(boundary_count):
    #         idx_a = np.argmin(np.abs(r_vec - a[i]))
    #         phi_left = phi[:, idx_a - 1]
    #         phi_right = phi[:, idx_a + 1]
    #         difference = np.abs(phi_left - phi_right)
    #         plt.figure(figsize=(8, 6))
    #         plt.plot(Z_line, difference, label=f'Difference at r = a[{i}]')
    #         plt.legend()
    #         plt.xlabel('Z')
    #         plt.ylabel('Potential Difference')
    #         plt.title(f'Continuity of Potential at Interface r = {a[i]}')
    #         plt.show()

    # verify_continuity_at_interfaces_multi()
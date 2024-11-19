import sys
import os
import numpy as np
import pandas as pd
from scipy import linalg
from scipy.integrate import quad
import matplotlib.pyplot as plt

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.append(src_path)

from equations import *
from constants import *
from meem_engine import MEEMEngine
from meem_problem import MEEMProblem
from geometry import Geometry

np.set_printoptions(threshold=np.inf, linewidth=np.inf, precision=8, suppress=True)
'''
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


N = 4
M = 4
K = 4

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
a_basic=[0.5,1]
h_basic=1.001
r_coordinates = {'a': a_basic}
z_coordinates = {'h': h_basic}
domain_params = [
    {'number_harmonics': N, 'height': 1, 'radial_width': 0.5, 'top_BC': None, 'bottom_BC': None, 'category': 'inner', 'di': 0.5},
    {'number_harmonics': M, 'height': 1, 'radial_width': 1.0, 'top_BC': None, 'bottom_BC': None, 'category': 'outer', 'di': 0.25},
    {'number_harmonics': K, 'height': 1, 'radial_width': 1.5, 'top_BC': None, 'bottom_BC': None, 'category': 'exterior'}
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
np.save("generated_A.npy", generated_A)
np.save("generated_b.npy", generated_b)
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

# 2.8801
print(m_k(1))
# 6.1538
print(m_k(2)) 
# 9.3340
print(m_k(3)) 

# Solve the system A x = b
X = linalg.solve(generated_A, generated_b)

# Extract coefficients
C_1n_1s = X[:N]
C_1n_2s = X[N:N+M]
C_2n_2s = X[N+M:N+2*M]
C_2n_1s = np.zeros(N, dtype=complex)  # Assuming C_2n_1s are zeros
B_ks = X[N+2*M:]

# Define functions for the potentials
def phi_h_n_i1_func(n, r, z):
    return (C_1n_1s[n] * R_1n_1(n, r) + C_2n_1s[n] * R_2n_1(n)) * Z_n_i1(n, z)

def phi_h_m_i2_func(m, r, z):
    return (C_1n_2s[m] * R_1n_2(m, r) + C_2n_2s[m] * R_2n_2(m, r)) * Z_n_i2(m, z)

def phi_e_k_func(k, r, z):
    return B_ks[k] * Lambda_k_r(k, r) * Z_n_e(k, z)

# Create spatial grid
spatial_res = 50
r_vec = np.linspace(0, 2 * a_basic[1], spatial_res)
z_vec = np.linspace(-h_basic, 0, spatial_res)
R, Z = np.meshgrid(r_vec, z_vec)

# Define regions
region1 = (R <= a_basic[0]) & (Z < -0.5)
region2 = (R > a_basic[0]) & (R <= a_basic[1]) & (Z < -0.25)
regione = R > a_basic[1]
region_body = ~region1 & ~region2 & ~regione  # The body of the cylinder

# Initialize potential arrays
phiH = np.zeros_like(R, dtype=complex)
phiP = np.zeros_like(R, dtype=complex)

# Compute the homogeneous potential in each region
# Region 1
for n in range(N):
    phiH[region1] += phi_h_n_i1_func(n, R[region1], Z[region1])

# Region 2
for m in range(M):
    phiH[region2] += phi_h_m_i2_func(m, R[region2], Z[region2])

# Exterior region
for k in range(K):
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
plot_matching(phi1, phi2, phie, a_basic[0], a_basic[1], R, Z, 'Potential')

'''

#############################################################################
#multi-region
from multi_constants import *
from multi_equations import *

np.set_printoptions(threshold=np.inf, linewidth=np.inf, precision=8, suppress=True)

# Number of harmonics in each region
NMK = [20, 20, 20, 20]  # Adjust these values as needed
boundary_count = len(NMK) - 1


# Create domain parameters
domain_params = []
for idx in range(len(NMK)):
    params = {
        'number_harmonics': NMK[idx],
        'height': h - d[idx] if idx < len(d) else h,
        'radial_width': a[idx] if idx < len(a) else a[-1]*1.5,
        'top_BC': None,
        'bottom_BC': None,
        'category': 'multi',  # Adjust category as needed
        'di': d[idx] if idx < len(d) else 0,
        'a': a[idx] if idx < len(a) else a[-1]*1.5,
        'heaving': heaving[idx] if idx < len(heaving) else False,
        'slant': [0, 0, 1]  # Set True if the region is slanted
    }
    domain_params.append(params)

# Create Geometry object
r_coordinates = {'a': a}
z_coordinates = {'h': h}
geometry = Geometry(r_coordinates, z_coordinates, domain_params)

# Create MEEMProblem object
problem = MEEMProblem(geometry)

# Create MEEMEngine object
engine = MEEMEngine([problem])

# Assemble A matrix and b vector using multi-region methods
A = engine.assemble_A_multi(problem)
b = engine.assemble_b_multi(problem)

# Solve the linear system A x = b
X = linalg.solve(A, b)

size = NMK[0] + NMK[-1] + 2 * sum(NMK[1:len(NMK) - 1])
hydro_terms = np.zeros((size - NMK[-1]), dtype=complex)

col = 0
for n in range(NMK[0]):
    hydro_terms[n] = heaving[0] * int_R_1n(0, n)* X[n] * z_n_d(n)
col += NMK[0]
for i in range(1, boundary_count):
    M = NMK[i]
    for m in range(M):
        hydro_terms[col + m] = heaving[i] * int_R_1n(i, m)* X[col + m] * z_n_d(m)
        hydro_terms[col + M + m] = heaving[i] * int_R_2n(i, m)* X[col + M + m] * z_n_d(m)
    col += 2 * M

hydro_p_terms = np.zeros(boundary_count, dtype=complex)
for i in range(boundary_count):
    hydro_p_terms[i] = heaving[i] * int_phi_p_i_no_coef(i)

hydro_coef =2*pi*(sum(hydro_terms) + sum(hydro_p_terms))
hydro_coef_real = hydro_coef.real
hydro_coef_imag = hydro_coef.imag/omega

# find maximum heaving radius
max_rad = a[0]
for i in range(boundary_count - 1, 0, -1):
    if heaving[i]:
        max_rad = a[i]
        break

hydro_coef_nondim = h**3/(max_rad**3 * pi)*hydro_coef

print("real", hydro_coef_real)
print("imag", hydro_coef_imag)
print(hydro_coef_nondim)

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

#phi_h_n_i1s = np.vectorize(phi_h_n_i1_func, excluded=['n'], signature='(),(),()->()')
#phi_h_m_i2s = np.vectorize(phi_h_m_i2_func, excluded=['m'], signature='(),(),()->()')
#phi_e_ks = np.vectorize(phi_e_k_func, excluded=['k'], signature='(),(),()->()')

spatial_res=50
r_vec = np.linspace(2 * a[-1] / spatial_res, 2*a[-1], spatial_res)
z_vec = np.linspace(-h, 0, spatial_res)

#add values at the radii
a_eps = 1.0e-4
for i in range(len(a)):
    r_vec = np.append(r_vec, a[i]*(1-a_eps))
    r_vec = np.append(r_vec, a[i]*(1+a_eps))
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

# region_body = ~region1 & ~region2 & ~regione


phi = np.full_like(R, np.nan + np.nan*1j, dtype=complex) 
phiH = np.full_like(R, np.nan + np.nan*1j, dtype=complex) 
phiP = np.full_like(R, np.nan + np.nan*1j, dtype=complex) 

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

phiP[regions[0]] = heaving[0] * phi_p_i_vec(d[0], R[regions[0]], Z[regions[0]])
for i in range(1, boundary_count):
    phiP[regions[i]] = heaving[i] * phi_p_i_vec(d[i], R[regions[i]], Z[regions[i]])
phiP[regions[-1]] = 0

phi = phiH + phiP
def plot_potential(field, R, Z, title):
    plt.figure(figsize=(8, 6))
    plt.contourf(R, Z, field, levels=50, cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('Radial Distance (R)')
    plt.ylabel('Axial Distance (Z)')
    plt.show()

plot_potential(np.real(phiH), R, Z, 'Homogeneous Potential')
plot_potential(np.imag(phiH), R, Z, 'Homogeneous Potential Imaginary')

plot_potential(np.real(phiP), R, Z, 'Particular Potential')
plot_potential(np.imag(phiP), R, Z, 'Particular Potential Imaginary')

plot_potential(np.real(phi), R, Z, 'Total Potential')
plot_potential(np.imag(phi), R, Z, 'Total Potential Imaginary')

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

# import numpy as np
# import matplotlib.pyplot as plt
# from multi_constants import *
# from multi_equations import *
# from scipy import linalg

# # Set printing options
# np.set_printoptions(threshold=np.inf, linewidth=np.inf, precision=8, suppress=True)

# def initialize_geometry(NMK, h, d, a, heaving):
#     """Initialize geometry and domain parameters."""
#     domain_params = []
#     for idx in range(len(NMK)):
#         params = {
#             'number_harmonics': NMK[idx],
#             'height': h - d[idx] if idx < len(d) else h,
#             'radial_width': a[idx] if idx < len(a) else a[-1] * 1.5,
#             'top_BC': None,
#             'bottom_BC': None,
#             'category': 'multi',
#             'di': d[idx] if idx < len(d) else 0,
#             'a': a[idx] if idx < len(a) else a[-1] * 1.5,
#             'heaving': heaving[idx] if idx < len(heaving) else False,
#             'slant': [0, 0, 1]
#         }
#         domain_params.append(params)

#     r_coordinates = {'a': a}
#     z_coordinates = {'h': h}
#     return Geometry(r_coordinates, z_coordinates, domain_params)

# def assemble_matrices(engine, problem):
#     """Assemble matrix A and vector b."""
#     A = engine.assemble_A_multi(problem)
#     b = engine.assemble_b_multi(problem)
#     return A, b

# def solve_system(A, b):
#     """Solve the linear system A * x = b."""
#     return linalg.solve(A, b)

# def calculate_hydro_coefficients(X, NMK, heaving, a, d, h, omega):
#     """Calculate hydrodynamic coefficients."""
#     boundary_count = len(NMK) - 1
#     size = NMK[0] + NMK[-1] + 2 * sum(NMK[1:len(NMK) - 1])
#     hydro_terms = np.zeros((size - NMK[-1]), dtype=complex)

#     col = 0
#     for n in range(NMK[0]):
#         hydro_terms[n] = heaving[0] * int_R_1n(0, n) * X[n] * z_n_d(n)
#     col += NMK[0]

#     for i in range(1, boundary_count):
#         M = NMK[i]
#         for m in range(M):
#             hydro_terms[col + m] = heaving[i] * int_R_1n(i, m) * X[col + m] * z_n_d(m)
#             hydro_terms[col + M + m] = heaving[i] * int_R_2n(i, m) * X[col + M + m] * z_n_d(m)
#         col += 2 * M

#     hydro_p_terms = np.zeros(boundary_count, dtype=complex)
#     for i in range(boundary_count):
#         hydro_p_terms[i] = heaving[i] * int_phi_p_i_no_coef(i)

#     hydro_coef = 2 * np.pi * (sum(hydro_terms) + sum(hydro_p_terms))
#     max_rad = max(a[i] for i in range(boundary_count) if heaving[i])
#     hydro_coef_nondim = h ** 3 / (max_rad ** 3 * np.pi) * hydro_coef

#     return hydro_coef.real, hydro_coef.imag / omega, hydro_coef_nondim

# ### Write Tests for pytest


# import numpy as np
# import pytest
# import matplotlib.pyplot as plt
# from multi_constants import *
# from multi_equations import *
# from scipy import linalg

# def test_initialize_geometry():
#     """Test geometry initialization."""
#     # Define test parameters
#     NMK = [20, 20, 20, 20]
#     h = 10.0
#     d = [2.0, 4.0, 6.0, 8.0]
#     a = [1.0, 1.5, 2.0, 2.5]
#     heaving = [True, False, True, False]

#     # Initialize geometry
#     geometry = initialize_geometry(NMK, h, d, a, heaving)

#     # Assertions to check correct initialization
#     assert geometry is not None
#     assert len(geometry.domain_params) == len(NMK)
#     for idx, params in enumerate(geometry.domain_params):
#         assert params['number_harmonics'] == NMK[idx]
#         assert params['height'] == h - d[idx]
#         assert params['radial_width'] == a[idx]
#         assert params['heaving'] == heaving[idx]

# def test_assemble_matrices():
#     """Test matrix assembly."""
#     # Define test parameters
#     NMK = [20, 20, 20, 20]
#     h = 10.0
#     d = [2.0, 4.0, 6.0, 8.0]
#     a = [1.0, 1.5, 2.0, 2.5]
#     heaving = [True, False, True, False]

#     # Initialize geometry and problem
#     geometry = initialize_geometry(NMK, h, d, a, heaving)
#     problem = MEEMProblem(geometry)
#     engine = MEEMEngine([problem])

#     # Assemble matrices
#     A, b = assemble_matrices(engine, problem)

#     # Assertions to check matrices
#     assert A is not None
#     assert b is not None
#     assert A.shape[0] == A.shape[1], "Matrix A is not square"
#     assert A.shape[0] == len(b), "Matrix A and vector b dimensions do not match"

# def test_solve_system():
#     """Test solving linear systems."""
#     # Create a simple system to test
#     A = np.array([[2, 1], [1, 3]], dtype=float)
#     b = np.array([5, 6], dtype=float)

#     # Solve the system
#     X = solve_system(A, b)

#     # Expected solution
#     expected_X = np.array([1, 2], dtype=float)

#     # Assertions to compare the solutions
#     np.testing.assert_allclose(X, expected_X, atol=1e-6)

# def test_calculate_hydro_coefficients():
#     """Test hydrodynamic coefficient calculation."""
#     # Define test parameters
#     NMK = [20, 20, 20, 20]
#     h = 10.0
#     d = [2.0, 4.0, 6.0, 8.0]
#     a = [1.0, 1.5, 2.0, 2.5]
#     heaving = [True, False, True, False]
#     omega = 1.0  # Wave frequency

#     # Initialize geometry and problem
#     geometry = initialize_geometry(NMK, h, d, a, heaving)
#     problem = MEEMProblem(geometry)
#     engine = MEEMEngine([problem])

#     # Assemble matrices and solve
#     A, b = assemble_matrices(engine, problem)
#     X = solve_system(A, b)

#     # Calculate hydrodynamic coefficients
#     real_coef, imag_coef, nondim_coef = calculate_hydro_coefficients(X, NMK, heaving, a, d, h, omega)

#     # Assertions to check coefficients
#     assert real_coef is not None
#     assert imag_coef is not None
#     assert nondim_coef is not None
#     assert isinstance(real_coef, float), "Real coefficient is not a float"
#     assert isinstance(imag_coef, float), "Imaginary coefficient is not a float"

# def test_potential_plots():
#     """Test and plot potential fields."""
#     # Define test parameters with reduced harmonics for faster plotting
#     NMK = [5, 5, 5, 5]
#     h = 10.0
#     d = [2.0, 4.0, 6.0, 8.0]
#     a = [1.0, 1.5, 2.0, 2.5]
#     heaving = [True, False, True, False]
#     omega = 1.0

#     # Initialize geometry and problem
#     geometry = initialize_geometry(NMK, h, d, a, heaving)
#     problem = MEEMProblem(geometry)
#     engine = MEEMEngine([problem])

#     # Assemble matrices and solve
#     A, b = assemble_matrices(engine, problem)
#     X = solve_system(A, b)

#     # Generate spatial grid
#     spatial_res = 50
#     r_vec = np.linspace(0.01, 2.5, spatial_res)
#     z_vec = np.linspace(-h, 0, spatial_res)
#     R, Z = np.meshgrid(r_vec, z_vec)

#     # Define regions
#     boundary_count = len(NMK) - 1
#     regions = []
#     regions.append((R <= a[0]) & (Z < -d[0]))
#     for i in range(1, boundary_count):
#         regions.append((R > a[i - 1]) & (R <= a[i]) & (Z < -d[i]))
#     regions.append(R > a[-1])

#     # Initialize potentials
#     phiH = np.zeros_like(R, dtype=complex)
#     phiP = np.zeros_like(R, dtype=complex)

#     # Split coefficients
#     Cs = []
#     row = 0
#     Cs.append(X[:NMK[0]])
#     row += NMK[0]
#     for i in range(1, boundary_count):
#         Cs.append(X[row: row + NMK[i] * 2])
#         row += NMK[i] * 2
#     Cs.append(X[row:])

#     def phi_h_n_inner_func(n, r, z):
#         return (Cs[0][n] * R_1n(n, r, 0)) * Z_n_i(n, z, 0)

#     def phi_h_m_i_func(i, m, r, z):
#         return (Cs[i][m] * R_1n(m, r, i) + Cs[i][NMK[i] + m] * R_2n(m, r, i)) * Z_n_i(m, z, i)

#     def phi_e_k_func(k, r, z):
#         return Cs[-1][k] * Lambda_k(k, r) * Z_n_e(k, z)

#     # Compute homogeneous potential
#     for idx, region in enumerate(regions):
#         if idx == 0:
#             for n in range(NMK[0]):
#                 temp_phiH = phi_h_n_inner_func(n, R[region], Z[region])
#                 phiH[region] += temp_phiH
#         elif idx == len(regions) - 1:
#             for k in range(NMK[-1]):
#                 temp_phiH = phi_e_k_func(k, R[region], Z[region])
#                 phiH[region] += temp_phiH
#         else:
#             i = idx
#             for m in range(NMK[i]):
#                 temp_phiH = phi_h_m_i_func(i, m, R[region], Z[region])
#                 phiH[region] += temp_phiH

#     # Compute particular potential
#     phi_p_i_vec = np.vectorize(phi_p_i)
#     phiP[regions[0]] = heaving[0] * phi_p_i_vec(d[0], R[regions[0]], Z[regions[0]])
#     for i in range(1, boundary_count):
#         phiP[regions[i]] = heaving[i] * phi_p_i_vec(d[i], R[regions[i]], Z[regions[i]])
#     phiP[regions[-1]] = 0

#     # Total potential
#     phi = phiH + phiP

#     # Plotting functions
#     def plot_potential(field, R, Z, title):
#         plt.figure(figsize=(8, 6))
#         plt.contourf(R, Z, field, levels=50, cmap='viridis')
#         plt.colorbar()
#         plt.title(title)
#         plt.xlabel('Radial Distance (R)')
#         plt.ylabel('Vertical Distance (Z)')
#         plt.show()

#     # Plot potentials
#     plot_potential(np.real(phiH), R, Z, 'Homogeneous Potential - Real Part')
#     plot_potential(np.imag(phiH), R, Z, 'Homogeneous Potential - Imaginary Part')
#     plot_potential(np.real(phiP), R, Z, 'Particular Potential - Real Part')
#     plot_potential(np.imag(phiP), R, Z, 'Particular Potential - Imaginary Part')
#     plot_potential(np.real(phi), R, Z, 'Total Potential - Real Part')
#     plot_potential(np.imag(phi), R, Z, 'Total Potential - Imaginary Part')

#     # Assertions to check potentials
#     assert phi.shape == R.shape, "Potential field shape does not match grid shape"

#     # Optionally, add more checks or save additional plots as needed

# def main():
#     test_potential_plots()

# if __name__ == "__main__":
#     main()
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

import capytaine as cpt
from capytaine.bem.airy_waves import airy_waves_potential, airy_waves_velocity, froude_krylov_force

np.set_printoptions(threshold=np.inf, linewidth=np.inf, precision=8, suppress=True)
'''
# Initialize solver
solver = cpt.BEMSolver()

# Function to create a body from profile points
def body_from_profile(x, y, z, nphi):
    xyz = np.array([np.array([x / np.sqrt(2), y / np.sqrt(2), z]) for x, y, z in zip(x, y, z)])  # Scale correction
    body = cpt.FloatingBody(cpt.AxialSymmetricMesh.from_profile(xyz, nphi=nphi))
    return body

def make_body(d, a, mesh_density):
    zt = np.linspace(0, 0, mesh_density)
    rt = np.linspace(0, a[-1], mesh_density)
    top_surface = body_from_profile(rt, rt, zt, mesh_density**2)

    zb = np.linspace(-d[0], -d[0], mesh_density)
    rb = np.linspace(0, a[0], mesh_density)
    bot_surface = body_from_profile(rb, rb, zb, mesh_density**2)

    zo = np.linspace(-d[-1], 0, mesh_density)
    ro = np.linspace(a[-1], a[-1], mesh_density)
    outer_surface = body_from_profile(ro, ro, zo, mesh_density**2)

    bod = top_surface + bot_surface + outer_surface

    for i in range(1, len(a)):
        zs = np.linspace(-d[i-1], -d[i], mesh_density)
        rs = np.linspace(a[i-1], a[i-1], mesh_density)
        side = body_from_profile(rs, rs, zs, mesh_density**2)

        zb = np.linspace(-d[i], -d[i], mesh_density)
        rb = np.linspace(a[i-1], a[i], mesh_density)
        bot = body_from_profile(rb, rb, zb, mesh_density**2)

        bod = bod + side + bot

    return bod

# Function to solve radiation problem and extract hydrodynamic coefficients
def extract_hydro_coeffs(d, a, mesh_density, w, h):
    body = make_body(d, a, mesh_density)
    body.add_translation_dof(name='Heave')
    body = body.immersed_part()

    rad_problem = cpt.RadiationProblem(body=body, wavenumber=w, water_depth=h)
    result = solver.solve(rad_problem, keep_details=True)

    added_mass = result.added_mass
    radiation_damping = result.radiation_damping

    print("Added Mass:")
    print(added_mass)
    print("Radiation Damping:")
    print(radiation_damping)

    return added_mass, radiation_damping

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

def test_main_basic():
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

    script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the current script
    expected_A_path = os.path.join(script_dir, "../value/A_values.csv")
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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.abspath(os.path.join(script_dir, "../value"))
    os.makedirs(output_dir, exist_ok=True)  
    output_file_path = os.path.join(output_dir, "A.txt")
    np.savetxt(output_file_path, generated_A)


    # Set threshold and check matches
    threshold = 0.001
    is_within_threshold = np.isclose(expected_A, generated_A, rtol=threshold)

    # Save matching results of A matrix to file
    output_file_path_A_match = os.path.join(output_dir, "A_match.txt")
    np.savetxt(output_file_path_A_match, is_within_threshold)

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
    output_file_path_b_match = os.path.join(output_dir, "b_match.txt")
    is_within_threshold_b = np.isclose(expected_b, generated_b, atol=threshold)
    np.savetxt(output_file_path_b_match, is_within_threshold_b, fmt='%d')
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

    phi_h_n_i1 = lambda n, r, z:  (C_1n_1s[n] * R_1n_1(n, r) + C_2n_1s(n) * R_2n_1(n)) * Z_n_i1(n, z)
    phi_h_m_i2 = lambda m, r, z: (C_1n_2s[m] * R_1n_2(m, r) + C_2n_2s(m) * R_2n_2(m, r)) * Z_n_i2(m, z)
    phi_e_k = lambda k, r, z: B_ks[k] * Lambda_k_r(k, r) * Z_n_e(k, z)

    # Define functions for the potentials
    def phi_h_n_i1_func(n, r, z):
        return (C_1n_1s[n] * R_1n_1(n, r) + C_2n_1s[n] * R_2n_1(n)) * Z_n_i1(n, z)

    def phi_h_m_i2_func(m, r, z):
        return (C_1n_2s[m] * R_1n_2(m, r) + C_2n_2s[m] * R_2n_2(m, r)) * Z_n_i2(m, z)

    def phi_e_k_func(k, r, z):
        return B_ks[k] * Lambda_k_r(k, r) * Z_n_e(k, z)

    phi_h_n_i1s = np.vectorize(phi_h_n_i1_func, excluded=['n'], signature='(),(),()->()')
    phi_h_m_i2s = np.vectorize(phi_h_m_i2_func, excluded=['m'], signature='(),(),()->()')
    phi_e_ks = np.vectorize(phi_e_k_func, excluded=['k'], signature='(),(),()->()')

    r_vec = lambda spatial_res: np.linspace(2 * a2 / spatial_res, 2*a2, spatial_res)
    z_vec = lambda spatial_res: np.linspace(-h, 0, spatial_res)
    R, Z = np.meshgrid(r_vec(spatial_res=50), z_vec(spatial_res=50))

    regione = R > a2
    region1 = (R <= a1) & (Z < -d1)
    region2 = (R > a1) & (R <= a2) & (Z < -d2)
    region_body = ~region1 & ~region2 & ~regione

    phi = np.full_like(R, np.nan + np.nan*1j, dtype=complex) 
    phiH = np.full_like(R, np.nan + np.nan*1j, dtype=complex) 
    phiP = np.full_like(R, np.nan + np.nan*1j, dtype=complex) 

    for n in range(N):
        temp_phiH = phi_h_n_i1_func(n, R[region1], Z[region1])
        phiH[region1] = temp_phiH if n == 0 else phiH[region1] + temp_phiH


    for m in range(M):
        temp_phiH = phi_h_m_i2_func(m, R[region2], Z[region2])
        phiH[region2] = temp_phiH if m == 0 else phiH[region2] + temp_phiH

    for k in range(K):
        temp_phiH = phi_e_k_func(k, R[regione], Z[regione])
        phiH[regione] = temp_phiH if k == 0 else phiH[regione] + temp_phiH

    phi_p_i1_vec = np.vectorize(phi_p_i1)
    phi_p_i2_vec = np.vectorize(phi_p_i2)

    phiP[region1] = phi_p_i1_vec(R[region1], Z[region1])
    phiP[region2] = phi_p_i2_vec(R[region2], Z[region2])
    phiP[regione] = 0

    phi = phiH + phiP

    ##slant hydro coeffs##

    #finding phi @corner 1
    phi_corner1_H = 0
    for n in range(N):
        temp_phiH = phi_h_n_i1_func(n, a1, -d1)
        phi_corner1_H += temp_phiH

    phi_corner1_P = phi_p_i1(a1, -d1)
    phi_corner1 = phi_corner1_H + phi_corner1_P

    #finding phi @corner 2
    phi_corner2_H = 0
    for m in range(M):
        temp_phiH = phi_h_m_i2_func(m, a2, -d2)
        phi_corner2_H += temp_phiH

    phi_corner2_P = phi_p_i2(a2, -d2)
    phi_corner2 = phi_corner2_H + phi_corner2_P

    #slant velocity z component approximation
    vel_z = (phi_corner2 - phi_corner1)*(d1-d2)/((d1-d2)**2+(a2-a1)**2)

    #calculating hydro coeffs
    hydro_terms = np.zeros((N+2*M), dtype=complex)

    for i in range(N):
        hydro_terms[i] = int_R_1n_1(i)*C_1n_1s[i]*z_n_d1_d2(i, d1)

    for i in range(M):
        hydro_terms[N+i] = vel_z*int_R_1n_2(i)*C_1n_2s[i]*z_n_d1_d2(i, d2)
        hydro_terms[N+M+i] = vel_z*int_R_2n_2(i)*C_2n_2s[i]*z_n_d1_d2(i, d2)

    hydro_coef =2*pi*(sum(hydro_terms) + int_phi_p_i1_no_coef() + vel_z*int_phi_p_i2_no_coef())
    hydro_coef_real = hydro_coef.real
    hydro_coef_imag = hydro_coef.imag/omega
    hydro_coef_nondim = h**3/(a2**3 * pi)*hydro_coef

    print("real", hydro_coef_real)
    print("imag", hydro_coef_imag)
    print(hydro_coef_nondim)

    # Define a function to plot a single potential field in a subplot
    def plot_potential_subplot(ax, field, R, Z, title):
        contour = ax.contourf(R, Z, field, levels=50, cmap='viridis')
        plt.colorbar(contour, ax=ax)
        ax.set_title(title)
        ax.set_xlabel('Radial Distance (R)')
        ax.set_ylabel('Axial Distance (Z)')

    def plot_potential_direct(field, R, Z, title):
        plt.figure(figsize=(8, 8))
        contour = plt.contourf(R, Z, field, levels=50, cmap='viridis')  # 绘制等高图
        plt.colorbar(contour)
        plt.title(title) 
        plt.xlabel('Radial Distance (R)') 
        plt.ylabel('Axial Distance (Z)') 
        plt.tight_layout() 
        plt.show() 

    # First set of subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 10)) 
    axes = axes.flatten()  # Flatten the 2D array of axes into a 1D array for easier iteration

    # Plot each potential field on a different subplot
    plot_potential_subplot(axes[0], np.real(phiH), R, Z, 'Homogeneous Potential')
    plot_potential_subplot(axes[1], np.imag(phiH), R, Z, 'Homogeneous Potential Imaginary')
    plot_potential_subplot(axes[2], np.real(phi), R, Z, 'Total Potential')
    plot_potential_subplot(axes[3], np.imag(phi), R, Z, 'Total Potential Imaginary')

    # Adjust layout and show the first figure
    plt.tight_layout()
    plt.show()
    plot_potential_direct(phiP, R, Z, 'Particular Potential')



    # # Use Capytaine to compute hydrodynamic coefficients
    # mesh_density = 5
    # wavenumber = 2.0
    # water_depth = h_basic
    # added_mass, radiation_damping = extract_hydro_coeffs(a_basic, a_basic, mesh_density, wavenumber, water_depth)

    # print("Keys in added_mass:", added_mass.keys())
    # print("Keys in radiation_damping:", radiation_damping.keys())

    # # Normalize MEEM coefficients
    # added_mass_capytaine = added_mass['Heave']
    # radiation_damping_capytaine = radiation_damping['Heave']

    # # Normalize MEEM coefficients
    # added_mass_meem = hydro_coef_real
    # radiation_damping_meem = hydro_coef_imag

    # # Comparison
    # added_mass_diff = np.abs(added_mass_meem - added_mass_capytaine) / np.abs(added_mass_capytaine)
    # radiation_damping_diff = np.abs(radiation_damping_meem - radiation_damping_capytaine) / np.abs(radiation_damping_capytaine)

    # print("Relative difference in Added Mass:", added_mass_diff)
    # print("Relative difference in Radiation Damping:", radiation_damping_diff)

if __name__ == "__main__":
    test_main_basic()
'''
#############################################################################
#multi-region   
from multi_constants import *
from multi_equations import *

np.set_printoptions(threshold=np.inf, linewidth=np.inf, precision=8, suppress=True)

def test_main():
    NMK = [30, 30, 30]  # Adjust these values as needed
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
    X = engine.solve_linear_system_multi(problem)

    hydro_coefficients = engine.compute_hydrodynamic_coefficients(problem, X)
    print(hydro_coefficients)

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

if __name__ == "__main__":
    test_main()
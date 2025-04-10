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

#############################################################################
#multi-region   
from multi_constants import *
from multi_equations import *
from constants import *

scale = np.mean(a)

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
    r_coordinates = {'a1': a[0], 'a2': a[1], 'a3': a[2]}
    z_coordinates = {'h': h}
    geometry = Geometry(r_coordinates, z_coordinates, domain_params)

    # Create MEEMProblem object
    problem = MEEMProblem(geometry)

    # Create MEEMEngine object
    engine = MEEMEngine([problem])

    # Assemble A matrix and b vector using multi-region methods
    A = engine.assemble_A_multi(problem, m0)
    b = engine.assemble_b_multi(problem, m0)

    # Solve the linear system A x = b
    X = engine.solve_linear_system_multi(problem, m0)

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
        return (Cs[0][n] * R_1n(n, r, 0, scale, h, d)) * Z_n_i(n, z, 0, h, d)

    def phi_h_m_i_func(i, m, r, z):
        return (Cs[i][m] * R_1n(m, r, i, scale, h, d) + Cs[i][NMK[i] + m] * R_2n(m, r, i, a, scale, h, d)) * Z_n_i(m, z, i, h, d)

    def phi_e_k_func(k, r, z):
        return Cs[-1][k] * Lambda_k(k, r, m0, scale, h) * Z_n_e(k, z, m0, h)

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

    phiP[regions[0]] = heaving[0] * phi_p_i_vec(d[0], R[regions[0]], Z[regions[0]], h)
    for i in range(1, boundary_count):
        phiP[regions[i]] = heaving[i] * phi_p_i_vec(d[i], R[regions[i]], Z[regions[i]], h)
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
        return (Cs[0][n] * diff_R_1n(n, r, 0, scale)) * Z_n_i(n, z, 0, h, d)

    def v_r_m_i_func(i, m, r, z):
        return (Cs[i][m] * diff_R_1n(m, r, i, scale) + Cs[i][NMK[i] + m] * diff_R_2n(m, r, i)) * Z_n_i(m, z, i, h, d)

    def v_r_e_k_func(k, r, z):
        return Cs[-1][k] * diff_Lambda_k(k, r, m0, scale) * Z_n_e(k, z, m0, h)

    def v_z_inner_func(n, r, z):
        return (Cs[0][n] * R_1n(n, r, 0, scale, h, d)) * diff_Z_n_i(n, z, 0, h)

    def v_z_m_i_func(i, m, r, z):
        return (Cs[i][m] * R_1n(m, r, i, scale, h, d) + Cs[i][NMK[i] + m] * R_2n(m, r, i, a, scale, h, d)) * diff_Z_n_i(m, z, i, h)

    def v_z_e_k_func(k, r, z):
        return Cs[-1][k] * Lambda_k(k, r, m0, scale, h) * diff_Z_n_e(k, z, m0, h)

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
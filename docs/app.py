import streamlit as st
import numpy as np
import pandas as pd
from scipy import linalg
import matplotlib.pyplot as plt
import os
import sys

# Import your custom modules
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../package/src'))
sys.path.append(src_path)

from equations import *
from multi_equations import *
from meem_engine import MEEMEngine
from meem_problem import MEEMProblem
from geometry import Geometry



# Set global numpy print options
np.set_printoptions(threshold=np.inf, linewidth=np.inf, precision=8, suppress=True)

def plot_potential(field, R, Z, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    c = ax.contourf(R, Z, field, levels=50, cmap='viridis')
    plt.colorbar(c, ax=ax)
    plt.title(title)
    plt.xlabel("Radial Distance (R)")
    plt.ylabel("Axial Distance (Z)")
    st.pyplot(fig)

def main():
    st.title("MEEM Simulation")
    st.sidebar.header("Configuration Parameters")

    # Sidebar for user customization
    st.sidebar.header("Simulation Parameters")

    # User inputs for customization
    h = st.sidebar.slider("Height (h)", 0.5, 2.0, 1.001, step=0.001)
    d = st.sidebar.text_input("(d)", "0.5,0.25,0.25")
    a = st.sidebar.text_input("(a)", "0.5,1,1")
    heaving = st.sidebar.text_input("Heaving States (1=True, 0=False)", "1,1")
    # Sidebar input for slant customization
    slant_input = st.sidebar.text_input(
        "Slant Vectors (e.g., 0,0,1 for each region separated by semicolon)", 
        "0,0,1;0,0,1;0,0,1"
    )

    # Parse inputs
    d = list(map(float, d.split(',')))
    a = list(map(float, a.split(',')))
    a = [val for val in a if val is not None]
    heaving = list(map(int, heaving.split(',')))
    slants = [
        list(map(float, slant.split(','))) 
        for slant in slant_input.split(';')
    ]

    # Sidebar inputs
    NMK = st.sidebar.text_input("Number of Harmonics (NMK)", "30,30,30")
    NMK = list(map(int, NMK.split(',')))  # Convert input to a list of integers
    m0 = st.sidebar.number_input("Input value for m0", value=1)


    # Ensure slants align with the number of regions
    if len(slants) != len(NMK):
        st.error("Please provide slant vectors for each region, separated by semicolons.")
        return
    
    spatial_res = st.sidebar.slider("Spatial Resolution", min_value=10, max_value=100, value=50, step=5)
    show_total = st.sidebar.checkbox("Show Total Potential Plots", value=True)
    show_homogeneous = st.sidebar.checkbox("Show Homogeneous Potential Plots", value=True)
    show_particular = st.sidebar.checkbox("Show Particular Potential Plots", value=True)
    show_radial = st.sidebar.checkbox("Show Radial Velocity Potential Plots", value=True)
    show_vertical = st.sidebar.checkbox("Show Vertical Velocity Potential Plots", value=True)

    # Geometry and engine setup
    boundary_count = len(NMK) - 1
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
            # Use user input or default
            # Set True if the region is slanted
            'slant': slants[idx] if idx < len(slants) else [0, 0, 1]  
        }
        domain_params.append(params)
    
    # Show domain parameters
    with st.expander("View Domain Parameters"):
        st.write("Domain Parameters:")
        st.json(domain_params)

    # Create Geometry object
    r_coordinates = {'a': a}
    z_coordinates = {'h': h}
    geometry = Geometry(r_coordinates, z_coordinates, domain_params)

    # Create MEEMProblem object
    problem = MEEMProblem(geometry)

    # Create MEEMEngine object
    engine = MEEMEngine([problem])

    # Solve linear system
    A = engine.assemble_A_multi(problem, m0)
    b = engine.assemble_b_multi(problem, m0)

    # Solve the linear system A x = b
    X = linalg.solve(A, b)
    hydro_coefficients = engine.compute_hydrodynamic_coefficients(problem, X)

    st.write("Hydrodynamic Coefficients:")
    st.json(hydro_coefficients)

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
        print(f"Value of d: {d}, type of d: {type(d)}")
        print("i: ", i)
        return (Cs[0][n] * R_1n(n, r, 0, h, d, a)) * Z_n_i(n, z, 0, h, d)

    def phi_h_m_i_func(i, m, r, z):
        return (Cs[i][m] * R_1n(m, r, i, h, d, a) + Cs[i][NMK[i] + m] * R_2n(m, r, i, a, h, d)) * Z_n_i(m, z, i, h, d)

    def phi_e_k_func(k, r, z, m0):
        return Cs[-1][k] * Lambda_k(k, r, m0, a, NMK, h) * Z_n_e(k, z, m0, h)

    # Visualization grid
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
        temp_phiH = phi_e_k_func(k, R[regions[-1]], Z[regions[-1]], m0)
        phiH[regions[-1]] = temp_phiH if k == 0 else phiH[regions[-1]] + temp_phiH

    phi_p_i_vec = np.vectorize(phi_p_i)

    phiP[regions[0]] = heaving[0] * phi_p_i_vec(d[0], R[regions[0]], Z[regions[0]], h)
    for i in range(1, boundary_count):
        phiP[regions[i]] = heaving[i] * phi_p_i_vec(d[i], R[regions[i]], Z[regions[i]], h)
    phiP[regions[-1]] = 0

    phi = phiH + phiP

    def v_r_inner_func(n, r, z):
        return (Cs[0][n] * diff_R_1n(n, r, 0, h, d, a)) * Z_n_i(n, z, 0, h, d)

    def v_r_m_i_func(i, m, r, z):
        return (Cs[i][m] * diff_R_1n(m, r, i, h, d, a) + Cs[i][NMK[i] + m] * diff_R_2n(m, r, i, h, d, a)) * Z_n_i(m, z, i, h, d)

    def v_r_e_k_func(k, r, z, m0):
        return Cs[-1][k] * diff_Lambda_k(k, r, m0, NMK, h, a) * Z_n_e(k, z, m0, h)

    def v_z_inner_func(n, r, z):
        return (Cs[0][n] * R_1n(n, r, 0, h, d, a)) * diff_Z_n_i(n, z, 0, h, d)

    def v_z_m_i_func(i, m, r, z):
        return (Cs[i][m] * R_1n(m, r, i, h, d, a) + Cs[i][NMK[i] + m] * R_2n(m, r, i, a, h, d)) * diff_Z_n_i(m, z, i, h, d)

    def v_z_e_k_func(k, r, z, m0):
        return Cs[-1][k] * Lambda_k(k, r, m0, a, NMK, h) * diff_Z_k_e(k, z, m0, h, NMK)

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
        temp_vrH = v_r_e_k_func(k, R[regions[-1]], Z[regions[-1]], m0)
        temp_vzH = v_z_e_k_func(k, R[regions[-1]], Z[regions[-1]], m0)
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

    # Plot potentials
    if show_total:
        st.subheader("Total Potential")
        plot_potential(np.real(phi), R, Z, "Real Part of Total Potential")
        plot_potential(np.imag(phi), R, Z, "Imaginary Part of Total Potential")

    if show_homogeneous:
        st.subheader("Homogeneous Potential")
        plot_potential(np.real(phi), R, Z, "Real Part of Homogeneous Potential")
        plot_potential(np.imag(phi), R, Z, "Imaginary Part of Homogeneous Potential")

    if show_particular:
        st.subheader("Particular Potential")
        plot_potential(np.real(phi), R, Z, "Real Part of Particular Potential")
        plot_potential(np.imag(phi), R, Z, "Imaginary Part of Particular Potential")

    if show_radial:
        st.subheader("Radial Velocity Potential")
        plot_potential(np.real(vr), R, Z, 'Radial Velocity - Real')
        plot_potential(np.imag(vr), R, Z, 'Radial Velocity - Imaginary')

    if show_vertical:
        st.subheader("Vertical Velocity Potential")
        plot_potential(np.real(vz), R, Z, 'Vertical Velocity - Real')
        plot_potential(np.imag(vz), R, Z, 'Vertical Velocity - Imaginary')

if __name__ == "__main__":
    main()
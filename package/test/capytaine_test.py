import numpy as np
from scipy.linalg import norm
import sys
import os
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.append(src_path)
from meem_engine import MEEMEngine
from meem_problem import MEEMProblem
from geometry import Geometry

import capytaine as cpt
import numpy as np
import matplotlib.pyplot as plt
from capytaine.bem.airy_waves import airy_waves_potential, airy_waves_velocity, froude_krylov_force

import time

from multi_constants import *
from multi_equations import *

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.append(src_path)

# Set printing options for clarity
np.set_printoptions(threshold=np.inf, linewidth=np.inf, precision=8, suppress=True)

# Initialize solver
solver = cpt.BEMSolver()

# Function to create a body from profile points
def body_from_profile(x, y, z, nphi):
    xyz = np.array([np.array([x / np.sqrt(2), y / np.sqrt(2), z]) for x, y, z in zip(x, y, z)])  # Scale correction
    body = cpt.FloatingBody(cpt.AxialSymmetricMesh.from_profile(xyz, nphi=nphi))
    return body

def make_body(d, a, mesh_density):

    zt = np.linspace(0,0,mesh_density)
    rt = np.linspace(0, a[-1], mesh_density)
    # yt = np.linspace(D_f/2, D_s/2, mesh_density)
    top_surface = body_from_profile(rt, rt, zt, mesh_density**2)

    zb = np.linspace(- d[0], - d[0], mesh_density)
    rb = np.linspace(0, a[0], mesh_density)
    # yt = np.linspace(D_f/2, D_s/2, mesh_density)
    bot_surface = body_from_profile(rb, rb, zb, mesh_density**2)

    zo = np.linspace(- d[-1], 0, mesh_density)
    ro = np.linspace(a[-1], a[-1], mesh_density)
    # yt = np.linspace(D_f/2, D_s/2, mesh_density)
    outer_surface = body_from_profile(ro, ro, zo, mesh_density**2)

    bod = top_surface + bot_surface + outer_surface

    for i in range(1, len(a)):
      # make sides
      zs = np.linspace(- d[i-1], - d[i], mesh_density)
      rs = np.linspace(a[i-1], a[i-1], mesh_density)
      side = body_from_profile(rs, rs, zs ,mesh_density**2)

      # make bottoms
      zb = np.linspace(- d[i], - d[i], mesh_density)
      rb = np.linspace(a[i-1], a[i], mesh_density)
      bot = body_from_profile(rb, rb, zb ,mesh_density**2)

      bod = bod + side + bot

    return bod

def print_radiation_result(result):
    with open("radiation_result_details.txt", "w") as file:
        for attr in dir(result):
            if not attr.startswith("__"):
                try:
                    value = getattr(result, attr)
                    file.write(f"{attr}: {value}\n")
                except Exception as e:
                    file.write(f"{attr}: <Error accessing attribute: {e}>\n")

# Function to solve radiation problem and extract A matrix and b vector
def extract_A_b(d, a, mesh_density, w, h):
    body = make_body(d, a, mesh_density)
    body.add_translation_dof(name='Heave')
    body = body.immersed_part()
    
    # Set keep_details=True to retain detailed information
    rad_problem = cpt.RadiationProblem(body=body, wavenumber=w, water_depth=h)
    result = solver.solve(rad_problem, keep_details=True)
    
    print_radiation_result(result)
    print(dir(result))

    # Access the details from the result object
    details = result.solver_details
    A_matrix = details["lhs_matrix"]  # A matrix
    b_vector = details["rhs_vector"]  # b vector
    
    return A_matrix, b_vector

# Function to generate MEEM results
def generate_meem_results(h, d, a, mesh_density):
    # Define domain parameters for MEEM
    NMK = [mesh_density] * len(a)
    domain_params = []
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
        }
        domain_params.append(params)

    # Create Geometry, Problem, and Engine
    r_coordinates = {'a': a}
    z_coordinates = {'h': h}
    geometry = Geometry(r_coordinates, z_coordinates, domain_params)
    problem = MEEMProblem(geometry)
    engine = MEEMEngine([problem])

    # Generate A matrix and b vector
    A_meem = engine.assemble_A_multi(problem)
    b_meem = engine.assemble_b(problem)
    return A_meem, b_meem

# Function to compare two matrices/vectors
def compare_results(A1, b1, A2, b2):
    A_diff = norm(A1 - A2) / norm(A2)
    b_diff = norm(b1 - b2) / norm(b2)

    print(f"Relative difference in A matrix: {A_diff * 100:.2f}%")
    print(f"Relative difference in b vector: {b_diff * 100:.2f}%")

    if A_diff > 0.01:
        print("A matrix difference exceeds 1%.")
    else:
        print("A matrix difference is within 1%.")

    if b_diff > 0.01:
        print("b vector difference exceeds 1%.")
    else:
        print("b vector difference is within 1%.")

# Main script
def main():
    # Parameters for Capytaine
    h = 100
    d = [29, 7, 4]
    a = [3, 5, 10]
    heaving = [0, 1, 1]  # 0/false if not heaving, 1/true if yes heaving
    m0 = 1
    n = 3
    z = 6
    omega = 2
    mesh_density = 5
    w = 1

    # Generate Capytaine results
    A_capytaine, b_capytaine = extract_A_b(d, a, mesh_density, w, h)
    print("A_capytaine shape:", A_capytaine.shape)
    print("b_capytaine shape:", b_capytaine.shape)

    # Generate MEEM results
    A_meem, b_meem = generate_meem_results(h, d, a, mesh_density)
    print("A_meem shape:", A_meem.shape)
    print("b_meem shape:", b_meem.shape)

    # Compare results
    compare_results(A_capytaine, b_capytaine, A_meem, b_meem)

    # Save results to files
    np.save("A_capytaine.npy", A_capytaine)
    np.save("b_capytaine.npy", b_capytaine)
    np.save("A_meem.npy", A_meem)
    np.save("b_meem.npy", b_meem)

if __name__ == "__main__":
    main()
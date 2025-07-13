import numpy as np
from openflash.meem_engine import MEEMEngine
from openflash.meem_problem import MEEMProblem
from openflash.geometry import Geometry
from openflash.multi_equations import *
from openflash.multi_constants import *
import pytest

from math import pi, isinf


def test_hydro_coefficients_agree():
    r_coords_for_geometry = {'a1': 0.5, 'a2': 1.0}
    z_coordinates = {'h': 1.001}
    domain_params = [
        {'number_harmonics': 5, 'height': 1.0, 'radial_width': 0.5, 'category': 'inner', 'di': 0.5, 'a': 0.5, 'heaving': 1.0},
        {'number_harmonics': 8, 'height': 1.0, 'radial_width': 1.0, 'category': 'outer', 'di': 0.25, 'a': 1.0, 'heaving': 0.0}
    ]

    geometry = Geometry(r_coords_for_geometry, z_coordinates, domain_params)
    problem = MEEMProblem(geometry)

    # You probably need to set problem frequencies and modes here before using m0
    # This depends on your implementation. Example:
    boundary_count = len(domain_params) - 1
    problem_frequencies = np.array([omega(0, domain_params[0]['height'], g)])  # use m0=0 here as initial mode
    problem_modes = np.arange(boundary_count)
    problem.set_frequencies_modes(problem_frequencies, problem_modes)

    m0 = 1


    engine = MEEMEngine(problem_list=[problem])

    # Solve the system
    A = engine.assemble_A_multi(problem, m0)
    b = engine.assemble_b_multi(problem, m0)
    X = np.linalg.solve(A, b)

    # Extract problem data
    domain_list = problem.domain_list
    domain_keys = list(domain_list.keys())
    boundary_count = len(domain_keys) - 1

    NMK = [domain_list[idx].number_harmonics for idx in domain_keys]
    h = domain_list[0].h
    local_omega = omega(m0, h, g)  # Now you can calculate local_omega

    d = [domain_list[idx].di for idx in domain_keys]
    a = [domain_list[idx].a for idx in domain_keys if domain_list[idx].a is not None]
    heaving = [domain_list[idx].heaving for idx in domain_keys]

    size = NMK[0] + NMK[-1] + 2 * sum(NMK[1:-1])
    c = np.zeros((size - NMK[-1]), dtype=complex)

    col = 0
    for n in range(NMK[0]):
        c[n] = heaving[0] * int_R_1n(0, n, a, h, d) * z_n_d(n)
    col += NMK[0]
    for i in range(1, boundary_count):
        M = NMK[i]
        for m in range(M):
            c[col + m] = heaving[i] * int_R_1n(i, m, a, h, d) * z_n_d(m)
            c[col + M + m] = heaving[i] * int_R_2n(i, m, a, h, d) * z_n_d(m)
        col += 2 * M

    hydro_p_terms = np.zeros(boundary_count, dtype=complex)
    for i in range(boundary_count):
        hydro_p_terms[i] = heaving[i] * int_phi_p_i_no_coef(i, h, d, a)

    hydro_coef = 2 * pi * (np.dot(c, X[:-NMK[-1]]) + sum(hydro_p_terms))
    hydro_coef_real = hydro_coef.real * h**3 * rho
    hydro_coef_imag = 0 if isinf(m0) else hydro_coef.imag * local_omega * h**3 * rho

    # Compare to new implementation
    new_coeffs = engine.compute_hydrodynamic_coefficients(problem, X, m0)

    print("Old hydro_coef_real:", hydro_coef_real)
    print("New hydro_coeffs['real']:", new_coeffs['real'])
    print("Old hydro_coef_imag:", hydro_coef_imag)
    print("New hydro_coeffs['imag']:", new_coeffs['imag'])

    # Calculate relative and absolute errors for real part
    real_rel_error = abs(hydro_coef_real - new_coeffs["real"][0]) / abs(hydro_coef_real)
    real_abs_error = abs(hydro_coef_real - new_coeffs["real"][0])
    print(f"Real part relative error: {real_rel_error:.3e}, absolute error: {real_abs_error:.3e}")

    # Calculate relative and absolute errors for imaginary part
    imag_rel_error = abs(hydro_coef_imag - new_coeffs["imag"][0]) / abs(hydro_coef_imag)
    imag_abs_error = abs(hydro_coef_imag - new_coeffs["imag"][0])
    print(f"Imag part relative error: {imag_rel_error:.3e}, absolute error: {imag_abs_error:.3e}")

    # Use relaxed tolerances
    assert np.allclose(hydro_coef_real, new_coeffs["real"][0], rtol=1e-4, atol=1e-2), \
        f"Real parts differ: {hydro_coef_real} vs {new_coeffs['real'][0]}"
    assert np.allclose(hydro_coef_imag, new_coeffs["imag"][0], rtol=1e-4, atol=1e-2), \
        f"Imag parts differ: {hydro_coef_imag} vs {new_coeffs['imag'][0]}"

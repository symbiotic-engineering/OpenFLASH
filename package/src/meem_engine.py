#meem_engine.py
from typing import List, Dict
import numpy as np
from scipy.integrate import quad
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from equations import *
from meem_problem import MEEMProblem
from coupling import A_nm, A_mk
from multi_equations import *
import geometry
from results import Results
import xarray as xr


class MEEMEngine:
    """
    Manages multiple MEEMProblem instances and performs actions such as solving systems of equations,
    assembling matrices, and visualizing results.
    """

    def __init__(self, problem_list: List[MEEMProblem]):
        """
        Initialize the MEEMEngine object.

        
        :param problem_list: List of MEEMProblem instances.
        """
        self.problem_list = problem_list

    def assemble_A(self, problem: MEEMProblem, m0) -> np.ndarray:
        """
        Assemble the system matrix A for a given problem.

        :param problem: MEEMProblem instance.
        :return: Assembled matrix A.
        """
        # Extract domains
        inner_domain = problem.domain_list[0]
        outer_domain = problem.domain_list[1]
        exterior_domain = problem.domain_list[2]

        N = inner_domain.number_harmonics
        M = outer_domain.number_harmonics
        K = exterior_domain.number_harmonics

        size = N + 2 * M + K
        A = np.zeros((size, size), dtype=complex)

        h, d1, d2 = inner_domain.h, inner_domain.di, outer_domain.di
        a1, a2 = inner_domain.a, outer_domain.a

        # First row of block matrices (using d1)
        for i in range(N):
            A[i][i] = (h - d1) * R_1n_1(i, a1, a2, h, d1)
        for n in range(N):
            for m in range(M):
                A[n][N+m] = -R_1n_2(m, a1, a2, h, d2) * A_nm(n, m)
                A[n][N+M+m] = -R_2n_2(m, a1, a2, h, d2) * A_nm(n, m)

        # Second row of block matrices (using d2)
        for i in range(M):
            A[N + i, N + i] = (h - d2) * R_1n_2(i, a2, a2, h, d2)
            A[N + i, N + M + i] = (h - d2) * R_2n_2(i, a2, a2, h, d2)
        for m in range(M):
            for k in range(K):
                A[N + m, N + 2 * M + k] = -Lambda_k_r(k, a2, m0, a2, h) * A_mk(m, k)

        # Third row of block matrices (using d1)
        for m in range(M):
            for n in range(N):
                A[N + M + m, n] = -diff_R_1n_1(n, a1, d1, h, a2) * A_nm(n, m)
        for m in range(M):
            A[N + M + m, N + m] = (h - d2) * diff_R_1n_2(m, a1, d2, h, a2)
            A[N + M + m, N + M + m] = (h - d2) * diff_R_2n_2(m, a1, d2, h, a2)

        # Fourth row of block matrices (using d2)
        for k in range(K):
            for m in range(M):
                A[N + 2 * M + k, N + m] = -diff_R_1n_2(m, a2, d2, h, a2) * A_mk(m, k)
                A[N + 2 * M + k, N + M + m] = -diff_R_2n_2(m, a2, d2, h, a2) * A_mk(m, k)
        for k in range(K):
            A[N + 2 * M + k, N + 2 * M + k] = h * diff_Lambda_k_a2(k, m0, a2, h)

        return A
    
    def assemble_A_multi(self, problem: MEEMProblem, m0) -> np.ndarray:
        """
        Assemble the system matrix A for a given problem.

        :param problem: MEEMProblem instance.
        :return: Assembled matrix A.
        """
        domain_list = problem.domain_list
        domain_keys = list(domain_list.keys())
        boundary_count = len(domain_keys) - 1
        

        # Collect number of harmonics for each domain
        NMK = [domain_list[idx].number_harmonics for idx in domain_keys]
        size = NMK[0] + NMK[-1] + 2 * sum(NMK[1:-1])

        A = np.zeros((size, size), dtype=complex)

        # Extract parameters
        h = domain_list[0].h
        d = [domain_list[idx].di for idx in domain_keys]
        a = [domain_list[idx].a for idx in domain_keys]
        a_cleaned = [val for val in a if val is not None]
        scale = np.mean(a_cleaned)

        ###########################################################################
        # Potential Matching

        col = 0
        row = 0
        for bd in range(boundary_count):
            N = NMK[bd]
            M = NMK[bd + 1]
            if bd == (boundary_count - 1):  # i-e boundary
                if bd == 0:  # one cylinder
                    for n in range(N):
                        A[row + n][col + n] = (h - d[bd]) * R_1n(n, a[bd], bd, scale, h, d)
                        for m in range(M):
                            A[row + n][col + N + m] = - I_mk(n, m, bd, d, m0, h) * Lambda_k(m, a[bd], m0, scale, h)
                    row += N
                else:
                    for n in range(N):
                        A[row + n][col + n] = (h - d[bd]) * R_1n(n, a[bd], bd, scale, h, d)
                        A[row + n][col + N + n] = (h - d[bd]) * R_2n(n, a[bd], bd, a, scale, h, d)
                        for m in range(M):
                            A[row + n][col + 2*N + m] = - I_mk(n, m, bd, d, m0, h) * Lambda_k(m, a[bd], m0, scale, h)
                    row += N
            elif bd == 0:
                left_diag = d[bd] > d[bd + 1]  # which of the two regions gets diagonal entries
                if left_diag:
                    for n in range(N):
                        A[row + n][col + n] = (h - d[bd]) * R_1n(n, a[bd], bd, scale, h, d)
                        for m in range(M):
                            A[row + n][col + N + m] = - I_nm(n, m, bd, d, h) * R_1n(m, a[bd], bd + 1, scale, h, d)
                            A[row + n][col + N + M + m] = - I_nm(n, m, bd, d, h) * R_2n(m, a[bd], bd + 1, a, scale, h, d)
                    row += N
                else:
                    for m in range(M):
                        for n in range(N):
                            A[row + m][col + n] = I_nm(n, m, bd, d, h) * R_1n(n, a[bd], bd, scale, h, d)
                        A[row + m][col + N + m] = - (h - d[bd + 1]) * R_1n(m, a[bd], bd + 1, scale, h, d)
                        A[row + m][col + N + M + m] = - (h - d[bd + 1]) * R_2n(m, a[bd], bd + 1, a, scale, h, d)
                    row += M
                col += N
            else:  # i-i boundary
                left_diag = d[bd] > d[bd + 1]  # which of the two regions gets diagonal entries
                if left_diag:
                    for n in range(N):
                        A[row + n][col + n] = (h - d[bd]) * R_1n(n, a[bd], bd, scale, h, d)
                        A[row + n][col + N + n] = (h - d[bd]) * R_2n(n, a[bd], bd, h, d)
                        for m in range(M):
                            A[row + n][col + 2*N + m] = - I_nm(n, m, bd, d, h) * R_1n(m, a[bd], bd + 1, scale, h, d)
                            A[row + n][col + 2*N + M + m] = - I_nm(n, m, bd, d, h) * R_2n(m, a[bd], bd + 1, a, scale, h, d)
                    row += N
                else:
                    for m in range(M):
                        for n in range(N):
                            A[row + m][col + n] = I_nm(n, m, bd, d, h) * R_1n(n, a[bd], bd, scale, h, d)
                            A[row + m][col + N + n] = I_nm(n, m, bd, d, h) * R_2n(n, a[bd], bd, a, scale, h, d)
                        A[row + m][col + 2*N + m] = - (h - d[bd + 1]) * R_1n(m, a[bd], bd + 1, scale, h, d)
                        A[row + m][col + 2*N + M + m] = - (h - d[bd + 1]) * R_2n(m, a[bd], bd + 1, a, scale, h, d)
                    row += M
                col += 2 * N

        ###########################################################################
        # Velocity Matching 

        col = 0
        for bd in range(boundary_count):
            N = NMK[bd]
            M = NMK[bd + 1]
            if bd == (boundary_count - 1):  # i-e boundary
                if bd == 0:  # one cylinder
                    for m in range(M):
                        for n in range(N):
                            A[row + m][col + n] = - I_mk(n, m, bd, d, m0, h) * diff_R_1n(n, a[bd], bd, scale, h, d)
                        A[row + m][col + N + m] = h * diff_Lambda_k(m, a[bd], m0, scale, h)
                    row += N
                else:
                    for m in range(M):
                        for n in range(N):
                            A[row + m][col + n] = - I_mk(n, m, bd, d, m0, h) * diff_R_1n(n, a[bd], bd, scale, h, d)
                            A[row + m][col + N + n] = - I_mk(n, m, bd, d, m0, h) * diff_R_2n(n, a[bd], bd, scale, h, d)
                        A[row + m][col + 2*N + m] = h * diff_Lambda_k(m, a[bd], m0, scale, h)
                    row += N
            elif bd == 0:
                left_diag = d[bd] < d[bd + 1]  # which of the two regions gets diagonal entries
                if left_diag:
                    for n in range(N):
                        A[row + n][col + n] = - (h - d[bd]) * diff_R_1n(n, a[bd], bd, scale, h, d)
                        for m in range(M):
                            A[row + n][col + N + m] = I_nm(n, m, bd, d, h) * diff_R_1n(m, a[bd], bd + 1, scale, h, d)
                            A[row + n][col + N + M + m] = I_nm(n, m, bd, d, h) * diff_R_2n(m, a[bd], bd + 1, scale, h, d)
                    row += N
                else:
                    for m in range(M):
                        for n in range(N):
                            A[row + m][col + n] = - I_nm(n, m, bd, d, h) * diff_R_1n(n, a[bd], bd, scale, h, d)
                        A[row + m][col + N + m] = (h - d[bd + 1]) * diff_R_1n(m, a[bd], bd + 1, scale, h, d)
                        A[row + m][col + N + M + m] = (h - d[bd + 1]) * diff_R_2n(m, a[bd], bd + 1, scale, h, d)
                    row += M
                col += N
            else:  # i-i boundary
                left_diag = d[bd] < d[bd + 1]  # which of the two regions gets diagonal entries
                if left_diag:
                    for n in range(N):
                        A[row + n][col + n] = - (h - d[bd]) * diff_R_1n(n, a[bd], bd, scale, h, d)
                        A[row + n][col + N + n] = - (h - d[bd]) * diff_R_2n(n, a[bd], bd, scale, h, d)
                        for m in range(M):
                            A[row + n][col + 2*N + m] = I_nm(n, m, bd, d, h) * diff_R_1n(m, a[bd], bd + 1, scale, h, d)
                            A[row + n][col + 2*N + M + m] = I_nm(n, m, bd, d, h) * diff_R_2n(m, a[bd], bd + 1, scale, h, d)
                    row += N
                else:
                    for m in range(M):
                        for n in range(N):
                            A[row + m][col + n] = - I_nm(n, m, bd, d, h) * diff_R_1n(n, a[bd], bd, scale, h, d)
                            A[row + m][col + N + n] = - I_nm(n, m, bd, d, h) * diff_R_2n(n, a[bd], bd, scale, h, d)
                        A[row + m][col + 2*N + m] = (h - d[bd + 1]) * diff_R_1n(m, a[bd], bd + 1, scale, h, d)
                        A[row + m][col + 2*N + M + m] = (h - d[bd + 1]) * diff_R_2n(m, a[bd], bd + 1, scale, h, d)
                    row += M
                col += 2 * N

        return A


    def assemble_b(self, problem: MEEMProblem, m0) -> np.ndarray:
        """
        Assemble the right-hand side vector b for a given problem.

        :param problem: MEEMProblem instance.
        :return: Assembled vector b.
        """
        inner_domain = problem.domain_list[0]
        outer_domain = problem.domain_list[1]
        exterior_domain = problem.domain_list[2]

        N = inner_domain.number_harmonics
        M = outer_domain.number_harmonics
        K = exterior_domain.number_harmonics

        size = N + 2 * M + K
        b = np.zeros(size, dtype=complex)

        h, d1, d2 = inner_domain.h, inner_domain.di, outer_domain.di
        a1, a2 = inner_domain.a, outer_domain.a
        
        # Extract the integral result from the quad output
        rhs_12 = np.array([integrate.quad(lambda z: phi_p_i1_i2_a1(z, h, a1, d1, d2) * Z_n_i1(n, z, h, d1), -h, -d1)[0] for n in range(N)])
        rhs_2E = np.array([-integrate.quad(lambda z: phi_p_a2(z, a2, h, d2) * Z_n_i2(m, z, h, d2), -h, -d2)[0] for m in range(M)])
        rhs_velocity_12 = np.array([integrate.quad(lambda z: diff_phi_i1(a1, d1, h) * Z_n_i2(m, z, h, d2), -h, -d1)[0] - integrate.quad(lambda z: diff_phi_i2(a1, d2, h) * Z_n_i2(m, z, h, d2), -h, -d2)[0] for m in range(M)])
        rhs_velocity_2E = np.array([integrate.quad(lambda z: diff_phi_i2(a2, d2, h) * Z_n_e(k, z, m0, h), -h, -d2)[0] for k in range(K)])


        b = np.concatenate((rhs_12, rhs_2E, rhs_velocity_12, rhs_velocity_2E))


        return b
    
    def assemble_b_multi(self, problem: MEEMProblem, m0) -> np.ndarray:
        """
        Assemble the right-hand side vector b for a given problem (multi-region).

        :param problem: MEEMProblem instance.
        :return: Assembled vector b.
        """
        domain_list = problem.domain_list
        domain_keys = list(domain_list.keys())
        boundary_count = len(domain_keys) - 1

        # Collect number of harmonics for each domain
        NMK = [domain_list[idx].number_harmonics for idx in domain_keys]
        size = NMK[0] + NMK[-1] + 2 * sum(NMK[1:-1])

        b = np.zeros(size, dtype=complex)

        # Extract parameters
        h = domain_list[0].h
        d = [domain_list[idx].di for idx in domain_keys]
        a = [domain_list[idx].a for idx in domain_keys]
        heaving = [domain_list[idx].heaving for idx in domain_keys]

        index = 0

        # Potential matching
        for boundary in range(boundary_count):
            if boundary == (boundary_count - 1):  # i-e boundary
                for n in range(NMK[boundary]):
                    b[index] = b_potential_end_entry(n, boundary, heaving, h, d, a)
                    index += 1
            else:  # i-i boundary
                i = boundary
                if d[i] > d[i + 1]:
                    N = NMK[i]
                else:
                    N = NMK[i+1]
                for n in range(N):
                    b[index] = b_potential_entry(n, boundary, d, heaving, h, a)
                    index += 1

        # Velocity matching
        for boundary in range(boundary_count):
            if boundary == (boundary_count - 1):  # i-e boundary
                for k in range(NMK[-1]):
                    assert index < size, f"Index {index} out of bounds for size {size}"  # Explicit check
                    b[index] = b_velocity_end_entry(k, boundary, heaving, a, h, d, m0)
                    index += 1
            else:  # i-i boundary
                if d[boundary] < d[boundary + 1]:
                    N = NMK[boundary + 1]
                else:
                    N = NMK[boundary]
                for n in range(N):
                    assert index < size, f"Index {index} out of bounds for size {size}"  # Explicit check
                    b[index] = b_velocity_entry(n, boundary, heaving, a, h, d)
                    index += 1

        return b
        
    def solve_linear_system(self, problem: MEEMProblem, m0) -> np.ndarray:
        """
        Solve the linear system A x = b for the given problem.

        :param problem: MEEMProblem instance.
        :return: Solution vector X.
        """
        from scipy import linalg

        A = self.assemble_A(problem, m0)
        b = self.assemble_b(problem, m0)
        X = linalg.solve(A, b)
        return X
    
    def solve_linear_system_multi(self, problem: MEEMProblem, m0) -> np.ndarray:
        """
        Solve the linear system A x = b for the given problem.

        :param problem: MEEMProblem instance.
        :return: Solution vector X.
        """
        from scipy import linalg

        A = self.assemble_A_multi(problem, m0)
        b = self.assemble_b_multi(problem, m0)
        X = linalg.solve(A, b)
        return X

    def compute_hydrodynamic_coefficients(self, problem: MEEMProblem, X: np.ndarray) -> Dict[str, any]:
        """
        Compute the hydrodynamic coefficients for a given problem and solution X.

        :param problem: MEEMProblem instance.
        :param X: Solution vector X from solving A x = b.
        :return: Dictionary containing hydrodynamic coefficients and related values.
        """
        from multi_equations import int_R_1n, int_R_2n, z_n_d, int_phi_p_i_no_coef
        from multi_constants import rho, omega
        from math import pi

        domain_list = problem.domain_list
        domain_keys = list(domain_list.keys())
        boundary_count = len(domain_keys) - 1

        # Collect number of harmonics for each domain
        NMK = [domain_list[idx].number_harmonics for idx in domain_keys]
        size = NMK[0] + NMK[-1] + 2 * sum(NMK[1:-1])

        # Extract parameters
        h = domain_list[0].h
        d = [domain_list[idx].di for idx in domain_keys]
        a = [domain_list[idx].a for idx in domain_keys]
        heaving = [domain_list[idx].heaving for idx in domain_keys]
        a_cleaned = [val for val in a if val is not None]
        scale = np.mean(a_cleaned)

        ###########################################################################
        ###SEA Calculation: c-Matrix### 
        # NOTICE: hydro coeff values are too high!!!!!!!!!!!
        c = np.zeros((2*len(NMK)-2, max(NMK)), dtype=complex)
        X_matrix = np.zeros((2*len(NMK)-2, max(NMK)), dtype=complex)
        heaving_matrix = np.zeros((2*len(NMK)-2, len(NMK)-1), dtype=complex)
        col = 0
        for n in range(NMK[0]):
            c[0, n] = int_R_1n(0, n, a, scale, h, d) * z_n_d(n)
            X_matrix[0, n] = X[n]
        col += NMK[0]
        for i in range(1, boundary_count):
            M = NMK[i]
            for m in range(M):
                c[i, m] = int_R_1n(i, m, a, scale, h, d) * z_n_d(m)
                c[i+boundary_count-1, m] = int_R_2n(i, m, a, scale, h, d) * z_n_d(m)
                X_matrix[i, m] = X[col + m]  # for first eigen-coeff in region M
            col += M
        for i in range(1, boundary_count):
            M = NMK[i]
            for m in range(M):
                X_matrix[i+boundary_count-1, m] = X[col + m]  # for second eigen-coeff in region M
            col += M
        for i in range(boundary_count):
            for j in range(boundary_count):
                heaving_matrix[i, j] = heaving[j] * (h-d[i]) / (h-d[j])
                if i != 0:
                    heaving_matrix[i+boundary_count-1, j] = heaving[j] * (h-d[i]) / (h-d[j])
        cX_identity = np.diag(np.sum(c * X_matrix, axis=1))
        hydro_h_terms = np.dot(cX_identity, heaving_matrix)
        hydro_p_terms = np.zeros((boundary_count, boundary_count), dtype=complex)
        for i in range(boundary_count):
            for j in range(boundary_count):
                #hydro_p_terms[i, j] = heaving[j] * (h-d[i]) / (h-d[j]) * int_phi_p_i_no_coef(i)
                if (h-d[j]) != 0:
                    hydro_p_terms[i, j] = heaving[j] * (h-d[i]) / (h-d[j]) * int_phi_p_i_no_coef(i, h, d, a)
                else:
                    hydro_p_terms[i,j] = 0 #handle divide by zero error.
        indices_h = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 1), (5, 2), (6, 3)]
        indices_p = [(0, 0), (1, 1), (2, 2), (3, 3)]
        #hydro_coeff_list = 2 * pi * (sum(hydro_h_terms[i, j] for i, j in indices_h) + sum(hydro_p_terms[i, j] for i, j in indices_p))
        # Ensure indices are within bounds
        valid_indices_h = [(i, j) for i, j in indices_h if i < hydro_h_terms.shape[0] and j < hydro_h_terms.shape[1]]
        valid_indices_p = [(i, j) for i, j in indices_p if i < hydro_p_terms.shape[0] and j < hydro_p_terms.shape[1]]

        # Compute hydro_coeff_list using valid indices
        hydro_coeff_list = 2 * pi * (
            sum(hydro_h_terms[i, j] for i, j in valid_indices_h) +
            sum(hydro_p_terms[i, j] for i, j in valid_indices_p)
        )
        ###########################################################################
        # Convert the complex number to a dictionary
        hydro_coeffs = {
            "real": hydro_coeff_list.real,
            "imag": hydro_coeff_list.imag
        }
        return hydro_coeffs


    def calculate_potentials(self, problem: MEEMProblem, solution_vector: np.ndarray) -> Dict[str, dict]:
        """
        Calculate the potentials for the domains in the problem.
        :param problem: MEEMProblem instance containing domain definitions.
        :param solution_vector: Solution vector obtained from solving Ax = b.
        :return: A dictionary with domain names as keys and their corresponding potentials and coordinates as values.
        """
        potentials = {}
        domain_list = problem.domain_list
        start_idx = 0
        geometry_instance = problem.geometry
        #for domain_name, domain in domain_list.items():
        for domain_index, domain in domain_list.items():
            domain_name = f"domain_{domain_index}"
            # Get the number of harmonics for this domain
            num_harmonics = domain.number_harmonics
            # Extract the corresponding part of the solution vector
            domain_potential = solution_vector[start_idx:start_idx + num_harmonics]
            # Package the potential with domain-specific coordinates
            potentials[domain_name] = {
                'potentials': domain_potential,
                #error here
                #AttributeError: 'Domain' object has no attribute 'r_coordinates'
                #'r': domain.r_coordinates,
                #'z': domain.z_coordinates
                'r': geometry_instance.r_coordinates,
                'z': geometry_instance.z_coordinates
            }
            # Update the starting index for the next domain
            start_idx += num_harmonics
        return potentials
    def visualize_potential(self, potentials: Dict[str, np.ndarray], domain_names: List[str] = None):
        """
        Visualize the potentials for the given domains.
        :param potentials: Dictionary containing domain names and their corresponding potentials.
        :param domain_names: List of domain names to visualize. If None, visualize all.
        """
        domain_names = domain_names or potentials.keys()
        plt.figure(figsize=(10, 6))
        for domain_name in domain_names:
            potential = np.abs(potentials[domain_name])  # Magnitude of the potential
            plt.plot(potential, label=f"{domain_name} Potential")
        plt.title("Potential Magnitudes Across Domains")
        plt.xlabel("Harmonic Index")
        plt.ylabel("Magnitude")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def run_and_store_results(self, problem_index: int, m0) -> Results:
        """
        Perform the full MEEM computation and store results in the Results class.
        :param problem_index: Index of the MEEMProblem instance to process.
        :return: Results object containing the computed data.
        """
        problem = self.problem_list[problem_index]

        # Assemble the system matrix A and right-hand side vector b
        A = self.assemble_A_multi(problem, m0)
        b = self.assemble_b_multi(problem, m0)

        # Solve the linear system
        X = np.linalg.solve(A, b)

        # Compute hydrodynamic coefficients
        hydro_coeffs = self.compute_hydrodynamic_coefficients(problem, X)

        # Create a Results object
        geometry = problem.geometry #MEEMProblem contains a Geometry instance
        results = Results(geometry, problem.frequencies, problem.modes)

        # Let's say you have some dummy eigenfunction data:
        dummy_radial_data = np.zeros((len(problem.frequencies), len(problem.modes), 2))  # Adjust shape as needed
        dummy_vertical_data = np.zeros((len(problem.frequencies), len(problem.modes), 1))  # Adjust shape as needed
        results.store_results(0, dummy_radial_data, dummy_vertical_data)

        # Store eigenfunction results

        #store the results
        potentials = self.calculate_potentials(problem, X)
        results.store_potentials(potentials)

        #store the hydrodynamic coefficients.

        results.dataset['hydrodynamic_coefficients_real'] = hydro_coeffs['real']
        results.dataset['hydrodynamic_coefficients_imag'] = hydro_coeffs['imag']


        
        return results
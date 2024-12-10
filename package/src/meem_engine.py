#meem_engine.py
from typing import List, Dict
import numpy as np
from scipy.integrate import quad
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from meem_problem import MEEMProblem
from coupling import A_nm, A_mk
import equations
import multi_equations
from results import Results


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

    def assemble_A(self, problem: MEEMProblem) -> np.ndarray:
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
            A[i][i] = (h - d1) * equations.R_1n_1(i, a1)
        for n in range(N):
            for m in range(M):
                A[n][N+m] = -equations.R_1n_2(m, a1) * A_nm(n, m)
                A[n][N+M+m] = -equations.R_2n_2(m, a1) * A_nm(n, m)

        # Second row of block matrices (using d2)
        for i in range(M):
            A[N + i, N + i] = (h - d2) * equations.R_1n_2(i, a2)
            A[N + i, N + M + i] = (h - d2) * equations.R_2n_2(i, a2)
        for m in range(M):
            for k in range(K):
                A[N + m, N + 2 * M + k] = -equations.Lambda_k_r(k, a2) * A_mk(m, k)

        # Third row of block matrices (using d1)
        for m in range(M):
            for n in range(N):
                A[N + M + m, n] = -equations.diff_R_1n_1(n, a1) * A_nm(n, m)
        for m in range(M):
            A[N + M + m, N + m] = (h - d2) * equations.diff_R_1n_2(m, a1)
            A[N + M + m, N + M + m] = (h - d2) * equations.diff_R_2n_2(m, a1)

        # Fourth row of block matrices (using d2)
        for k in range(K):
            for m in range(M):
                A[N + 2 * M + k, N + m] = -equations.diff_R_1n_2(m, a2) * A_mk(m, k)
                A[N + 2 * M + k, N + M + m] = -equations.diff_R_2n_2(m, a2) * A_mk(m, k)
        for k in range(K):
            A[N + 2 * M + k, N + 2 * M + k] = h * equations.diff_Lambda_k_a2(k)

        return A
    
    def assemble_A_multi(self, problem: MEEMProblem) -> np.ndarray:
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
                        A[row + n][col + n] = (h - d[bd]) * multi_equations.R_1n(n, a[bd], bd)
                        for m in range(M):
                            A[row + n][col + N + m] = - multi_equations.I_mk(n, m, bd) * multi_equations.Lambda_k(m, a[bd])
                    row += N
                else:
                    for n in range(N):
                        A[row + n][col + n] = (h - d[bd]) * multi_equations.R_1n(n, a[bd], bd)
                        A[row + n][col + N + n] = (h - d[bd]) * multi_equations.R_2n(n, a[bd], bd)
                        for m in range(M):
                            A[row + n][col + 2*N + m] = - multi_equations.I_mk(n, m, bd) * multi_equations.Lambda_k(m, a[bd])
                    row += N
            elif bd == 0:
                left_diag = d[bd] > d[bd + 1]  # which of the two regions gets diagonal entries
                if left_diag:
                    for n in range(N):
                        A[row + n][col + n] = (h - d[bd]) * multi_equations.R_1n(n, a[bd], bd)
                        for m in range(M):
                            A[row + n][col + N + m] = - multi_equations.I_nm(n, m, bd) * multi_equations.R_1n(m, a[bd], bd + 1)
                            A[row + n][col + N + M + m] = - multi_equations.I_nm(n, m, bd) * multi_equations.R_2n(m, a[bd], bd + 1)
                    row += N
                else:
                    for m in range(M):
                        for n in range(N):
                            A[row + m][col + n] = multi_equations.I_nm(n, m, bd) * multi_equations.R_1n(n, a[bd], bd)
                        A[row + m][col + N + m] = - (h - d[bd + 1]) * multi_equations.R_1n(m, a[bd], bd + 1)
                        A[row + m][col + N + M + m] = - (h - d[bd + 1]) * multi_equations.R_2n(m, a[bd], bd + 1)
                    row += M
                col += N
            else:  # i-i boundary
                left_diag = d[bd] > d[bd + 1]  # which of the two regions gets diagonal entries
                if left_diag:
                    for n in range(N):
                        A[row + n][col + n] = (h - d[bd]) * multi_equations.R_1n(n, a[bd], bd)
                        A[row + n][col + N + n] = (h - d[bd]) * multi_equations.R_2n(n, a[bd], bd)
                        for m in range(M):
                            A[row + n][col + 2*N + m] = - multi_equations.I_nm(n, m, bd) * multi_equations.R_1n(m, a[bd], bd + 1)
                            A[row + n][col + 2*N + M + m] = - multi_equations.I_nm(n, m, bd) * multi_equations.R_2n(m, a[bd], bd + 1)
                    row += N
                else:
                    for m in range(M):
                        for n in range(N):
                            A[row + m][col + n] = multi_equations.I_nm(n, m, bd) * multi_equations.R_1n(n, a[bd], bd)
                            A[row + m][col + N + n] = multi_equations.I_nm(n, m, bd) * multi_equations.R_2n(n, a[bd], bd)
                        A[row + m][col + 2*N + m] = - (h - d[bd + 1]) * multi_equations.R_1n(m, a[bd], bd + 1)
                        A[row + m][col + 2*N + M + m] = - (h - d[bd + 1]) * multi_equations.R_2n(m, a[bd], bd + 1)
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
                            A[row + m][col + n] = - multi_equations.I_mk(n, m, bd) * multi_equations.diff_R_1n(n, a[bd], bd)
                        A[row + m][col + N + m] = h * multi_equations.diff_Lambda_k(m, a[bd])
                    row += N
                else:
                    for m in range(M):
                        for n in range(N):
                            A[row + m][col + n] = - multi_equations.I_mk(n, m, bd) * multi_equations.diff_R_1n(n, a[bd], bd)
                            A[row + m][col + N + n] = - multi_equations.I_mk(n, m, bd) * multi_equations.diff_R_2n(n, a[bd], bd)
                        A[row + m][col + 2*N + m] = h * multi_equations.diff_Lambda_k(m, a[bd])
                    row += N
            elif bd == 0:
                left_diag = d[bd] < d[bd + 1]  # which of the two regions gets diagonal entries
                if left_diag:
                    for n in range(N):
                        A[row + n][col + n] = - (h - d[bd]) * multi_equations.diff_R_1n(n, a[bd], bd)
                        for m in range(M):
                            A[row + n][col + N + m] = multi_equations.I_nm(n, m, bd) * multi_equations.diff_R_1n(m, a[bd], bd + 1)
                            A[row + n][col + N + M + m] = multi_equations.I_nm(n, m, bd) * multi_equations.diff_R_2n(m, a[bd], bd + 1)
                    row += N
                else:
                    for m in range(M):
                        for n in range(N):
                            A[row + m][col + n] = - multi_equations.I_nm(n, m, bd) * multi_equations.diff_R_1n(n, a[bd], bd)
                        A[row + m][col + N + m] = (h - d[bd + 1]) * multi_equations.diff_R_1n(m, a[bd], bd + 1)
                        A[row + m][col + N + M + m] = (h - d[bd + 1]) * multi_equations.diff_R_2n(m, a[bd], bd + 1)
                    row += M
                col += N
            else:  # i-i boundary
                left_diag = d[bd] < d[bd + 1]  # which of the two regions gets diagonal entries
                if left_diag:
                    for n in range(N):
                        A[row + n][col + n] = - (h - d[bd]) * multi_equations.diff_R_1n(n, a[bd], bd)
                        A[row + n][col + N + n] = - (h - d[bd]) * multi_equations.diff_R_2n(n, a[bd], bd)
                        for m in range(M):
                            A[row + n][col + 2*N + m] = multi_equations.I_nm(n, m, bd) * multi_equations.diff_R_1n(m, a[bd], bd + 1)
                            A[row + n][col + 2*N + M + m] = multi_equations.I_nm(n, m, bd) * multi_equations.diff_R_2n(m, a[bd], bd + 1)
                    row += N
                else:
                    for m in range(M):
                        for n in range(N):
                            A[row + m][col + n] = - multi_equations.I_nm(n, m, bd) * multi_equations.diff_R_1n(n, a[bd], bd)
                            A[row + m][col + N + n] = - multi_equations.I_nm(n, m, bd) * multi_equations.diff_R_2n(n, a[bd], bd)
                        A[row + m][col + 2*N + m] = (h - d[bd + 1]) * multi_equations.diff_R_1n(m, a[bd], bd + 1)
                        A[row + m][col + 2*N + M + m] = (h - d[bd + 1]) * multi_equations.diff_R_2n(m, a[bd], bd + 1)
                    row += M
                col += 2 * N

        return A


    def assemble_b(self, problem: MEEMProblem) -> np.ndarray:
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
        
        rhs_12 = np.array([(integrate.romberg(lambda z: equations.phi_p_i1_i2_a1(z) * equations.Z_n_i1(n, z),-h, -d1)) for n in range(N)])
        rhs_2E =np.array ([-integrate.romberg(lambda z: equations.phi_p_a2(z) * equations.Z_n_i2(m, z), -h, -d2) for m in range(M)]) #at a2 phi_p_i2
        rhs_velocity_12 = np.array([(integrate.romberg(lambda z: equations.diff_phi_i1(a1) * equations.Z_n_i2(m, z), -h, -d1)) - (integrate.romberg(lambda z: equations.diff_phi_i2(a1) * equations.Z_n_i2(m, z), -h, -d2)) for m in range(M)])
        rhs_velocity_2E = np.array([integrate.romberg(lambda z: equations.diff_phi_i2(a2) * equations.Z_n_e(k, z), -h, -d2) for k in range(K)])


        b = np.concatenate((rhs_12, rhs_2E, rhs_velocity_12, rhs_velocity_2E))


        return b
    
    def assemble_b_multi(self, problem: MEEMProblem) -> np.ndarray:
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
                for n in range(NMK[-2]):
                    b[index] = multi_equations.b_potential_end_entry(n, boundary)
                    index += 1
            else:  # i-i boundary
                # Iterate over eigenfunctions for smaller h - d
                i = boundary
                if d[i] < d[i + 1]:
                    N = NMK[i + 1]
                else:
                    N = NMK[i]
                for n in range(N):
                    b[index] = multi_equations.b_potential_entry(n, boundary)
                    index += 1

        # Velocity matching
        for boundary in range(boundary_count):
            if boundary == (boundary_count - 1):  # i-e boundary
                for k in range(NMK[-1]):
                    b[index] = multi_equations.b_velocity_end_entry(k, boundary)
                    index += 1
            else:  # i-i boundary
                # Iterate over eigenfunctions for larger h - d
                i = boundary
                if d[i] > d[i + 1]:
                    N = NMK[i]
                else:
                    N = NMK[i + 1]
                for n in range(N):
                    b[index] = multi_equations.b_velocity_entry(n, boundary)
                    index += 1

        return b

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

        # Initialize c vector
        c = np.zeros((size - NMK[-1]), dtype=complex)
        col = 0
        for n in range(NMK[0]):
            c[n] = heaving[0] * int_R_1n(0, n) * z_n_d(n)
        col += NMK[0]

        for i in range(1, boundary_count):
            M = NMK[i]
            for m in range(M):
                c[col + m] = heaving[i] * int_R_1n(i, m) * z_n_d(m)
                c[col + M + m] = heaving[i] * int_R_2n(i, m) * z_n_d(m)
            col += 2 * M
        #print(c)

        # Compute hydro_p_terms
        hydro_p_terms = np.zeros(boundary_count, dtype=complex)
        for i in range(boundary_count):
            hydro_p_terms[i] = heaving[i] * int_phi_p_i_no_coef(i)

        print(hydro_p_terms)

        # Compute hydrodynamic coefficient
        hydro_coef = 2 * pi * (np.dot(c, X[:-NMK[-1]]) + np.sum(hydro_p_terms))

        #print(X[:-NMK[-1]])
        #print(np.sum(hydro_p_terms))
        #print(hydro_coef)

        # Compute hydro_coef_real and hydro_coef_imag
        # Ensure rho and omega are defined in your multi_constants module
        hydro_coef_real = hydro_coef.real * h ** 3 * rho
        hydro_coef_imag = hydro_coef.imag * omega * h ** 3 * rho
        
        #print(hydro_coef_imag,hydro_coef.imag,omega,h,rho)

        # Find maximum heaving radius
        max_rad = a[0]
        for i in range(boundary_count - 1, -1, -1):
            if heaving[i]:
                max_rad = a[i]
                break

        hydro_coef_nondim = h ** 3 / (max_rad ** 3 * pi) * hydro_coef

        result = {
            'hydro_coef': hydro_coef,
            'hydro_coef_real': hydro_coef_real,
            'hydro_coef_imag': hydro_coef_imag,
            'hydro_coef_nondim': hydro_coef_nondim
        }
        return result
    
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
    for domain_name, domain in domain_list.items():
        # Get the number of harmonics for this domain
        num_harmonics = domain.number_harmonics

        # Extract the corresponding part of the solution vector
        domain_potential = solution_vector[start_idx:start_idx + num_harmonics]

        # Package the potential with domain-specific coordinates
        potentials[domain_name] = {
            'potentials': domain_potential,
            'r': domain.r_coordinates,
            'z': domain.z_coordinates
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
    
def run_and_store_results(self, problem_index: int) -> Results:
        """
        Perform the full MEEM computation and store results in the Results class.

        :param problem_index: Index of the MEEMProblem instance to process.
        :return: Results object containing the computed data.
        """
        problem = self.problem_list[problem_index]

        # Assemble the system matrix A and right-hand side vector b
        A = self.assemble_A(problem)
        b = self.assemble_b(problem)

        # Solve the linear system
        X = np.linalg.solve(A, b)

        # Compute hydrodynamic coefficients
        hydro_coeffs = self.compute_hydrodynamic_coefficients(problem, X)

        # Create a Results object
        geometry = problem.geometry #MEEMProblem contains a Geometry instance
        results = Results(geometry, self.frequencies, self.modes)

        # Store eigenfunction results
        for domain_index, domain in problem.domain_list.items():
            radial_data = X[:domain.number_harmonics].reshape(
                len(self.frequencies), len(self.modes), domain.number_harmonics
            )
            vertical_data = X[domain.number_harmonics:].reshape(
                len(self.frequencies), len(self.modes), domain.number_harmonics
            )
            results.store_results(domain_index, radial_data, vertical_data)

        # Add hydrodynamic coefficients to the results
        for key, value in hydro_coeffs.items():
            setattr(results, key, value)

        return results
from typing import List, Dict
import numpy as np
from scipy.integrate import quad
from scipy import linalg
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from meem_problem import MEEMProblem
from coupling import A_nm, A_mk
from equations import *
from constants import *
from constants import h 


from multi_equations import *
from multi_constants import *

class MEEMEngine:
    """
    Manages multiple MEEMProblem instances and performs actions such as solving systems of equations,
    assembling matrices, and visualizing results.
    """

    def __init__(self, problem_list: List[MEEMProblem], multi_region=False):
        """
        Initialize the MEEMEngine object.

        :param problem_list: List of MEEMProblem instances.
        :param multi_region: Boolean indicating if multi-region functionality is enabled.
        """
        self.problem_list = problem_list
        self.multi_region = multi_region

    def assemble_A(self, problem: MEEMProblem) -> np.ndarray:
        """
        Assemble the system matrix A for a given problem.

        :param problem: MEEMProblem instance.
        :return: Assembled matrix A.
        """
        if not self.multi_region:
            # Implement single-region assembly code here
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
            a1, a2 = inner_domain.a1, outer_domain.a2

            # First row of block matrices (using d1)
            for i in range(N):
                A[i][i] = (h - d1) * R_1n_1(i, a1)
            for n in range(N):
                for m in range(M):
                    A[n][N+m] = -R_1n_2(m, a1) * A_nm(n, m)
                    A[n][N+M+m] = -R_2n_2(m, a1) * A_nm(n, m)

            # Second row of block matrices (using d2)
            for i in range(M):
                A[N + i, N + i] = (h - d2) * R_1n_2(i, a2)
                A[N + i, N + M + i] = (h - d2) * R_2n_2(i, a2)
            for m in range(M):
                for k in range(K):
                    A[N + m, N + 2 * M + k] = -Lambda_k_r(k, a2) * A_mk(m, k)

            # Third row of block matrices (using d1)
            for m in range(M):
                for n in range(N):
                    A[N + M + m, n] = -diff_R_1n_1(n, a1) * A_nm(n, m)
            for m in range(M):
                A[N + M + m, N + m] = (h - d2) * diff_R_1n_2(m, a1)
                A[N + M + m, N + M + m] = (h - d2) * diff_R_2n_2(m, a1)

            # Fourth row of block matrices (using d2)
            for k in range(K):
                for m in range(M):
                    A[N + 2 * M + k, N + m] = -diff_R_1n_2(m, a2) * A_mk(m, k)
                    A[N + 2 * M + k, N + M + m] = -diff_R_2n_2(m, a2) * A_mk(m, k)
            for k in range(K):
                A[N + 2 * M + k, N + 2 * M + k] = h * diff_Lambda_k_a2(k)
            return A

        else:
            # Multi-region assembly using multi-region functions
            h = problem.geometry.z_coordinates.get('h')
            domain_list = problem.domain_list
            NMK = [domain.number_harmonics for domain in domain_list.values()]
            boundary_count = len(NMK) - 1
            size = NMK[0] + NMK[-1] + 2 * sum(NMK[1:-1])

            A = np.zeros((size, size), dtype=complex)
            col = 0
            row = 0
            for bd in range(boundary_count):
                N = NMK[bd]
                M = NMK[bd + 1]
                if bd == (boundary_count - 1):  # i-e boundary
                    if bd == 0:  # one cylinder
                        for n in range(N):
                            A[row + n][col + n] = (h - d[bd]) * R_1n(n, a[bd], bd)
                            for m in range(M):
                                A[row + n][col + N + m] = -I_mk(n, m, bd) * Lambda_k(m, a[bd])
                    else:
                        for n in range(N):
                            A[row + n][col + n] = (h - d[bd]) * R_1n(n, a[bd], bd)
                            A[row + n][col + N + n] = (h - d[bd]) * R_2n(n, a[bd], bd)
                            for m in range(M):
                                A[row + n][col + 2*N + m] = -I_mk(n, m, bd) * Lambda_k(m, a[bd])
                    row += N
                elif bd == 0:
                    left_diag = d[bd] > d[bd + 1]
                    if left_diag:
                        for n in range(N):
                            A[row + n][col + n] = (h - d[bd]) * R_1n(n, a[bd], bd)
                            for m in range(M):
                                A[row + n][col + N + m] = -I_nm(n, m, bd) * R_1n(m, a[bd], bd + 1)
                                A[row + n][col + N + M + m] = -I_nm(n, m, bd) * R_2n(m, a[bd], bd + 1)
                        row += N
                    else:
                        for m in range(M):
                            for n in range(N):
                                A[row + m][col + n] = I_nm(n, m, bd) * R_1n(n, a[bd], bd)
                            A[row + m][col + N + m] = -(h - d[bd + 1]) * R_1n(m, a[bd], bd + 1)
                            A[row + m][col + N + M + m] = -(h - d[bd + 1]) * R_2n(m, a[bd], bd + 1)
                        row += M
                    col += N
                else:  # i-i boundary
                    left_diag = d[bd] > d[bd + 1]
                    if left_diag:
                        for n in range(N):
                            A[row + n][col + n] = (h - d[bd]) * R_1n(n, a[bd], bd)
                            A[row + n][col + N + n] = (h - d[bd]) * R_2n(n, a[bd], bd)
                            for m in range(M):
                                A[row + n][col + 2*N + m] = -I_nm(n, m, bd) * R_1n(m, a[bd], bd + 1)
                                A[row + n][col + 2*N + M + m] = -I_nm(n, m, bd) * R_2n(m, a[bd], bd + 1)
                        row += N
                    else:
                        for m in range(M):
                            for n in range(N):
                                A[row + m][col + n] = I_nm(n, m, bd) * R_1n(n, a[bd], bd)
                                A[row + m][col + N + n] = I_nm(n, m, bd) * R_2n(n, a[bd], bd)
                            A[row + m][col + 2*N + m] = -(h - d[bd + 1]) * R_1n(m, a[bd], bd + 1)
                            A[row + m][col + 2*N + M + m] = -(h - d[bd + 1]) * R_2n(m, a[bd], bd + 1)
                        row += M
                    col += 2 * N

            # Do not reset col here; continue updating
            # Velocity Matching
            for bd in range(boundary_count):
                N = NMK[bd]
                M = NMK[bd + 1]
                if bd == (boundary_count - 1):  # i-e boundary
                    if bd == 0:  # one cylinder
                        for m in range(M):
                            for n in range(N):
                                A[row + m][n] = -I_mk(n, m, bd) * diff_R_1n(n, a[bd], bd)
                            A[row + m][N + m] = h * diff_Lambda_k(m, a[bd])
                        row += M
                    else:
                        for m in range(M):
                            for n in range(N):
                                A[row + m][col - N - M - M + n] = -I_mk(n, m, bd) * diff_R_1n(n, a[bd], bd)
                                A[row + m][col - N - M + n] = -I_mk(n, m, bd) * diff_R_2n(n, a[bd], bd)
                            A[row + m][col - M + m] = h * diff_Lambda_k(m, a[bd])
                        row += M
                elif bd == 0:
                    left_diag = d[bd] < d[bd + 1]
                    if left_diag:
                        for n in range(N):
                            A[row + n][col - N - 2*M + n] = -(h - d[bd]) * diff_R_1n(n, a[bd], bd)
                            for m in range(M):
                                A[row + n][col - 2*M + N + m] = I_nm(n, m, bd) * diff_R_1n(m, a[bd], bd + 1)
                                A[row + n][col - 2*M + N + M + m] = I_nm(n, m, bd) * diff_R_2n(m, a[bd], bd + 1)
                        row += N
                    else:
                        for m in range(M):
                            for n in range(N):
                                A[row + m][col - 2*N - 2*M + n] = -I_nm(n, m, bd) * diff_R_1n(n, a[bd], bd)
                            A[row + m][col - 2*M + N + m] = (h - d[bd + 1]) * diff_R_1n(m, a[bd], bd + 1)
                            A[row + m][col - 2*M + N + M + m] = (h - d[bd + 1]) * diff_R_2n(m, a[bd], bd + 1)
                        row += M
                else:  # i-i boundary
                    left_diag = d[bd] < d[bd + 1]
                    if left_diag:
                        for n in range(N):
                            A[row + n][col - 2*N - 2*M + n] = -(h - d[bd]) * diff_R_1n(n, a[bd], bd)
                            A[row + n][col - 2*M + N + n] = -(h - d[bd]) * diff_R_2n(n, a[bd], bd)
                            for m in range(M):
                                A[row + n][col + m] = I_nm(n, m, bd) * diff_R_1n(m, a[bd], bd + 1)
                                A[row + n][col + M + m] = I_nm(n, m, bd) * diff_R_2n(m, a[bd], bd + 1)
                        row += N
                    else:
                        for m in range(M):
                            for n in range(N):
                                A[row + m][col - 2*N - 2*M + n] = -I_nm(n, m, bd) * diff_R_1n(n, a[bd], bd)
                                A[row + m][col - 2*M + N + n] = -I_nm(n, m, bd) * diff_R_2n(n, a[bd], bd)
                            A[row + m][col + m] = (h - d[bd + 1]) * diff_R_1n(m, a[bd], bd + 1)
                            A[row + m][col + M + m] = (h - d[bd + 1]) * diff_R_2n(m, a[bd], bd + 1)
                        row += M

                # Update col based on the columns added in this iteration
                # For consistency, you might need to adjust col here as well

            return A

    def assemble_b(self, problem: MEEMProblem) -> np.ndarray:
        """
        Assemble the right-hand side vector b for a given problem.

        :param problem: MEEMProblem instance.
        :return: Assembled vector b.
        """
        if not self.multi_region:
            inner_domain = problem.domain_list[0]
            outer_domain = problem.domain_list[1]
            exterior_domain = problem.domain_list[2]

            N = inner_domain.number_harmonics
            M = outer_domain.number_harmonics
            K = exterior_domain.number_harmonics

            size = N + 2 * M + K
            b = np.zeros(size, dtype=complex)

            h, d1, d2 = inner_domain.h, inner_domain.di, outer_domain.di
            a1, a2 = inner_domain.a1, outer_domain.a2

            rhs_12 = np.array([(integrate.romberg(lambda z: phi_p_i1_i2_a1(z) * Z_n_i1(n, z),-h, -d1)) for n in range(N)])
            rhs_2E =np.array ([-integrate.romberg(lambda z: phi_p_a2(z) * Z_n_i2(m, z), -h, -d2) for m in range(M)]) #at a2 phi_p_i2
            rhs_velocity_12 = np.array([(integrate.romberg(lambda z: diff_phi_i1(a1) * Z_n_i2(m, z), -h, -d1)) - (integrate.romberg(lambda z: diff_phi_i2(a1) * Z_n_i2(m, z), -h, -d2)) for m in range(M)])
            rhs_velocity_2E = np.array([integrate.romberg(lambda z: diff_phi_i2(a2) * Z_n_e(k, z), -h, -d2) for k in range(K)])

            b = np.concatenate((rhs_12, rhs_2E, rhs_velocity_12, rhs_velocity_2E))

            return b
        else:
            # Multi-region assembly using multi-region functions
            h = problem.geometry.z_coordinates.get('h')
            domain_list = problem.domain_list
            NMK = [domain.number_harmonics for domain in domain_list.values()]
            boundary_count = len(NMK) - 1
            size = NMK[0] + NMK[-1] + 2 * sum(NMK[1:-1])

            b = np.zeros(size, dtype=complex)
            index = 0

            # Potential matching
            for boundary in range(boundary_count):
                if boundary == (boundary_count - 1):  # Interior-Exterior boundary
                    for n in range(NMK[-2]):
                        b[index] = b_potential_end_entry(n, boundary)
                        index += 1
                else:  # Interior-Interior boundary
                    if d[boundary] < d[boundary + 1]:
                        N = NMK[boundary + 1]
                    else:
                        N = NMK[boundary]
                    for n in range(N):
                        b[index] = b_potential_entry(n, boundary)
                        index += 1

            # Velocity matching
            for boundary in range(boundary_count):
                if boundary == (boundary_count - 1):  # Interior-Exterior boundary
                    for n in range(NMK[-1]):
                        b[index] = b_velocity_end_entry(n, boundary)
                        index += 1
                else:  # Interior-Interior boundary
                    if d[boundary] > d[boundary + 1]:
                        N = NMK[boundary + 1]
                    else:
                        N = NMK[boundary]
                    for n in range(N):
                        b[index] = b_velocity_entry(n, boundary)
                        index += 1

            return b

    def linear_solve_Axb(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Solve the linear system A x = b.

        :param A: Coefficient matrix A.
        :param b: Right-hand side vector b.
        :return: Solution vector x.
        """
        try:
            x = linalg.solve(A, b)
            return x
        except linalg.LinAlgError as e:
            print(f"Linear algebra error: {e}")
            return None

    def extract_coefficients(self, X: np.ndarray, NMK: List[int]) -> List[np.ndarray]:
        """
        Extract coefficients from the solution vector X.

        :param X: Solution vector X.
        :param NMK: List of number of harmonics for each region.
        :return: List of coefficient arrays for each region.
        """
        Cs = []
        row = 0
        Cs.append(X[:NMK[0]])
        row += NMK[0]
        for i in range(1, len(NMK) - 1):
            Cs.append(X[row: row + NMK[i] * 2])
            row += NMK[i] * 2
        Cs.append(X[row:])
        return Cs

    def compute_potentials(self, Cs: List[np.ndarray], R: np.ndarray, Z: np.ndarray, regions: List[np.ndarray]) -> np.ndarray:
        """
        Compute the total potential phi at given spatial points.

        :param Cs: List of coefficient arrays for each region.
        :param R: Radial coordinate grid.
        :param Z: Vertical coordinate grid.
        :param regions: List of boolean arrays defining the regions.
        :return: Total potential phi.
        """
        phi = np.full_like(R, np.nan + np.nan*1j, dtype=complex)
        phiH = np.full_like(R, np.nan + np.nan*1j, dtype=complex)
        phiP = np.full_like(R, np.nan + np.nan*1j, dtype=complex)

        # Compute the homogeneous potential in each region
        # Implement the computations as per the problem's requirements

        return phi



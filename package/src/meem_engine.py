from typing import List, Dict
import numpy as np
from scipy.integrate import quad
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from meem_problem import MEEMProblem
from coupling import A_nm, A_mk
import equations
import multi_equations


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

        print(h, d1, d2,a1, a2)

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

    def linear_solve_Axb(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Solve the linear system A x = b.

        :param A: Coefficient matrix A.
        :param b: Right-hand side vector b.
        :return: Solution vector x.
        """
        try:
            x = np.linalg.solve(A, b)
            return x
        except np.linalg.LinAlgError as e:
            print(f"Error solving linear system: {e}")
            return None

    def visualize_A(self, A: np.ndarray):
        """
        Visualize non-zero entries of matrix A.

        :param A: Coefficient matrix A.
        """
        rows, cols = np.nonzero(A)
        plt.figure(figsize=(6, 6))
        plt.scatter(cols, rows, color='blue', marker='o', s=100)
        plt.gca().invert_yaxis()
        plt.xticks(range(A.shape[1]))
        plt.yticks(range(A.shape[0]))

        # Add separation lines for different blocks
        N = self.problem_list[0].domain_list[0].number_harmonics
        M = self.problem_list[0].domain_list[1].number_harmonics
        K = self.problem_list[0].domain_list[2].number_harmonics
        block_cols = [N, N + M, N + 2 * M]
        for val in block_cols:
            plt.axvline(val - 0.5, color='black', linestyle='-', linewidth=1)
            plt.axhline(val - 0.5, color='black', linestyle='-', linewidth=1)

        plt.grid(True)
        plt.title('Non-Zero Entries of the Matrix A')
        plt.xlabel('Column Index')
        plt.ylabel('Row Index')
        plt.show()

    def perform_matching(self, matching_info: Dict[int, Dict[str, bool]]) -> bool:
        """
        Perform matching for all problems based on matching information.

        :param matching_info: Matching information between domains.
        :return: True if all matchings are successful, False otherwise.
        """
        success = True
        for problem in self.problem_list:
            result = problem.perform_matching(matching_info)
            if not result:
                success = False
        return success

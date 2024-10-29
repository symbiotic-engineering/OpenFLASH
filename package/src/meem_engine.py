from typing import List, Dict
import numpy as np
from scipy.integrate import quad
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from meem_problem import MEEMProblem
from coupling import A_nm, A_mk
from equations import (
    phi_p_i1, phi_p_i2, diff_phi_i1, diff_phi_i2, Z_n_i1, Z_n_e,
    m_k, Lambda_k_r, diff_Lambda_k_a2, R_1n_1, R_1n_2, R_2n_2,
    diff_R_1n_1, diff_R_1n_2, diff_R_2n_2, Z_n_i2,phi_p_i1_i2_a1,phi_p_a2
)


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
        a1, a2 = inner_domain.a1, outer_domain.a2

        rhs_12 = np.array([(integrate.romberg(lambda z: phi_p_i1_i2_a1(z) * Z_n_i1(n, z),-h, -d1)) for n in range(N)])
        rhs_2E =np.array ([-integrate.romberg(lambda z: phi_p_a2(z) * Z_n_i2(m, z), -h, -d2) for m in range(M)]) #at a2 phi_p_i2
        rhs_velocity_12 = np.array([(integrate.romberg(lambda z: diff_phi_i1(a1) * Z_n_i2(m, z), -h, -d1)) - (integrate.romberg(lambda z: diff_phi_i2(a1) * Z_n_i2(m, z), -h, -d2)) for m in range(M)])
        rhs_velocity_2E = np.array([integrate.romberg(lambda z: diff_phi_i2(a2) * Z_n_e(k, z), -h, -d2) for k in range(K)])

        b = np.concatenate((rhs_12, rhs_2E, rhs_velocity_12, rhs_velocity_2E))


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

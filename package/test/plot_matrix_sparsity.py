# package/test/plot_matrix_sparsity.py
import numpy as np
import matplotlib.pyplot as plt
from openflash.basic_region_geometry import BasicRegionGeometry
from openflash.meem_problem import MEEMProblem
from openflash.meem_engine import MEEMEngine
from openflash.multi_equations import (
    R_1n, R_2n, diff_R_1n, diff_R_2n, Lambda_k, diff_Lambda_k, 
    I_nm, I_mk, m_k, N_k_multi
)

# Constants
h = 1.001
a1 = 0.5
a2 = 1.0
d1 = 0.5
d2 = 0.25
m0 = 1.0
N, M, K = 4, 4, 4

def build_openflash_matrix():
    NMK = [N, M, K]
    geo = BasicRegionGeometry.from_vectors(
        a=np.array([a1, a2]),
        d=np.array([d1, d2]),
        h=h,
        NMK=NMK,
        slant_angle=np.zeros(2)
    )
    prob = MEEMProblem(geo)
    engine = MEEMEngine([prob])
    engine._ensure_m_k_and_N_k_arrays(prob, m0)
    return engine.assemble_A_multi(prob, m0)

def build_manual_matrix_from_snippet():
    # Re-implementing the manual logic provided in the prompt
    # Note: I must use openflash math functions as proxies for the script's custom ones
    d_list = [d1, d2]
    a_list = [a1, a2]
    NMK = [N, M, K]
    
    # Calculate dimension based on snippet logic
    # Snippet: A = zeros(N + 2*M + K, ...)
    dim = N + 2 * M + K
    A = np.zeros((dim, dim), dtype=complex)
    
    m_k_arr = m_k(NMK, m0, h)
    N_k_arr = np.array([N_k_multi(k, m0, h, m_k_arr) for k in range(K)])

    # 1. Row Group 1 (d1)
    for i in range(N):
        A[i, i] = (h - d1) * R_1n(i, a1, 0, h, d_list, a_list)
        
    for n in range(N):
        for m in range(M):
            # A[n][N+m] = -R_1n_2 * A_nm
            c = I_nm(n, m, 0, d_list, h)
            A[n, N + m] = -1 * R_1n(m, a1, 1, h, d_list, a_list) * c
            A[n, N + M + m] = -1 * R_2n(m, a1, 1, a_list, h, d_list) * c

    # 2. Row Group 2 (d2)
    for i in range(M):
        val = (h - d2) # R_1n/R_2n at a2 are 1
        A[N + i, N + i] = val
        A[N + i, N + M + i] = val
        
    for m in range(M):
        for k in range(K):
            c = I_mk(m, k, 1, d_list, m0, h, m_k_arr, N_k_arr)
            A[N + m, N + 2*M + k] = -1 * Lambda_k(k, a2, m0, a_list, m_k_arr) * c

    # 3. Row Group 3 (d1) - Velocity
    # Snippet: A[N+M+m][n] = -diff_R_1n_1 * A_nm(n, m)
    # Note indices: Row N+M+m (starts at N+M, length M)
    # Projects on Region 2 modes (M)
    for m in range(M):
        for n in range(N):
            c = I_nm(n, m, 0, d_list, h) # A_nm(n, m)
            A[N + M + m, n] = -1 * diff_R_1n(n, a1, 0, h, d_list, a_list) * c
            
    for m in range(M):
        A[N + M + m, N + m] = (h - d2) * diff_R_1n(m, a1, 1, h, d_list, a_list)
        A[N + M + m, N + M + m] = (h - d2) * diff_R_2n(m, a1, 1, h, d_list, a_list)

    # 4. Row Group 4 (d2) - Velocity
    # Snippet: A[N+2M+k][N+m] ...
    for k in range(K):
        for m in range(M):
            c = I_mk(m, k, 1, d_list, m0, h, m_k_arr, N_k_arr)
            A[N + 2*M + k, N + m] = -1 * diff_R_1n(m, a2, 1, h, d_list, a_list) * c
            A[N + 2*M + k, N + M + m] = -1 * diff_R_2n(m, a2, 1, h, d_list, a_list) * c
            
    for k in range(K):
        A[N + 2*M + k, N + 2*M + k] = h * diff_Lambda_k(k, a2, m0, a_list, m_k_arr)
        
    return A

def plot_sparsity_comparison():
    A_open = build_openflash_matrix()
    A_manual = build_manual_matrix_from_snippet()
    
    # Plot 1: OpenFLASH Non-Zero Entries
    rows, cols = np.nonzero(np.abs(A_open) > 1e-10)
    plt.figure(figsize=(6, 6))
    plt.scatter(cols, rows, color='blue', marker='o', s=100)
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.title('OpenFLASH: Non-Zero Entries')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    plt.savefig('openflash_sparsity.png')
    plt.close()
    
    # Plot 2: Mismatches
    # Threshold from snippet: 0.001
    is_close = np.isclose(A_open, A_manual, rtol=0.001, atol=1e-8)
    rows_diff, cols_diff = np.nonzero(~is_close)
    
    plt.figure(figsize=(6, 6))
    if len(rows_diff) > 0:
        plt.scatter(cols_diff, rows_diff, color='red', marker='x', s=100)
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.title('Mismatch: OpenFLASH vs Manual (Tol=0.001)')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    
    # Draw grid lines for blocks (optional, helpful for debugging)
    # N=4, M=4, K=4. Blocks at 4, 8, 12.
    for val in [4, 8, 12]:
        plt.axvline(val - 0.5, color='black', linestyle='-', linewidth=1)
        plt.axhline(val - 0.5, color='black', linestyle='-', linewidth=1)
        
    plt.savefig('openflash_mismatch.png')
    plt.close()
    print("Plots saved: openflash_sparsity.png, openflash_mismatch.png")

if __name__ == "__main__":
    plot_sparsity_comparison()
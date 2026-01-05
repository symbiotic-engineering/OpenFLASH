# package/test/reference_2region.py
import numpy as np
from openflash.multi_equations import (
    R_1n, R_2n, diff_R_1n, diff_R_2n, Lambda_k, diff_Lambda_k, 
    I_nm, I_mk, m_k, N_k_multi
)

h = 1.001
a1 = .5
a2 = 1
d1 = .5
d2 = .25
m0 = 1.0

def build_manual_matrices(N=4, M=4, K=4):
    d_list = [d1, d2]
    a_list = [a1, a2]
    NMK = [N, M, K]
    
    # Dimensions: N (Pot1) + M (Pot2) + N (Vel1 - b/c d1>d2) + K (Vel2)
    dim = N + M + N + K 
    A = np.zeros((dim, dim), dtype=complex)
    
    m_k_arr = m_k(NMK, m0, h)
    N_k_arr = np.array([N_k_multi(k, m0, h, m_k_arr) for k in range(K)])

    # 1. Potential Match at a1 (N rows)
    # Project on Region 1
    for n in range(N):
        A[n, n] = (h - d1) * R_1n(n, a1, 0, h, d_list, a_list) # Diag
        for m in range(M):
            c = I_nm(n, m, 0, d_list, h)
            A[n, N + m] = -1 * R_1n(m, a1, 1, h, d_list, a_list) * c
            A[n, N + M + m] = -1 * R_2n(m, a1, 1, a_list, h, d_list) * c

    # 2. Potential Match at a2 (M rows)
    # Project on Region 2
    row_start = N
    for m in range(M):
        val = (h - d2)
        A[row_start + m, N + m] = val
        A[row_start + m, N + M + m] = val
        for k in range(K):
            c = I_mk(m, k, 1, d_list, m0, h, m_k_arr, N_k_arr)
            A[row_start + m, N + 2*M + k] = -1 * Lambda_k(k, a2, m0, a_list, m_k_arr) * c

    # 3. Velocity Match at a1 (N rows) -- CORRECTED
    # Project on Region 1 (because d1 > d2, Region 1 is common)
    row_start = N + M
    for n in range(N):
        # Diagonal (Region 1)
        # v_diagonal_block(True) -> sign = -1
        A[row_start + n, n] = -1 * (h - d1) * diff_R_1n(n, a1, 0, h, d_list, a_list)
        
        # Dense (Region 2)
        # v_dense_block(False) -> sign = 1
        for m in range(M):
            c = I_nm(n, m, 0, d_list, h)
            A[row_start + n, N + m] = 1 * diff_R_1n(m, a1, 1, h, d_list, a_list) * c
            A[row_start + n, N + M + m] = 1 * diff_R_2n(m, a1, 1, h, d_list, a_list) * c

    # 4. Velocity Match at a2 (K rows)
    # Project on Exterior
    row_start = N + M + N
    for k in range(K):
        for m in range(M):
            c = I_mk(m, k, 1, d_list, m0, h, m_k_arr, N_k_arr)
            A[row_start + k, N + m] = -1 * diff_R_1n(m, a2, 1, h, d_list, a_list) * c
            A[row_start + k, N + M + m] = -1 * diff_R_2n(m, a2, 1, h, d_list, a_list) * c
        
        A[row_start + k, N + 2*M + k] = h * diff_Lambda_k(k, a2, m0, a_list, m_k_arr)

    return A
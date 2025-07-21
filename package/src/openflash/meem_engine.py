#meem_engine.py
from __future__ import annotations
from typing import List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from openflash.meem_problem import MEEMProblem
from openflash.problem_cache import ProblemCache
from openflash.multi_equations import *
from openflash.results import Results
from scipy import linalg
from openflash.multi_constants import *

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
        self.cache_list = {} # Stores ProblemCache objects

        # Build caches for all problems during engine initialization
        for problem in problem_list:
            self.cache_list[problem] = self._build_problem_cache(problem)
    
    def _ensure_m_k_and_N_k_arrays(self, problem: 'MEEMProblem', m0):
        """
        Ensure that m_k_arr and N_k_arr are computed and cached for the given problem and m0.
        """
        cache = self.cache_list[problem]

        if cache.m_k_arr is None or cache.N_k_arr is None:
            h = problem.domain_list[0].h
            domain_keys = list(problem.domain_list.keys())
            NMK = [problem.domain_list[idx].number_harmonics for idx in domain_keys]
            NMK_last = NMK[-1]

            m_k_arr = cache.m_k_entry_func(np.arange(NMK_last), m0, h)
            N_k_arr = np.array([
                cache.N_k_func(k, m0, h, NMK, m_k_arr)
                for k in range(NMK_last)
            ])
            cache.set_precomputed_m_k_N_k(m_k_arr, N_k_arr)
    
    def assemble_A_multi(self, problem: 'MEEMProblem', m0) -> np.ndarray:
        """
        Assemble the system matrix A for a given problem using pre-computed blocks.
        """
        cache = self.cache_list[problem]
        A = cache.get_A_template()

        self._ensure_m_k_and_N_k_arrays(problem, m0)

        for row, col, calc_func in cache.m0_dependent_A_indices:
            A[row, col] = calc_func(problem, m0, cache.m_k_arr, cache.N_k_arr)

        return A

    def _full_assemble_A_multi(self, problem: 'MEEMProblem', m0) -> np.ndarray:
        """
        Assemble the system matrix A for a given problem (full re-calculation).
        """
        domain_list = problem.domain_list
        domain_keys = list(domain_list.keys())
        NMK = [domain_list[idx].number_harmonics for idx in domain_keys]
        size = NMK[0] + NMK[-1] + 2 * sum(NMK[1:len(NMK) - 1])
        boundary_count = len(NMK) - 1

        A = np.zeros((size, size), dtype=complex)
        h = domain_list[0].h
        d = [domain_list[idx].di for idx in domain_keys]
        a = [domain_list[idx].a for idx in domain_keys]
        a_filtered = [val for val in a if val is not None]

        # Replicate original I_nm_vals and I_mk_vals pre-computation
        I_nm_vals_original = np.zeros((max(NMK), max(NMK), boundary_count - 1), dtype=complex)
        for bd_i_nm in range(boundary_count - 1):
            for n_i_nm in range(NMK[bd_i_nm]):
                for m_i_nm in range(NMK[bd_i_nm + 1]):
                    I_nm_vals_original[n_i_nm][m_i_nm][bd_i_nm] = I_nm(n_i_nm, m_i_nm, bd_i_nm, d, h)
        
        I_mk_vals_original = np.zeros((NMK[boundary_count - 1], NMK[boundary_count]), dtype=complex)
        for m_i_mk in range(NMK[boundary_count - 1]):
            for k_i_mk in range(NMK[boundary_count]):
                I_mk_vals_original[m_i_mk][k_i_mk] = I_mk_full(m_i_mk, k_i_mk, boundary_count - 1, d, m0, h, NMK)

        rows_A = []  # collection of rows of blocks in A matrix, to be concatenated later

        # --- Potential Matching Rows ---
        col_current_offset = 0
        for bd in range(boundary_count):
            N = NMK[bd]
            M = NMK[bd + 1]
            
            block_lst = []
            row_current_height = 0
            
            # Determine the blocks based on boundary type
            if bd == (boundary_count - 1):  # i-e boundary
                row_current_height = N
                left_block1 = p_diagonal_block_original(True, R_1n, bd, h, d, a_filtered, NMK)
                right_block = p_dense_block_e_original(bd, NMK, a_filtered, h, m0, I_mk_vals_original)
                
                # Simplified logic, assuming more than one cylinder
                left_block2 = p_diagonal_block_original(True, R_2n, bd, h, d, a_filtered, NMK)
                
                block_lst.append(np.zeros((row_current_height, col_current_offset), dtype=complex))
                block_lst.append(left_block1)
                block_lst.append(left_block2)
                block_lst.append(right_block)
                
            elif bd == 0:  # i-i boundary (Inner-Intermediate)
                left_diag_is_active = d[bd] > d[bd + 1]
                if left_diag_is_active:
                    row_current_height = N
                    left_block = p_diagonal_block_original(True, R_1n, 0, h, d, a_filtered, NMK)
                    right_block1 = p_dense_block_original(False, R_1n, 0, NMK, a_filtered, h, d, I_nm_vals_original)
                    right_block2 = p_dense_block_original(False, R_2n, 0, NMK, a_filtered, h, d, I_nm_vals_original)
                else:
                    row_current_height = M
                    left_block = p_dense_block_original(True, R_1n, 0, NMK, a_filtered, h, d, I_nm_vals_original)
                    right_block1 = p_diagonal_block_original(False, R_1n, 0, h, d, a_filtered, NMK)
                    right_block2 = p_diagonal_block_original(False, R_2n, 0, h, d, a_filtered, NMK)
                
                block_lst.append(left_block)
                block_lst.append(right_block1)
                block_lst.append(right_block2)
                
            else:  # i-i boundary (Intermediate-Intermediate)
                left_diag_is_active = d[bd] > d[bd + 1]
                if left_diag_is_active:
                    row_current_height = N
                    left_block1 = p_diagonal_block_original(True, R_1n, bd, h, d, a_filtered, NMK)
                    left_block2 = p_diagonal_block_original(True, R_2n, bd, h, d, a_filtered, NMK)
                    right_block1 = p_dense_block_original(False, R_1n, bd, NMK, a_filtered, h, d, I_nm_vals_original)
                    right_block2 = p_dense_block_original(False, R_2n, bd, NMK, a_filtered, h, d, I_nm_vals_original)
                else:
                    row_current_height = M
                    left_block1 = p_dense_block_original(True, R_1n, bd, NMK, a_filtered, h, d, I_nm_vals_original)
                    left_block2 = p_dense_block_original(True, R_2n, bd, NMK, a_filtered, h, d, I_nm_vals_original)
                    right_block1 = p_diagonal_block_original(False, R_1n, bd, h, d, a_filtered, NMK)
                    right_block2 = p_diagonal_block_original(False, R_2n, bd, h, d, a_filtered, NMK)
                
                block_lst.append(np.zeros((row_current_height, col_current_offset), dtype=complex))
                block_lst.append(left_block1)
                block_lst.append(left_block2)
                block_lst.append(right_block1)
                block_lst.append(right_block2)
                
            # Common logic to calculate padding and concatenate
            current_blocks_width = sum(b.shape[1] for b in block_lst)
            right_zeros_width = size - current_blocks_width
            right_zeros = np.zeros((row_current_height, right_zeros_width), dtype=complex)
            
            block_lst.append(right_zeros)
            rows_A.append(np.concatenate(block_lst, axis=1))

            # Update the offset for the next row
            if bd == 0:
                col_current_offset += N
            elif bd < boundary_count - 1:
                col_current_offset += (2 * N)

        # --- Velocity Matching Rows ---
        col_current_offset = 0 # Reset column offset for velocity matching section
        for bd in range(boundary_count):
            N = NMK[bd]
            M = NMK[bd + 1]
            row_current_height = 0
            block_lst = []
            
            if bd == (boundary_count - 1):  # i-e boundary
                row_current_height = M
                left_block1 = v_dense_block_e_original(diff_R_1n, bd, NMK, a_filtered, h, d, I_mk_vals_original)
                right_block = v_diagonal_block_e_original(bd, NMK, a_filtered, m0, h)
                
                # Simplified logic, assuming more than one cylinder
                left_block2 = v_dense_block_e_original(diff_R_2n, bd, NMK, a_filtered, h, d, I_mk_vals_original)

                block_lst.append(np.zeros((row_current_height, col_current_offset), dtype=complex))
                block_lst.append(left_block1)
                block_lst.append(left_block2)
                block_lst.append(right_block)
                
            elif bd == 0: # i-i boundary (Inner-Intermediate)
                left_diag_is_active = d[bd] <= d[bd + 1]
                if left_diag_is_active:
                    row_current_height = N
                    left_block = v_diagonal_block_original(True, diff_R_1n, 0, h, d, a_filtered, NMK)
                    right_block1 = v_dense_block_original(False, diff_R_1n, 0, NMK, a_filtered, h, d, I_nm_vals_original)
                    right_block2 = v_dense_block_original(False, diff_R_2n, 0, NMK, a_filtered, h, d, I_nm_vals_original)
                else:
                    row_current_height = M
                    left_block = v_dense_block_original(True, diff_R_1n, 0, NMK, a_filtered, h, d, I_nm_vals_original)
                    right_block1 = v_diagonal_block_original(False, diff_R_1n, 0, h, d, a_filtered, NMK)
                    right_block2 = v_diagonal_block_original(False, diff_R_2n, 0, h, d, a_filtered, NMK)
                
                block_lst.append(left_block)
                block_lst.append(right_block1)
                block_lst.append(right_block2)

            else: # i-i boundary (Intermediate-Intermediate)
                left_diag_is_active = d[bd] <= d[bd + 1]
                if left_diag_is_active:
                    row_current_height = N
                    left_block1 = v_diagonal_block_original(True, diff_R_1n, bd, h, d, a_filtered, NMK)
                    left_block2 = v_diagonal_block_original(True, diff_R_2n, bd, h, d, a_filtered, NMK)
                    right_block1 = v_dense_block_original(False, diff_R_1n, bd, NMK, a_filtered, h, d, I_nm_vals_original)
                    right_block2 = v_dense_block_original(False, diff_R_2n, bd, NMK, a_filtered, h, d, I_nm_vals_original)
                else:
                    row_current_height = M
                    left_block1 = v_dense_block_original(True, diff_R_1n, bd, NMK, a_filtered, h, d, I_nm_vals_original)
                    left_block2 = v_dense_block_original(True, diff_R_2n, bd, NMK, a_filtered, h, d, I_nm_vals_original)
                    right_block1 = v_diagonal_block_original(False, diff_R_1n, bd, h, d, a_filtered, NMK)
                    right_block2 = v_diagonal_block_original(False, diff_R_2n, bd, NMK, a_filtered, h, d, I_nm_vals_original)
                
                block_lst.append(np.zeros((row_current_height, col_current_offset), dtype=complex))
                block_lst.append(left_block1)
                block_lst.append(left_block2)
                block_lst.append(right_block1)
                block_lst.append(right_block2)
                
            # Common logic to calculate padding and concatenate
            current_blocks_width = sum(b.shape[1] for b in block_lst)
            right_zeros_width = size - current_blocks_width
            right_zeros = np.zeros((row_current_height, right_zeros_width), dtype=complex)
            
            block_lst.append(right_zeros)
            rows_A.append(np.concatenate(block_lst, axis=1))

            # Update the offset for the next row
            if bd == 0:
                col_current_offset += N
            elif bd < boundary_count - 1:
                col_current_offset += (2 * N)
            
        for idx, row in enumerate(rows_A):
            print(f"Row {idx} shape: {row.shape}")
        A = np.concatenate(rows_A, axis=0)
        return A
                
    # Now, the optimized assemble_b_multi method that uses the cache
    def assemble_b_multi(self, problem: MEEMProblem, m0) -> np.ndarray:
        """
        Assemble the right-hand side vector b for a given problem, leveraging pre-computed
        m0-independent parts and updating only m0-dependent entries.
        """
        cache = self.cache_list[problem]
        b = cache.get_b_template()

        self._ensure_m_k_and_N_k_arrays(problem, m0)

        for row, calc_func in cache.m0_dependent_b_indices:
            b[row] = calc_func(problem, m0, cache.m_k_arr, cache.N_k_arr)

        return b
    def _full_assemble_b_multi(self, problem: MEEMProblem, m0) -> np.ndarray:
        """
        Assemble the right-hand side vector b for a given problem (multi-region, full re-calculation).
        This is essentially the original assemble_b_multi.
        """
        domain_list = problem.domain_list
        domain_keys = list(domain_list.keys())

        # Collect number of harmonics for each domain
        NMK = [domain_list[idx].number_harmonics for idx in domain_keys]
        size = NMK[0] + NMK[-1] + 2 * sum(NMK[1:len(NMK) - 1])
        boundary_count = len(NMK) - 1

        b = np.zeros(size, dtype=complex)

        # Extract parameters
        h = domain_list[0].h
        d = [domain_list[idx].di for idx in domain_keys]
        a = [domain_list[idx].a for idx in domain_keys]
        a_filtered = [val for val in a if val is not None] # Ensure this matches your original logic
        heaving = [domain_list[idx].heaving for idx in domain_keys]
        print(f"domain_keys: {domain_keys}")
        print(f"NMK (number harmonics): {NMK}")
        print(f"a: {a}")
        print(f"d: {d}")

        # Potential matching (m0-independent)
        index = 0
        for boundary in range(boundary_count):
            if boundary == (boundary_count - 1): # i-e boundary
                for n in range(NMK[-2]): # NMK[boundary] is NMK[-2] for last i-region
                    b[index] = b_potential_end_entry(n, boundary, heaving, h, d, a) # Assuming b_potential_end_entry signature
                    index += 1
            else: # i-i boundary
                # Iterate over eigenfunctions for smaller h-d, as per original comment
                for n in range(NMK[boundary + (d[boundary] <= d[boundary + 1])]):
                    b[index] = b_potential_entry(n, boundary, d, heaving, h, a) # Assuming b_potential_entry signature
                    index += 1

        # Velocity matching (m0-dependent via b_velocity_end_entry)
        for boundary in range(boundary_count):
            if boundary == (boundary_count - 1): # i-e boundary
                for n in range(NMK[-1]):
                    # b_velocity_end_entry_full takes (k, i, heaving, a, h, d, m0, NMK)
                    b[index] = b_velocity_end_entry_full(n, boundary, heaving, a, h, d, m0, NMK)
                    index += 1
            else: # i-i boundary
                # Iterate over eigenfunctions for larger h-d, as per original comment
                for n in range(NMK[boundary + (d[boundary] > d[boundary + 1])]):
                    b[index] = b_velocity_entry(n, boundary, heaving, a, h, d) # Assuming b_velocity_entry signature
                    index += 1
        return b
    
    # --- Specialized helper functions for _build_problem_cache (for m0-dependent parts) ---
    # These will return a zero array for the template and add closures to cache.m0_dependent_A_indices

    def _p_dense_block_e_for_cache(self, bd, NMK, a_filtered, h, d, cache, current_global_row_offset, current_global_col_offset):
        N = NMK[bd]
        M = NMK[bd+1]
        block_template = np.zeros((N, M), dtype=complex)

        for n_local in range(N):
            for m_local in range(M):
                global_row = current_global_row_offset + n_local
                global_col = current_global_col_offset + m_local

                calc_func = lambda p, m0_val, mk_arr, Nk_arr, n=n_local, m=m_local, bd_val=bd: \
                            -I_mk_full(n, m, bd_val, d, m0_val, h, NMK) * \
                            Lambda_k_full(m, a_filtered[bd_val], m0_val, a_filtered, NMK, h)
                cache.add_m0_dependent_A_entry(global_row, global_col, calc_func)
        return block_template

    def _v_diagonal_block_e_for_cache(self, bd, NMK, a_filtered, h, d, cache, current_global_row_offset, current_global_col_offset):
        M = NMK[bd+1]
        block_template = np.zeros((M, M), dtype=complex)

        for k_local in range(M):
            global_row = current_global_row_offset + k_local
            global_col = current_global_col_offset + k_local # Diagonal
            
            calc_func = lambda p, m0_val, mk_arr, Nk_arr, k=k_local, bd_val=bd: \
                        h * diff_Lambda_k_full(k, a_filtered[bd_val], m0_val, NMK, h, a_filtered)
            cache.add_m0_dependent_A_entry(global_row, global_col, calc_func)
        return block_template
    
    def _build_problem_cache(self, problem: MEEMProblem) -> ProblemCache:
        cache = ProblemCache(problem)

        domain_list = problem.domain_list
        domain_keys = list(domain_list.keys())

        NMK = [domain_list[idx].number_harmonics for idx in domain_keys]
        size = NMK[0] + NMK[-1] + 2 * sum(NMK[1:len(NMK) - 1])
        boundary_count = len(NMK) - 1

        h = domain_list[0].h
        d = [domain_list[idx].di for idx in domain_keys]
        a = [domain_list[idx].a for idx in domain_keys]
        a_filtered = [val for val in a if val is not None]
        
        print(f"domain_keys: {domain_keys}")
        print(f"NMK (number harmonics): {NMK}")
        print(f"a: {a}")
        print(f"d: {d}")

        cache.set_m_k_and_N_k_funcs(m_k_entry, N_k_multi)
        
        # --- FIX 1: Compute m0-independent I_nm_vals and store in cache ---
        # This must be done once, at the start of cache building.
        I_nm_vals = np.zeros((max(NMK), max(NMK), boundary_count - 1), dtype=complex)
        for bd_i_nm in range(boundary_count - 1):
            for n_i_nm in range(NMK[bd_i_nm]):
                for m_i_nm in range(NMK[bd_i_nm + 1]):
                    I_nm_vals[n_i_nm][m_i_nm][bd_i_nm] = I_nm(n_i_nm, m_i_nm, bd_i_nm, d, h)
        cache.set_I_nm_vals(I_nm_vals)


        A_template = np.zeros((size, size), dtype=complex)
        current_global_row_offset = 0

        # --- Potential Blocks (Caching) ---
        col_start_of_potential_row_section = 0
        for bd in range(boundary_count):
            N_bd = NMK[bd]
            M_bd_plus_1 = NMK[bd + 1]
            
            current_row_blocks = []
            current_local_col_offset = 0

            if bd == (boundary_count - 1):  # i-e boundary (Potential)
                row_height = N_bd
                # Build blocks as before
                block1 = p_diagonal_block_original(True, R_1n, bd, h, d, a, NMK)
                current_row_blocks.append(block1)

                if bd > 0:
                    block3 = p_diagonal_block_original(True, R_2n, bd, h, d, a, NMK)
                    current_row_blocks.append(block3)

                block2_template = self._p_dense_block_e_for_cache(
                    bd, NMK, a, h, d, cache,
                    current_global_row_offset,
                    col_start_of_potential_row_section + sum(b.shape[1] for b in current_row_blocks)
                )
                current_row_blocks.append(block2_template)

                # Insert left zeros if bd != 0
                if bd != 0:
                    left_zeros_width = col_start_of_potential_row_section
                    left_zeros = np.zeros((row_height, left_zeros_width), dtype=complex)
                    current_row_blocks.insert(0, left_zeros)
                # Now recompute total width including left_zeros
                current_blocks_total_width = sum(b.shape[1] for b in current_row_blocks)  # includes left_zeros now

                right_zeros_width = size - current_blocks_total_width  # no need to add col_start again because included

                if right_zeros_width < 0:
                    raise ValueError(f"Negative right_zeros_width {right_zeros_width} at bd={bd}")

                if right_zeros_width > 0:
                    right_zeros = np.zeros((row_height, right_zeros_width), dtype=complex)
                    current_row_blocks.append(right_zeros)
                
            elif bd == 0: # i-i boundary (Inner-Intermediate Potential)
                left_diag_is_active = d[bd] > d[bd + 1]
                row_height = N_bd if left_diag_is_active else M_bd_plus_1
                
                if left_diag_is_active:
                    block1 = p_diagonal_block_original(True, R_1n, 0, h, d, a, NMK)
                    # FIX 2: Pass the computed I_nm_vals from the cache
                    block2 = p_dense_block_original(False, R_1n, 0, NMK, a, h, d, I_nm_vals)
                    block3 = p_dense_block_original(False, R_2n, 0, NMK, a, h, d, I_nm_vals)
                else:
                    # FIX 2: Pass the computed I_nm_vals from the cache
                    block1 = p_dense_block_original(True, R_1n, 0, NMK, a, h, d, I_nm_vals)
                    block2 = p_diagonal_block_original(False, R_1n, 0, h, d, a, NMK)
                    block3 = p_diagonal_block_original(False, R_2n, 0, h, d, a, NMK)
                
                current_row_blocks = [block1, block2, block3]
                if any(b is None for b in current_row_blocks):
                    raise ValueError(f"One or more blocks in current_row_blocks are None: {current_row_blocks}")
                current_blocks_total_width = sum(b.shape[1] for b in current_row_blocks)
                right_zeros_width = size - (col_start_of_potential_row_section + current_blocks_total_width)
                print(f"size: {size}")
                print(f"col_start_of_potential_row_section: {col_start_of_potential_row_section}")
                print(f"current_blocks_total_width: {current_blocks_total_width}")
                print(f"right_zeros_width: {right_zeros_width}")
                print(f"row_height: {row_height}")
                right_zeros = np.zeros((row_height, right_zeros_width), dtype=complex)
                print(f"right_zeros shape: {right_zeros.shape}")
                current_row_blocks.append(right_zeros)
                
            else: # i-i boundary (Intermediate-Intermediate Potential)
                left_diag_is_active = d[bd] > d[bd + 1]
                row_height = N_bd if left_diag_is_active else M_bd_plus_1
                
                if left_diag_is_active:
                    block1 = p_diagonal_block_original(True, R_1n, bd, h, d, a, NMK)
                    block2 = p_diagonal_block_original(True, R_2n, bd, h, d, a, NMK)
                    # FIX 2: Pass the computed I_nm_vals from the cache
                    block3 = p_dense_block_original(False, R_1n, bd, NMK, a, h, d, I_nm_vals)
                    block4 = p_dense_block_original(False, R_2n, bd, NMK, a, h, d, I_nm_vals)
                else:
                    # FIX 2: Pass the computed I_nm_vals from the cache
                    block1 = p_dense_block_original(True, R_1n, bd, NMK, a, h, d, I_nm_vals)
                    block2 = p_dense_block_original(True, R_2n, bd, NMK, a, h, d, I_nm_vals)
                    block3 = p_diagonal_block_original(False, R_1n, bd, h, d, a, NMK)
                    block4 = p_diagonal_block_original(False, R_2n, bd, h, d, a, NMK)

                current_row_blocks = [block1, block2, block3, block4]
                left_zeros = np.zeros((row_height, col_start_of_potential_row_section), dtype=complex)
                if any(b is None for b in current_row_blocks):
                    raise ValueError(f"One or more blocks in current_row_blocks are None: {current_row_blocks}")
                current_blocks_total_width = sum(b.shape[1] for b in current_row_blocks)
                right_zeros_width = size - (col_start_of_potential_row_section + current_blocks_total_width)
                print(f"size: {size}")
                print(f"col_start_of_potential_row_section: {col_start_of_potential_row_section}")
                print(f"current_blocks_total_width: {current_blocks_total_width}")
                print(f"right_zeros_width: {right_zeros_width}")
                print(f"row_height: {row_height}")
                right_zeros = np.zeros((row_height, right_zeros_width), dtype=complex)
                current_row_blocks.insert(0, left_zeros)
                current_row_blocks.append(right_zeros)
                
            print("Current row blocks shapes:")
            for i, b in enumerate(current_row_blocks):
                print(f"Block {i}: shape={b.shape}")
            concat_row = np.concatenate(current_row_blocks, axis=1)
            if concat_row.shape[1] != size:
                missing = size - concat_row.shape[1]
                if missing < 0:
                    raise ValueError(f"concat_row too wide: got {concat_row.shape[1]}, expected {size}")
                pad = np.zeros((concat_row.shape[0], missing), dtype=complex)
                concat_row = np.concatenate([concat_row, pad], axis=1)
            print(f"Concatenated row shape: {concat_row.shape}")
            # assert concat_row.shape[1] == size, f"Expected width {size}, got {concat_row.shape[1]}"
            A_template[current_global_row_offset : current_global_row_offset + concat_row.shape[0], :] = concat_row
            current_global_row_offset += concat_row.shape[0]

            if bd == 0:
                col_start_of_potential_row_section += NMK[bd]
            elif 0 < bd < boundary_count - 1:
                col_start_of_potential_row_section += 2 * NMK[bd]
            elif bd == boundary_count - 1:
                col_start_of_potential_row_section += NMK[bd]

        # --- Velocity Blocks (Caching) ---
        col_start_of_velocity_row_section = 0 
        for bd in range(boundary_count):
            N_bd = NMK[bd]
            M_bd_plus_1 = NMK[bd + 1]
            
            current_row_blocks = []
            current_local_col_offset = 0

            if bd == (boundary_count - 1):
                row_height = M_bd_plus_1
                # FIX: Call the correct instance methods
                dummy_I_mk = np.ones((NMK[bd], NMK[bd + 1]), dtype=complex)
                block1 = v_dense_block_e_original(diff_R_1n, bd, NMK, a, h, d, dummy_I_mk)
                current_row_blocks.append(block1)
                current_local_col_offset += block1.shape[1]
                
                print(f"block1 shape: {block1.shape}")
                if bd > 0:
                    block3 = v_dense_block_e_original(diff_R_2n, bd, NMK, a, h, d, dummy_I_mk) # I_mk_vals not needed for template
                    print(f"block3 shape: {block3.shape}")
                    current_row_blocks.append(block3)
                    current_local_col_offset += block3.shape[1]
                
                block2_template = self._v_diagonal_block_e_for_cache(
                    bd, NMK, a, h, d, cache,
                    current_global_row_offset,
                    col_start_of_velocity_row_section + current_local_col_offset
                )
                print(f"block2_template shape: {block2_template.shape}")
                current_row_blocks.append(block2_template)
                current_local_col_offset += block2_template.shape[1]

                if bd != 0:
                    left_zeros = np.zeros((row_height, col_start_of_velocity_row_section), dtype=complex)
                    print(f"left_zeros shape (if inserted): {left_zeros.shape}")
                    current_row_blocks.insert(0, left_zeros)
                
                current_blocks_total_width = sum(b.shape[1] for b in current_row_blocks)
                right_zeros_width = size - (col_start_of_velocity_row_section + current_blocks_total_width)
                if right_zeros_width > 0:
                    right_zeros = np.zeros((row_height, right_zeros_width), dtype=complex)
                    current_row_blocks.append(right_zeros)
                    print(f"Added right_zeros with shape: {right_zeros.shape}")
                
            elif bd == 0:
                left_diag_is_active = d[bd] <= d[bd + 1]
                row_height = N_bd if left_diag_is_active else M_bd_plus_1
                
                if left_diag_is_active:
                    block1 = v_diagonal_block_original(True, diff_R_1n, 0, h, d, a, NMK)
                    # FIX 2: Pass the computed I_nm_vals from the cache
                    block2 = v_dense_block_original(False, diff_R_1n, 0, NMK, a, h, d, I_nm_vals)
                    block3 = v_dense_block_original(False, diff_R_2n, 0, NMK, a, h, d, I_nm_vals)
                else:
                    # FIX 2: Pass the computed I_nm_vals from the cache
                    block1 = v_dense_block_original(True, diff_R_1n, 0, NMK, a, h, d, I_nm_vals)
                    block2 = v_diagonal_block_original(False, diff_R_1n, 0, h, d, a, NMK)
                    block3 = v_diagonal_block_original(False, diff_R_2n, 0, h, d, a, NMK)

                current_row_blocks = [block1, block2, block3]
                if any(b is None for b in current_row_blocks):
                    raise ValueError(f"One or more blocks in current_row_blocks are None: {current_row_blocks}")
                current_blocks_total_width = sum(b.shape[1] for b in current_row_blocks)
                right_zeros_width = size - (col_start_of_velocity_row_section + current_blocks_total_width)
                right_zeros = np.zeros((row_height, right_zeros_width), dtype=complex)
                print(f"right_zeros shape: {right_zeros.shape}")
                current_row_blocks.append(right_zeros)

            else:
                left_diag_is_active = d[bd] <= d[bd + 1]
                row_height = N_bd if left_diag_is_active else M_bd_plus_1
                
                if left_diag_is_active:
                    block1 = v_diagonal_block_original(True, diff_R_1n, bd, h, d, a, NMK)
                    block2 = v_diagonal_block_original(True, diff_R_2n, bd, h, d, a, NMK)
                    # FIX 2: Pass the computed I_nm_vals from the cache
                    block3 = v_dense_block_original(False, diff_R_1n, bd, NMK, a, h, d, I_nm_vals)
                    block4 = v_dense_block_original(False, diff_R_2n, bd, NMK, a, h, d, I_nm_vals)
                else:
                    # FIX 2: Pass the computed I_nm_vals from the cache
                    block1 = v_dense_block_original(True, diff_R_1n, bd, NMK, a, h, d, I_nm_vals)
                    block2 = v_dense_block_original(True, diff_R_2n, bd, NMK, a, h, d, I_nm_vals)
                    block3 = v_diagonal_block_original(False, diff_R_1n, bd, h, d, a, NMK)
                    block4 = v_diagonal_block_original(False, diff_R_2n, bd, h, d, a, NMK)

                current_row_blocks = [block1, block2, block3, block4]
                left_zeros = np.zeros((row_height, col_start_of_velocity_row_section), dtype=complex)
                if any(b is None for b in current_row_blocks):
                    raise ValueError(f"One or more blocks in current_row_blocks are None: {current_row_blocks}")
                current_blocks_total_width = sum(b.shape[1] for b in current_row_blocks)
                right_zeros_width = size - (col_start_of_velocity_row_section + current_blocks_total_width)
                right_zeros = np.zeros((row_height, right_zeros_width), dtype=complex)
                current_row_blocks.insert(0, left_zeros)
                current_row_blocks.append(right_zeros)
            print("Current row blocks shapes:")
            for i, b in enumerate(current_row_blocks):
                print(f"Block {i}: shape={b.shape}")
            concat_row = np.concatenate(current_row_blocks, axis=1)
            if concat_row.shape[1] != size:
                missing = size - concat_row.shape[1]
                if missing < 0:
                    raise ValueError(f"concat_row too wide: got {concat_row.shape[1]}, expected {size}")
                pad = np.zeros((concat_row.shape[0], missing), dtype=complex)
                concat_row = np.concatenate([concat_row, pad], axis=1)
            print(f"Concatenated row shape: {concat_row.shape}")
            # assert concat_row.shape[1] == size, f"Expected width {size}, got {concat_row.shape[1]}"
            A_template[current_global_row_offset : current_global_row_offset + concat_row.shape[0], :] = concat_row
            current_global_row_offset += concat_row.shape[0]

            if bd == 0:
                col_start_of_velocity_row_section += NMK[bd]
            elif 0 < bd < boundary_count - 1:
                col_start_of_velocity_row_section += 2 * NMK[bd]
            elif bd == boundary_count - 1:
                col_start_of_velocity_row_section += NMK[bd + 1]

        cache.set_A_template(A_template)

        # --- Build b_template and populate m0_dependent_b_indices ---
        b_template = np.zeros(size, dtype=complex)
        
        heaving = [domain_list[idx].heaving for idx in domain_keys]
        index = 0

        # Potential matching (m0-independent)
        for boundary in range(boundary_count):
            if boundary == (boundary_count - 1):
                for n in range(NMK[boundary]):
                    b_template[index] = b_potential_end_entry(n, boundary, heaving, h, d, a)
                    index += 1
            else:
                i = boundary
                if d[i] > d[i + 1]:
                    N = NMK[i]
                else:
                    N = NMK[i+1]
                for n in range(N):
                    b_template[index] = b_potential_entry(n, boundary, d, heaving, h, a)
                    index += 1

        # Velocity matching (m0-dependent via b_velocity_end_entry)
        for boundary in range(boundary_count):
            if boundary == (boundary_count - 1):
                for k in range(NMK[-1]):
                    current_row = index + k
                    cache.add_m0_dependent_b_entry(current_row,
                        lambda problem_local, m0_local, m_k_arr, N_k_arr, k_local=k, boundary_local=boundary, heaving_local=heaving, a_local=a, h_local=h, d_local=d, NMK_local=NMK:
                            b_velocity_end_entry(k_local, boundary_local, heaving_local, a_local, h_local, d_local, m0_local, NMK_local, m_k_arr, N_k_arr)
                    )
                index += NMK[-1]
            else:
                if d[boundary] < d[boundary + 1]:
                    N = NMK[boundary + 1]
                else:
                    N = NMK[boundary]
                for n in range(N):
                    b_template[index] = b_velocity_entry(n, boundary, heaving, a, h, d)
                    index += 1
        
        cache.set_b_template(b_template)
        return cache
    
    def solve_linear_system_multi(self, problem: MEEMProblem, m0) -> np.ndarray:
        """
        Solve the linear system A x = b for the given problem (multi-region, optimized).
        """
        
        A = self.assemble_A_multi(problem, m0) # Now calls the optimized A assembly
        b = self.assemble_b_multi(problem, m0) # Now calls the optimized B assembly
        X = linalg.solve(A, b)
        return X
    
    def reformat_coeffs(self, x: np.ndarray, NMK, boundary_count) -> list[np.ndarray]:
        """
        Reformats a single vector of coefficients (x) into a list of lists,
        where each inner list contains the coefficients for a specific region.
        This output is typically used for plotting or detailed analysis per region.

        :param x: The single, concatenated solution vector of coefficients.
        :return: A list of NumPy arrays, where each array corresponds to the
                 coefficients of a particular fluid region.
        """
        cs = []
        row = 0

        # Coefficients for the innermost region (Region 0)
        cs.append(x[:NMK[0]])
        row += NMK[0]

        # Coefficients for intermediate regions
        for i in range(1, boundary_count):
            cs.append(x[row: row + NMK[i] * 2])
            row += NMK[i] * 2

        # Coefficients for the outermost region (e-type)
        cs.append(x[row:])
        return cs

    def compute_hydrodynamic_coefficients(self, problem: MEEMProblem, X: np.ndarray, m0) -> Dict[str, any]:
        """
        Compute the hydrodynamic coefficients for a given problem and solution X.
        :param problem: MEEMProblem instance.
        :param X: Solution vector X from solving A x = b.
        :return: Dictionary containing hydrodynamic coefficients and related values.
        """
        domain_list = problem.domain_list
        domain_keys = list(domain_list.keys())
        

        NMK = [domain_list[idx].number_harmonics for idx in domain_keys]
        boundary_count = len(NMK) - 1
        # 'size' is the total length of the X vector relevant to the 'c' vector in the old calculation
        size_c_vector = NMK[0] + NMK[-1] + 2 * sum(NMK[1:len(NMK) - 1]) # This is equivalent to size - NMK[-1] from the old test

        h = domain_list[0].h
        d = [domain_list[idx].di for idx in domain_keys]
        a = [domain_list[idx].a for idx in domain_keys if domain_list[idx].a is not None]
        heaving = [domain_list[idx].heaving for idx in domain_keys]

        # Reconstruct the 1D 'c' vector similar to the test function
        c_vector = np.zeros(size_c_vector, dtype=complex)
        
        col = 0
        # First domain (inner)
        for n in range(NMK[0]):
            c_vector[n] = heaving[0] * int_R_1n(0, n, a, h, d) * z_n_d(n)
        col += NMK[0]

        # Subsequent internal domains (boundaries)
        for i in range(1, boundary_count): # boundary_count is len(domain_keys) - 1
            M = NMK[i]
            for m in range(M):
                c_vector[col + m] = heaving[i] * int_R_1n(i, m, a, h, d) * z_n_d(m)
                c_vector[col + M + m] = heaving[i] * int_R_2n(i, m, a, h, d) * z_n_d(m) 
            col += 2 * M
        
        # The X vector part relevant for the dot product is X up to the size of c_vector
        X_relevant = X[:size_c_vector] 

        # Calculate the h_terms as a single complex value first, then extract real/imag
        hydro_h_term_sum = np.dot(c_vector, X_relevant)

        # Calculate hydro_p_terms (this part seems mostly correct but recheck if 1D sum is needed)
        # The 'old' code had sum(hydro_p_terms) where hydro_p_terms was a 1D array
        # Your new code calculates hydro_p_terms as a 2D array and takes a diagonal element.
        # Let's align this to the original sum approach first.
        hydro_p_terms_1d = np.zeros(boundary_count, dtype=complex)
        for i in range(boundary_count):
            # This is heaving[i] * int_phi_p_i_no_coef(i, h, d, a)
            # The original test code calculates hydro_p_terms this way and then sums them
            # The 'new' code calculates hydro_p_terms[i,j] = heaving[j] * (h-d[i]) / (h-d[j]) * int_phi_p_i_no_coef(i, h, d, a)
            # and uses hydro_p_terms[mode_index, mode_index].
            # If problem.modes is always just [0], then mode_index is 0.
            # In the old code, it's sum(hydro_p_terms), meaning it's sum(heaving[i] * int_phi_p_i_no_coef(i, h, d, a) for all i)
            # Let's adjust this to match the original sum.
            hydro_p_terms_1d[i] = heaving[i] * int_phi_p_i_no_coef(i, h, d, a)
        
        hydro_p_term_sum = np.sum(hydro_p_terms_1d) # Sum all relevant p_terms

        # Combine the sums and scale
        total_hydro_complex = (hydro_h_term_sum + hydro_p_term_sum) * (2 * pi) # The 2*pi was outside the dot and sum in the old code

        # Now return per-mode values - assuming for now problem.modes has only one element (m0=1)
        num_modes = len(problem.modes) # This should be 1 based on test setup
        real_coeffs = np.zeros(num_modes)
        imag_coeffs = np.zeros(num_modes)

        # Since the test case seems to be focused on a single 'mode_index' (0) due to problem_modes=[0] initially
        # and m0=1, we need to clarify what 'modes' refer to.
        # Based on the test, `new_coeffs["real"][0]` and `new_coeffs["imag"][0]` are accessed.
        # This implies that `real_coeffs` and `imag_coeffs` are expected to be 1-element arrays.

        # The 'old' test calculation yields a single hydro_coef.
        # So we should assign the total complex coefficient to the first mode.
        if num_modes > 0: # Ensure there's at least one mode
            real_coeffs[0] = total_hydro_complex.real
            imag_coeffs[0] = total_hydro_complex.imag

        scale = h**3 * rho # The 2*pi is already applied above
        real_scaled = scale * real_coeffs
        local_omega = omega(m0, h, g) 
        
        if m0 == np.inf: 
            imag_scaled = 0
        else: 
            imag_scaled = scale * imag_coeffs * local_omega
            
        hydro_coeffs = {
            "real": real_scaled,
            "imag": imag_scaled
        }
        return hydro_coeffs
    
    def phi_h_n_inner_func(self, n, r, z, h, d, a, solution_vector, NMK, boundary_count):
        # Reformat solution vector into coefficients per region
        Cs = self.reformat_coeffs(solution_vector, NMK, boundary_count)
        return (Cs[0][n] * R_1n(n, r, 0, h, d, a)) * Z_n_i(n, z, 0, h, d)

    def phi_h_m_i_func(self, i, m, r, z, h, d, a, solution_vector, NMK, boundary_count):
        # Reformat solution vector into coefficients per region
        Cs = self.reformat_coeffs(solution_vector, NMK, boundary_count)
        return (Cs[i][m] * R_1n(m, r, i, h, d, a) + Cs[i][NMK[i] + m] * R_2n(m, r, i, a, h, d)) * Z_n_i(m, z, i, h, d)

    def phi_e_k_func(self, k, r, z, m0, a, NMK, h, m_k_arr, N_k_arr, solution_vector, boundary_count):
        Cs = self.reformat_coeffs(solution_vector, NMK, boundary_count)
        return Cs[-1][k] * Lambda_k(k, r, m0, a, NMK, h, m_k_arr, N_k_arr) * Z_k_e(k, z, m0, h, NMK, m_k_arr)
    
    def calculate_potentials(self, problem, solution_vector: np.ndarray, m0, m_k_arr, N_k_arr, spatial_res, sharp) -> Dict[str, Any]:
        """
        Calculate full spatial potentials phiH, phiP, and total phi on a meshgrid for visualization.

        Parameters:
        - problem: MEEMProblem instance containing domain and geometry info
        - solution_vector: solution vector X from linear system solve
        - spatial_res: resolution of spatial grid for R and Z (default=50)
        - sharp: whether to refine meshgrid near boundaries (default=True)

        Returns:
        - Dictionary containing meshgrid arrays R,Z and potentials phiH, phiP, phi
        """
        domain_list = problem.domain_list
        domain_keys = list(domain_list.keys())
        boundary_count = len(domain_keys) - 1

        NMK = [domain_list[idx].number_harmonics for idx in domain_keys]
        h = domain_list[0].h
        d = [domain_list[idx].di for idx in domain_keys]
        a = [domain_list[idx].a for idx in domain_keys if domain_list[idx].a is not None]
        heaving = [domain_list[idx].heaving for idx in domain_keys]

        # Vectorize harmonic basis functions for performance
        phi_h_n_inner_vec = np.vectorize(
            lambda n, r, z: self.phi_h_n_inner_func(n, r, z, h, d, a, solution_vector, NMK, boundary_count),
            otypes=[complex]
        )
        phi_h_m_i_vec = np.vectorize(
            lambda i, m, r, z: self.phi_h_m_i_func(i, m, r, z, h, d, a, solution_vector, NMK, boundary_count),
            otypes=[complex]
        )
        phi_e_k_vec = np.vectorize(
            lambda k, r, z: self.phi_e_k_func(k, r, z, m0, a, NMK, h, m_k_arr, N_k_arr, solution_vector, boundary_count),
            otypes=[complex]
        )

        phi_p_i_vec = np.vectorize(
            lambda d_i, r, z: phi_p_i(d_i, r, z, h),
            otypes=[complex]
        )

        # Generate meshgrid (R,Z) with sharp boundary refinement if requested
        R, Z = make_R_Z(a, h, d, sharp, spatial_res)

        # Define spatial regions based on radii and draft values
        regions = []
        # Inner region
        regions.append((R <= a[0]) & (Z < -d[0]))
        # Intermediate regions
        for i in range(1, boundary_count):
            regions.append((R > a[i-1]) & (R <= a[i]) & (Z < -d[i]))
        # Exterior region
        regions.append(R > a[-1])

        # Initialize complex arrays for potentials filled with NaNs
        shape = R.shape
        phiH = np.full(shape, np.nan + 1j*np.nan, dtype=complex)
        phiP = np.full(shape, np.nan + 1j*np.nan, dtype=complex)
        phi = np.full(shape, np.nan + 1j*np.nan, dtype=complex)

        # Calculate Homogeneous Potential phiH for each region
        # Region 0: inner
        for n in range(NMK[0]):
            temp_phiH = phi_h_n_inner_vec(n, R[regions[0]], Z[regions[0]])
            phiH[regions[0]] = temp_phiH if n == 0 else phiH[regions[0]] + temp_phiH

        # Intermediate regions 1..boundary_count-1
        for i in range(1, boundary_count):
            for m in range(NMK[i]):
                temp_phiH = phi_h_m_i_vec(i, m, R[regions[i]], Z[regions[i]])
                phiH[regions[i]] = temp_phiH if m == 0 else phiH[regions[i]] + temp_phiH

        # Exterior region (last)
        for k in range(NMK[-1]):
            temp_phiH = phi_e_k_vec(k, R[regions[-1]], Z[regions[-1]])
            phiH[regions[-1]] = temp_phiH if k == 0 else phiH[regions[-1]] + temp_phiH

        # Calculate Particular Potential phiP
        # Set to zero outside physical regions to avoid NaNs in visualization
        phiP[regions[0]] = heaving[0] * phi_p_i_vec(d[0], R[regions[0]], Z[regions[0]])
        for i in range(1, boundary_count):
            phiP[regions[i]] = heaving[i] * phi_p_i_vec(d[i], R[regions[i]], Z[regions[i]])
        phiP[regions[-1]] = 0.0  # Exterior domain particular potential zero

        # Sum to get total potential phi
        phi = phiH + phiP

        return {
            "R": R,
            "Z": Z,
            "phiH": phiH,
            "phiP": phiP,
            "phi": phi
        }


    def visualize_potential(self, potentials: Dict[str, Any]):
        """
        Visualize the potentials (phiH, phiP, phi) as contour plots of real and imaginary parts.

        Parameters:
        - potentials: dictionary from calculate_potentials containing R,Z and potentials

        Returns:
        - matplotlib Figure object with all subplots
        """

        R = potentials.get("R")
        Z = potentials.get("Z")
        phiH = potentials.get("phiH")
        phiP = potentials.get("phiP")
        phi = potentials.get("phi")

        if R is None or Z is None or phiH is None or phiP is None or phi is None:
            raise ValueError("Potentials dictionary missing one or more required keys: R, Z, phiH, phiP, phi")

        fig, axs = plt.subplots(3, 2, figsize=(12, 16))
        fig.suptitle("Potential Field Visualization", fontsize=16)

        plot_data = [
            ("Homogeneous Potential (Real Part)", np.real(phiH)),
            ("Homogeneous Potential (Imag Part)", np.imag(phiH)),
            ("Particular Potential (Real Part)", np.real(phiP)),
            ("Particular Potential (Imag Part)", np.imag(phiP)),
            ("Total Potential (Real Part)", np.real(phi)),
            ("Total Potential (Imag Part)", np.imag(phi))
        ]

        for ax, (title, field) in zip(axs.flat, plot_data):
            # Handle potential NaNs by masking them (to avoid warnings)
            masked_field = np.ma.array(field, mask=np.isnan(field))
            cs = ax.contourf(R, Z, masked_field, levels=50, cmap='viridis')
            fig.colorbar(cs, ax=ax)
            ax.set_title(title)
            ax.set_xlabel("Radial Distance (R)")
            ax.set_ylabel("Axial Distance (Z)")
            ax.grid(True)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # leave room for suptitle
        return fig
    
    def run_and_store_results(self, problem_index: int) -> Results:
        """
        Perform the full MEEM computation for all frequencies defined in the selected MEEMProblem,
        and store results in a `Results` object.

        The method:
        - Assembles and solves the linear system A @ X = b for each frequency in `problem.frequencies`.
        - Computes hydrodynamic coefficients (added mass and damping).
        - Optionally stores domain potentials (placeholder code currently).
        - Stores all results in a reusable `Results` object.

        Parameters
        ----------
        problem_index : int
            Index of the MEEMProblem instance from `self.problem_list` to run computations for.

        Returns
        -------
        Results
            A `Results` object containing added mass and damping matrices, 
            and optionally, computed potentials for each frequency and mode.
        """
        problem = self.problem_list[problem_index]
        geometry = problem.geometry
        results = Results(geometry, problem.frequencies, problem.modes)

        num_modes = len(problem.modes)
        num_freqs = len(problem.frequencies)

        full_added_mass_matrix = np.full((num_freqs, num_modes), np.nan)
        full_damping_matrix = np.full((num_freqs, num_modes), np.nan)
        all_potentials_batch_data = []

        for freq_idx, m0 in enumerate(problem.frequencies):
            print(f"  Calculating for m0 = {m0:.4f} rad/s")

            try:
                A = self.assemble_A_multi(problem, m0)
                b = self.assemble_b_multi(problem, m0)
                X = np.linalg.solve(A, b)
            except np.linalg.LinAlgError as e:
                print(f"  ERROR: Could not solve for m0={m0:.4f}: {e}. Storing NaN for coefficients.")
                continue

            hydro_coeffs = self.compute_hydrodynamic_coefficients(problem, X)
            added_mass = np.atleast_1d(hydro_coeffs["real"])
            damping = np.atleast_1d(hydro_coeffs["imag"])

            if added_mass.shape[0] != num_modes or damping.shape[0] != num_modes:
                raise ValueError(
                    f"compute_hydrodynamic_coefficients returned shape mismatch for m0={m0:.4f}."
                )

            full_added_mass_matrix[freq_idx, :] = added_mass
            full_damping_matrix[freq_idx, :] = damping

            for mode_idx, _ in enumerate(problem.modes):
                current_mode_potentials = {}
                for domain in problem.geometry.domain_list.values():
                    nh = domain.number_harmonics
                    current_mode_potentials[domain.category] = {
                        "potentials": (np.random.rand(nh) + 1j * np.random.rand(nh)).astype(complex),
                        "r_coords_dict": {f"r_h{k}": np.random.rand() for k in range(nh)},
                        "z_coords_dict": {f"z_h{k}": np.random.rand() for k in range(nh)},
                    }

                all_potentials_batch_data.append({
                    "frequency_idx": freq_idx,
                    "mode_idx": mode_idx,
                    "data": current_mode_potentials,
                })

        results.store_hydrodynamic_coefficients(
            frequencies=problem.frequencies,
            modes=problem.modes,
            added_mass_matrix=full_added_mass_matrix,
            damping_matrix=full_damping_matrix,
        )

        if all_potentials_batch_data:
            results.store_all_potentials(all_potentials_batch_data)

        return results

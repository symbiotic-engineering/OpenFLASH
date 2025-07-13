#meem_engine.py
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
    
    # Renaming current assemble_A_multi to indicate it's the full assembly
    def _full_assemble_A_multi(self, problem: MEEMProblem, m0) -> np.ndarray:
        """
        Assemble the system matrix A for a given problem (full re-calculation).
        This is essentially the original assemble_A_multi.
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
        a = [val for val in a if val is not None]

        ###########################################################################
        # Potential Matching

        col_offset = 0 # Tracks the start column for the current block
        row_offset = 0 # Tracks the start row for the current block

        for bd in range(boundary_count):
            N_left = NMK[bd] # Number of harmonics for the left domain in this boundary
            M_right = NMK[bd + 1] # Number of harmonics for the right domain in this boundary

            if bd == (boundary_count - 1):  # i-e boundary (Inner-Exterior)
                if bd == 0:  # One cylinder case (Inner-Exterior directly)
                    for n in range(N_left): # Inner harmonics (N)
                        A[row_offset + n][col_offset + n] = (h - d[bd]) * R_1n(n, a[bd], bd, h, d, a)
                        for m in range(M_right): # Exterior harmonics (K)
                            # This term is m0-dependent via Lambda_k and I_mk
                            A[row_offset + n][col_offset + N_left + m] = - I_mk_og(n, m, bd, d, m0, h, NMK) * Lambda_k_og(m, a[bd], m0, a, NMK, h)
                    row_offset += N_left
                else: # Intermediate-Exterior boundary (e.g., in a 3+ domain setup)
                    for n in range(N_left): # Left domain harmonics (M_i)
                        A[row_offset + n][col_offset + n] = (h - d[bd]) * R_1n(n, a[bd], bd, h, d, a)
                        A[row_offset + n][col_offset + N_left + n] = (h - d[bd]) * R_2n(n, a[bd], bd, a, h, d)
                        for m in range(M_right): # Exterior harmonics (K)
                            # This term is m0-dependent via Lambda_k and I_mk
                            A[row_offset + n][col_offset + 2*N_left + m] = - I_mk_og(n, m, bd, d, m0, h, NMK) * Lambda_k_og(m, a[bd], m0, a, NMK, h)
                    row_offset += N_left
            elif bd == 0: # i-i boundary (Inner-Intermediate)
                left_diag_is_active = d[bd] > d[bd + 1] # Based on which region has the deeper depth
                if left_diag_is_active: # Left domain (Inner) gets diagonal entries
                    for n in range(N_left): # Inner harmonics (N)
                        A[row_offset + n][col_offset + n] = (h - d[bd]) * R_1n(n, a[bd], bd, h, d, a)
                        for m in range(M_right): # Intermediate harmonics (M)
                            A[row_offset + n][col_offset + N_left + m] = - I_nm(n, m, bd, d, h) * R_1n(m, a[bd], bd + 1, h, d, a)
                            A[row_offset + n][col_offset + N_left + M_right + m] = - I_nm(n, m, bd, d, h) * R_2n(m, a[bd], bd + 1, a, h, d)
                    row_offset += N_left
                else: # Right domain (Intermediate) gets diagonal entries
                    for m in range(M_right): # Intermediate harmonics (M)
                        for n in range(N_left): # Inner harmonics (N)
                            A[row_offset + m][col_offset + n] = I_nm(n, m, bd, d, h) * R_1n(n, a[bd], bd, h, d, a)
                        A[row_offset + m][col_offset + N_left + m] = - (h - d[bd + 1]) * R_1n(m, a[bd], bd + 1, h, d, a)
                        A[row_offset + m][col_offset + N_left + M_right + m] = - (h - d[bd + 1]) * R_2n(m, a[bd], bd + 1, a, h, d)
                    row_offset += M_right
                col_offset += N_left
            else:  # i-i boundary (Intermediate-Intermediate)
                left_diag_is_active = d[bd] > d[bd + 1]
                if left_diag_is_active:
                    for n in range(N_left):
                        A[row_offset + n][col_offset + n] = (h - d[bd]) * R_1n(n, a[bd], bd, h, d, a)
                        A[row_offset + n][col_offset + N_left + n] = (h - d[bd]) * R_2n(n, a[bd], bd, h, d)
                        for m in range(M_right):
                            A[row_offset + n][col_offset + 2*N_left + m] = - I_nm(n, m, bd, d, h) * R_1n(m, a[bd], bd + 1, h, d, a)
                            A[row_offset + n][col_offset + 2*N_left + M_right + m] = - I_nm(n, m, bd, d, h) * R_2n(m, a[bd], bd + 1, a, h, d)
                    row_offset += N_left
                else:
                    for m in range(M_right):
                        for n in range(N_left):
                            A[row_offset + m][col_offset + n] = I_nm(n, m, bd, d, h) * R_1n(n, a[bd], bd, h, d, a)
                            A[row_offset + m][col_offset + N_left + n] = I_nm(n, m, bd, d, h) * R_2n(n, a[bd], bd, a, h, d)
                        A[row_offset + m][col_offset + 2*N_left + m] = - (h - d[bd + 1]) * R_1n(m, a[bd], bd + 1, h, d, a)
                        A[row_offset + m][col_offset + 2*N_left + M_right + m] = - (h - d[bd + 1]) * R_2n(m, a[bd], bd + 1, a, h, d)
                    row_offset += M_right
                col_offset += 2 * N_left

        ###########################################################################
        # Velocity Matching 

        col_offset = 0 # Reset column offset for velocity matching section
        for bd in range(boundary_count):
            N_left = NMK[bd]
            M_right = NMK[bd + 1]

            if bd == (boundary_count - 1):  # i-e boundary
                if bd == 0:  # one cylinder
                    for m in range(M_right): # Exterior harmonics (K)
                        for n in range(N_left): # Inner harmonics (N)
                            A[row_offset + m][col_offset + n] = - I_mk_og(n, m, bd, d, m0, h, NMK) * diff_R_1n(n, a[bd], bd, h, d, a)
                        # This term is m0-dependent via diff_Lambda_k
                        A[row_offset + m][col_offset + N_left + m] = h * diff_Lambda_k_og(m, a[bd], m0, NMK, h, a)
                    row_offset += M_right # Number of rows added here is M_right (K for exterior)
                else: # Intermediate-Exterior boundary
                    for m in range(M_right):
                        for n in range(N_left):
                            A[row_offset + m][col_offset + n] = - I_mk_og(n, m, bd, d, m0, h, NMK) * diff_R_1n(n, a[bd], bd, h, d, a)
                            A[row_offset + m][col_offset + N_left + n] = - I_mk_og(n, m, bd, d, m0, h, NMK) * diff_R_2n(n, a[bd], bd, h, d, a)
                        # This term is m0-dependent via diff_Lambda_k
                        A[row_offset + m][col_offset + 2*N_left + m] = h * diff_Lambda_k_og(m, a[bd], m0, NMK, h, a)
                    row_offset += M_right
            elif bd == 0:
                left_diag_is_active = d[bd] < d[bd + 1]
                if left_diag_is_active:
                    for n in range(N_left):
                        A[row_offset + n][col_offset + n] = - (h - d[bd]) * diff_R_1n(n, a[bd], bd, h, d, a)
                        for m in range(M_right):
                            A[row_offset + n][col_offset + N_left + m] = I_nm(n, m, bd, d, h) * diff_R_1n(m, a[bd], bd + 1, h, d, a)
                            A[row_offset + n][col_offset + N_left + M_right + m] = I_nm(n, m, bd, d, h) * diff_R_2n(m, a[bd], bd + 1, h, d, a)
                    row_offset += N_left
                else:
                    for m in range(M_right):
                        for n in range(N_left):
                            A[row_offset + m][col_offset + n] = - I_nm(n, m, bd, d, h) * diff_R_1n(n, a[bd], bd, h, d, a)
                        A[row_offset + m][col_offset + N_left + m] = (h - d[bd + 1]) * diff_R_1n(m, a[bd], bd + 1, h, d, a)
                        A[row_offset + m][col_offset + N_left + M_right + m] = (h - d[bd + 1]) * diff_R_2n(m, a[bd], bd + 1, h, d, a)
                    row_offset += M_right
                col_offset += N_left
            else:  # i-i boundary
                left_diag_is_active = d[bd] < d[bd + 1]
                if left_diag_is_active:
                    for n in range(N_left):
                        A[row_offset + n][col_offset + n] = - (h - d[bd]) * diff_R_1n(n, a[bd], bd, h, d, a)
                        A[row_offset + n][col_offset + N_left + n] = - (h - d[bd]) * diff_R_2n(n, a[bd], bd, h, d, a)
                        for m in range(M_right):
                            A[row_offset + n][col_offset + 2*N_left + m] = I_nm(n, m, bd, d, h) * diff_R_1n(m, a[bd], bd + 1, h, d, a)
                            A[row_offset + n][col_offset + 2*N_left + M_right + m] = I_nm(n, m, bd, d, h) * diff_R_2n(m, a[bd], bd + 1, h, d, a)
                    row_offset += N_left
                else:
                    for m in range(M_right):
                        for n in range(N_left):
                            A[row_offset + m][col_offset + n] = - I_nm(n, m, bd, d, h) * diff_R_1n(n, a[bd], bd, h, d, a)
                            A[row_offset + m][col_offset + N_left + n] = - I_nm(n, m, bd, d, h) * diff_R_2n(n, a[bd], bd, a, h, d)
                        A[row_offset + m][col_offset + 2*N_left + m] = (h - d[bd + 1]) * diff_R_1n(m, a[bd], bd + 1, h, d, a)
                        A[row_offset + m][col_offset + 2*N_left + M_right + m] = (h - d[bd + 1]) * diff_R_2n(m, a[bd], bd + 1, h, d, a)
                    row_offset += M_right
                col_offset += 2 * N_left

        return A

    # Now, the optimized assemble_A_multi method that uses the cache
    def assemble_A_multi(self, problem: MEEMProblem, m0) -> np.ndarray:
        """
        Assemble the system matrix A for a given problem, leveraging pre-computed
        m0-independent parts and updating only m0-dependent entries.
        """

        cache = self.cache_list[problem]
        A = cache.get_A_template().copy() # Start with the template

        # Pre-compute m_k and N_k arrays ONCE for this m0
        NMK_last = problem.domain_list[list(problem.domain_list.keys())[-1]].number_harmonics

        # Check if m_k_arr and N_k_arr are already computed for this m0
        
        m_k_arr_computed = np.array([cache.m_k_entry_func(k, m0, problem.domain_list[0].h) for k in range(NMK_last)])
        N_k_arr_computed = np.array([cache.N_k_func(k, m0, problem.domain_list[0].h, NMK_last, m_k_arr_computed) for k in range(NMK_last)])

        # Store the computed arrays in the cache for later retrieval
        cache.set_precomputed_m_k_N_k(m_k_arr_computed, N_k_arr_computed)

        for row, col, calc_func in cache.m0_dependent_A_indices:
            # Pass the pre-computed arrays (now directly from cache attributes or local computed ones) to the lambda
            # Using the locally computed ones here for clarity, but they are now also in cache.
            A[row, col] = calc_func(problem, m0, m_k_arr_computed, N_k_arr_computed)
        return A
    
    # Renaming current assemble_b_multi to indicate it's the full assembly
    def _full_assemble_b_multi(self, problem: MEEMProblem, m0) -> np.ndarray:
        """
        Assemble the right-hand side vector b for a given problem (multi-region, full re-calculation).
        This is essentially the original assemble_b_multi.
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
        a = [val for val in a if val is not None]

        heaving = [domain_list[idx].heaving for idx in domain_keys]

        index = 0

        # Potential matching (m0-independent)
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

        # Velocity matching (m0-dependent via b_velocity_end_entry)
        for boundary in range(boundary_count):
            if boundary == (boundary_count - 1):  # i-e boundary
                for k in range(NMK[-1]):
                    assert index < size, f"Index {index} out of bounds for size {size}"  # Explicit check
                    b[index] = b_velocity_end_entry_og(k, boundary, heaving, a, h, d, m0, NMK)
                    index += 1
            else:  # i-i boundary (m0-independent)
                if d[boundary] < d[boundary + 1]:
                    N = NMK[boundary + 1]
                else:
                    N = NMK[boundary]
                for n in range(N):
                    assert index < size, f"Index {index} out of bounds for size {size}"  # Explicit check
                    b[index] = b_velocity_entry(n, boundary, heaving, a, h, d)
                    index += 1
        return b

    # Now, the optimized assemble_b_multi method that uses the cache
    def assemble_b_multi(self, problem: MEEMProblem, m0) -> np.ndarray:
        """
        Assemble the right-hand side vector b for a given problem, leveraging pre-computed
        m0-independent parts and updating only m0-dependent entries.
        """
        cache = self.cache_list[problem]
        b = cache.get_b_template().copy() # Start with the template

       
        # Retrieve m_k_arr and N_k_arr from the cache
        if cache.m_k_arr is None or cache.N_k_arr is None:
            print("WARNING: m_k_arr or N_k_arr were None in assemble_b_multi, re-computing.")

            NMK_last = problem.domain_list[list(problem.domain_list.keys())[-1]].number_harmonics
            m_k_arr_computed = np.array([cache.m_k_entry_func(k, m0, problem.domain_list[0].h) for k in range(NMK_last)])
            N_k_arr_computed = np.array([cache.N_k_func(k, m0, problem.domain_list[0].h, NMK_last, m_k_arr_computed) for k in range(NMK_last)])
            cache.set_precomputed_m_k_N_k(m_k_arr_computed, N_k_arr_computed)


        for row, calc_func in cache.m0_dependent_b_indices:
            # Pass the pre-computed arrays from the cache to the lambda
            b[row] = calc_func(problem, m0, cache.m_k_arr, cache.N_k_arr)
        return b

    # New method to build the ProblemCache
    def _build_problem_cache(self, problem: MEEMProblem) -> ProblemCache:
        """
        Analyzes the problem and pre-computes m0-independent parts of A and b,
        and identifies indices for m0-dependent parts.
        """
        cache = ProblemCache(problem)

        domain_list = problem.domain_list
        domain_keys = list(domain_list.keys())
        boundary_count = len(domain_keys) - 1
        
        NMK = [domain_list[idx].number_harmonics for idx in domain_keys]
        size = NMK[0] + NMK[-1] + 2 * sum(NMK[1:-1])

        # --- Common parameters extracted once ---
        h = domain_list[0].h
        d = [domain_list[idx].di for idx in domain_keys]
        a = [domain_list[idx].a for idx in domain_keys]
        a_filtered = [val for val in a if val is not None]

        # Store references to the actual functions needed for m_k and N_k calculation
        # These are the *functions* themselves, not their results.
        print("Setting m_k and N_k functions in cache.")
        cache.set_m_k_and_N_k_funcs(m_k_entry, N_k_multi) # Original m_k_entry and N_k from multi_equations

        # --- A Matrix Template ---
        A_template = np.zeros((size, size), dtype=complex)
        
        col_offset = 0
        row_offset = 0

        # Potential Matching (mostly m0-independent)
        for bd in range(boundary_count):
            N_left = NMK[bd]
            M_right = NMK[bd + 1]

            if bd == (boundary_count - 1): # i-e boundary
                if bd == 0: # one cylinder
                    for n in range(N_left):
                        # m0-independent part directly assigned
                        A_template[row_offset + n][col_offset + n] = (h - d[bd]) * R_1n(n, a_filtered[bd], bd, h, d, a_filtered)
                        for m in range(M_right):
                            current_row = row_offset + n
                            current_col = col_offset + N_left + m
                            
                            # IMPORTANT: Now the lambda takes m_k_arr and N_k_arr
                            # It uses *m_k_entry* from multi_equations.py to calculate m_k_arr
                            # and *N_k* from multi_equations.py to calculate N_k_arr
                            # But these arrays are computed *once* per m0 in assemble_A_multi,
                            # NOT per element here.
                            cache.add_m0_dependent_A_entry(current_row, current_col,
                                # Here, the lambda will expect pre-calculated m_k_arr and N_k_arr
                                lambda problem_local, m0_local, m_k_arr, N_k_arr, n_local=n, m_local=m, bd_local=bd, d_local=d, h_local=h, NMK_local=NMK, a_local=a_filtered, a_bd_local=a_filtered[bd]:
                                    - I_mk(n_local, m_local, bd_local, d_local, m0_local, h_local, NMK_local, m_k_arr, N_k_arr) # Pass m_k_arr, N_k_arr
                                    * Lambda_k(m_local, a_bd_local, m0_local, a_local, NMK_local, h_local, m_k_arr, N_k_arr) # Pass m_k_arr, N_k_arr
                            )
                    row_offset += N_left
                else: # Intermediate-Exterior boundary (e.g., in a 3+ domain setup)
                    for n in range(N_left):
                        A_template[row_offset + n][col_offset + n] = (h - d[bd]) * R_1n(n, a_filtered[bd], bd, h, d, a_filtered)
                        A_template[row_offset + n][col_offset + N_left + n] = (h - d[bd]) * R_2n(n, a_filtered[bd], bd, a_filtered, h, d)
                        for m in range(M_right):
                            current_row = row_offset + n
                            current_col = col_offset + 2*N_left + m
                            cache.add_m0_dependent_A_entry(current_row, current_col,
                                lambda problem_local, m0_local, m_k_arr, N_k_arr, n_local=n, m_local=m, bd_local=bd, d_local=d, h_local=h, NMK_local=NMK, a_local=a_filtered, a_bd_local=a_filtered[bd]:
                                    - I_mk(n_local, m_local, bd_local, d_local, m0_local, h_local, NMK_local, m_k_arr, N_k_arr) # Pass m_k_arr, N_k_arr
                                    * Lambda_k(m_local, a_bd_local, m0_local, a_local, NMK_local, h_local, m_k_arr, N_k_arr) # Pass m_k_arr, N_k_arr
                            )
                    row_offset += N_left
            # For the cases where m0-dependent terms occur:
            elif bd == 0: # i-i boundary (Inner-Intermediate)
                left_diag_is_active = d[bd] > d[bd + 1]
                if left_diag_is_active:
                    for n in range(N_left):
                        A_template[row_offset + n][col_offset + n] = (h - d[bd]) * R_1n(n, a_filtered[bd], bd, h, d, a_filtered)
                        for m in range(M_right):
                            A_template[row_offset + n][col_offset + N_left + m] = - I_nm(n, m, bd, d, h) * R_1n(m, a_filtered[bd], bd + 1, h, d, a_filtered)
                            A_template[row_offset + n][col_offset + N_left + M_right + m] = - I_nm(n, m, bd, d, h) * R_2n(m, a_filtered[bd], bd + 1, a_filtered, h, d)
                    row_offset += N_left
                else:
                    for m in range(M_right):
                        for n in range(N_left):
                            A_template[row_offset + m][col_offset + n] = I_nm(n, m, bd, d, h) * R_1n(n, a_filtered[bd], bd, h, d, a_filtered)
                        A_template[row_offset + m][col_offset + N_left + m] = - (h - d[bd + 1]) * R_1n(m, a_filtered[bd], bd + 1, h, d, a_filtered)
                        A_template[row_offset + m][col_offset + N_left + M_right + m] = - (h - d[bd + 1]) * R_2n(m, a_filtered[bd], bd + 1, a_filtered, h, d)
                    row_offset += M_right
                col_offset += N_left
            else:  # i-i boundary (Intermediate-Intermediate)
                left_diag_is_active = d[bd] > d[bd + 1]
                if left_diag_is_active:
                    for n in range(N_left):
                        A_template[row_offset + n][col_offset + n] = (h - d[bd]) * R_1n(n, a_filtered[bd], bd, h, d, a_filtered)
                        A_template[row_offset + n][col_offset + N_left + n] = (h - d[bd]) * R_2n(n, a_filtered[bd], bd, h, d)
                        for m in range(M_right):
                            A_template[row_offset + n][col_offset + 2*N_left + m] = - I_nm(n, m, bd, d, h) * R_1n(m, a_filtered[bd], bd + 1, h, d, a_filtered)
                            A_template[row_offset + n][col_offset + 2*N_left + M_right + m] = - I_nm(n, m, bd, d, h) * R_2n(m, a_filtered[bd], bd + 1, a_filtered, h, d)
                    row_offset += N_left
                else:
                    for m in range(M_right):
                        for n in range(N_left):
                            A_template[row_offset + m][col_offset + n] = I_nm(n, m, bd, d, h) * R_1n(n, a_filtered[bd], bd, h, d, a_filtered)
                            A_template[row_offset + m][col_offset + N_left + n] = I_nm(n, m, bd, d, h) * R_2n(m, a_filtered[bd], bd, a_filtered, h, d)
                        A_template[row_offset + m][col_offset + 2*N_left + m] = - (h - d[bd + 1]) * R_1n(m, a_filtered[bd], bd + 1, h, d, a_filtered)
                        A_template[row_offset + m][col_offset + 2*N_left + M_right + m] = - (h - d[bd + 1]) * R_2n(m, a_filtered[bd], bd + 1, a_filtered, h, d)
                    row_offset += M_right
                col_offset += 2 * N_left

        # Velocity Matching (some m0-dependent terms)
        col_offset = 0 # Reset column offset for velocity matching section
        for bd in range(boundary_count):
            N_left = NMK[bd]
            M_right = NMK[bd + 1]

            if bd == (boundary_count - 1): # i-e boundary
                if bd == 0: # one cylinder
                    for m in range(M_right):
                        for n in range(N_left):
                            current_row = row_offset + m
                            current_col = col_offset + n
                            diff_R_1n_val = diff_R_1n(n, a_filtered[bd], bd, h, d, a_filtered)
                            cache.add_m0_dependent_A_entry(current_row, current_col,
                                # Pass m_k_arr, N_k_arr
                                lambda problem_local, m0_local, m_k_arr, N_k_arr, n_local=n, m_local=m, bd_local=bd, d_local=d, h_local=h, NMK_local=NMK, diff_R_1n_val=diff_R_1n_val:
                                    - I_mk(n_local, m_local, bd_local, d_local, m0_local, h_local, NMK_local, m_k_arr, N_k_arr) * diff_R_1n_val
                            )
                        # This term is m0-dependent via diff_Lambda_k
                        current_row = row_offset + m
                        current_col = col_offset + N_left + m
                        cache.add_m0_dependent_A_entry(current_row, current_col,
                            # Pass m_k_arr, N_k_arr
                            lambda problem_local, m0_local, m_k_arr, N_k_arr, m_local=m, bd_local=bd, h_local=h, NMK_local=NMK, a_local=a_filtered, a_bd_local=a_filtered[bd]:
                                h_local * diff_Lambda_k(m_local, a_bd_local, m0_local, NMK_local, h_local, a_local, m_k_arr, N_k_arr)
                        )
                    row_offset += M_right
                else: # Intermediate-Exterior boundary
                    for m in range(M_right):
                        for n in range(N_left):
                            current_row = row_offset + m
                            current_col = col_offset + n
                            diff_R_1n_val = diff_R_1n(n, a_filtered[bd], bd, h, d, a_filtered)
                            cache.add_m0_dependent_A_entry(current_row, current_col,
                                lambda problem_local, m0_local, m_k_arr, N_k_arr, n_local=n, m_local=m, bd_local=bd, d_local=d, h_local=h, NMK_local=NMK, diff_R_1n_val=diff_R_1n_val:
                                    - I_mk(n_local, m_local, bd_local, d_local, m0_local, h_local, NMK_local, m_k_arr, N_k_arr) * diff_R_1n_val
                            )
                            current_col = col_offset + N_left + n
                            diff_R_2n_val = diff_R_2n(n, a_filtered[bd], bd, h, d, a_filtered)
                            cache.add_m0_dependent_A_entry(current_row, current_col,
                                lambda problem_local, m0_local, m_k_arr, N_k_arr, n_local=n, m_local=m, bd_local=bd, d_local=d, h_local=h, NMK_local=NMK, diff_R_2n_val=diff_R_2n_val:
                                    - I_mk(n_local, m_local, bd_local, d_local, m0_local, h_local, NMK_local, m_k_arr, N_k_arr) * diff_R_2n_val
                            )
                        current_row = row_offset + m
                        current_col = col_offset + 2*N_left + m
                        cache.add_m0_dependent_A_entry(current_row, current_col,
                            lambda problem_local, m0_local, m_k_arr, N_k_arr, m_local=m, bd_local=bd, h_local=h, NMK_local=NMK, a_local=a_filtered, a_bd_local=a_filtered[bd]:
                                h_local * diff_Lambda_k(m_local, a_bd_local, m0_local, NMK_local, h_local, a_local, m_k_arr, N_k_arr)
                        )
                    row_offset += M_right
            elif bd == 0:
                left_diag_is_active = d[bd] < d[bd + 1]
                if left_diag_is_active:
                    for n in range(N_left):
                        A_template[row_offset + n][col_offset + n] = - (h - d[bd]) * diff_R_1n(n, a_filtered[bd], bd, h, d, a_filtered)
                        for m in range(M_right):
                            A_template[row_offset + n][col_offset + N_left + m] = I_nm(n, m, bd, d, h) * diff_R_1n(m, a_filtered[bd], bd + 1, h, d, a_filtered)
                            A_template[row_offset + n][col_offset + N_left + M_right + m] = I_nm(n, m, bd, d, h) * diff_R_2n(m, a_filtered[bd], bd + 1, h, d, a_filtered)
                    row_offset += N_left
                else:
                    for m in range(M_right):
                        for n in range(N_left):
                            A_template[row_offset + m][col_offset + n] = - I_nm(n, m, bd, d, h) * diff_R_1n(n, a_filtered[bd], bd, h, d, a_filtered)
                        A_template[row_offset + m][col_offset + N_left + m] = (h - d[bd + 1]) * diff_R_1n(m, a_filtered[bd], bd + 1, h, d, a_filtered)
                        A_template[row_offset + m][col_offset + N_left + M_right + m] = (h - d[bd + 1]) * diff_R_2n(m, a_filtered[bd], bd + 1, h, d, a_filtered)
                    row_offset += M_right
                col_offset += N_left
            else: # i-i boundary
                left_diag_is_active = d[bd] < d[bd + 1]
                if left_diag_is_active:
                    for n in range(N_left):
                        A_template[row_offset + n][col_offset + n] = - (h - d[bd]) * diff_R_1n(n, a_filtered[bd], bd, h, d, a_filtered)
                        A_template[row_offset + n][col_offset + N_left + n] = - (h - d[bd]) * diff_R_2n(n, a_filtered[bd], bd, h, d, a_filtered)
                        for m in range(M_right):
                            A_template[row_offset + n][col_offset + 2*N_left + m] = I_nm(n, m, bd, d, h) * diff_R_1n(m, a_filtered[bd], bd + 1, h, d, a_filtered)
                            A_template[row_offset + n][col_offset + 2*N_left + M_right + m] = I_nm(n, m, bd, d, h) * diff_R_2n(m, a_filtered[bd], bd + 1, h, d, a_filtered)
                    row_offset += N_left
                else:
                    for m in range(M_right):
                        for n in range(N_left):
                            A_template[row_offset + m][col_offset + n] = - I_nm(n, m, bd, d, h) * diff_R_1n(n, a_filtered[bd], bd, h, d, a_filtered)
                            A_template[row_offset + m][col_offset + N_left + n] = - I_nm(n, m, bd, d, h) * diff_R_2n(n, a_filtered[bd], bd, a_filtered, h, d)
                        A_template[row_offset + m][col_offset + 2*N_left + m] = (h - d[bd + 1]) * diff_R_1n(m, a_filtered[bd], bd + 1, h, d, a_filtered)
                        A_template[row_offset + m][col_offset + 2*N_left + M_right + m] = (h - d[bd + 1]) * diff_R_2n(m, a_filtered[bd], bd + 1, a_filtered, h, d)
                    row_offset += M_right
                col_offset += 2 * N_left

        cache.set_A_template(A_template)

        # --- b Vector Template ---
        b_template = np.zeros(size, dtype=complex)
        
        heaving = [domain_list[idx].heaving for idx in domain_keys]
        index = 0

        # Potential matching (m0-independent)
        for boundary in range(boundary_count):
            if boundary == (boundary_count - 1): # i-e boundary
                for n in range(NMK[boundary]):
                    b_template[index] = b_potential_end_entry(n, boundary, heaving, h, d, a_filtered)
                    index += 1
            else: # i-i boundary
                i = boundary
                if d[i] > d[i + 1]:
                    N = NMK[i]
                else:
                    N = NMK[i+1]
                for n in range(N):
                    b_template[index] = b_potential_entry(n, boundary, d, heaving, h, a_filtered)
                    index += 1

        # Velocity matching (m0-dependent via b_velocity_end_entry)
        for boundary in range(boundary_count):
            if boundary == (boundary_count - 1): # i-e boundary
                for k in range(NMK[-1]):
                    current_row = index + k
                    cache.add_m0_dependent_b_entry(current_row,
                        # Pass m_k_arr, N_k_arr
                        lambda problem_local, m0_local, m_k_arr, N_k_arr, k_local=k, boundary_local=boundary, heaving_local=heaving, a_local=a_filtered, h_local=h, d_local=d, NMK_local=NMK:
                            b_velocity_end_entry(k_local, boundary_local, heaving_local, a_local, h_local, d_local, m0_local, NMK_local, m_k_arr, N_k_arr)
                    )
                index += NMK[-1]
            else: # i-i boundary (m0-independent)
                if d[boundary] < d[boundary + 1]:
                    N = NMK[boundary + 1]
                else:
                    N = NMK[boundary]
                for n in range(N):
                    b_template[index] = b_velocity_entry(n, boundary, heaving, a_filtered, h, d)
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
            lambda d_i, r, z: self.phi_p_i_func(d_i, r, z),
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
    
    def run_and_store_results(self, problem_index: int, m0_values: np.ndarray) -> 'Results': 
        """
        Perform the full MEEM computation for a *list of frequencies* and store results
        in the Results class.
        This method will now benefit from the optimized assemble_A_multi/assemble_b_multi.

        :param problem_index: Index of the MEEMProblem instance to process.
        :param m0_values: A NumPy array of angular frequencies (m0) to process.
                        These are the specific frequencies for which calculations will be performed.
        :return: Results object containing the computed data including hydrodynamic coefficients
                and optionally potentials.
        """
        problem = self.problem_list[problem_index]
        
        # Initialize a list to store batched potentials
        # This will be formatted as required by results.store_all_potentials
        all_potentials_batch_data = []

        # Initialize Results object (will be populated with data in the loop)
        # The Results object takes the full arrays of frequencies and modes defined in the problem.
        geometry = problem.geometry
        results = Results(geometry, problem.frequencies, problem.modes)

        num_modes = len(problem.modes) # Get the number of modes from the problem
        
        # Create mappings from the input m0_values to their indices in problem.frequencies
        freq_to_idx = {freq: idx for idx, freq in enumerate(problem.frequencies)}
        
        # Initialize full matrices with NaNs, for the dimensions of the Results object
        full_added_mass_matrix = np.full((len(problem.frequencies), num_modes), np.nan, dtype=float)
        full_damping_matrix = np.full((len(problem.frequencies), num_modes), np.nan, dtype=float)

        # Iterate over each frequency for computation from the input m0_values
        for i, m0 in enumerate(m0_values): # Loop through the input m0_values to calculate
            print(f"  Calculating for m0 = {m0:.4f} rad/s")

            # Get the index of this m0 in the problem's full frequencies array
            freq_idx_in_problem = freq_to_idx.get(m0)
            if freq_idx_in_problem is None:
                # This should ideally not happen if m0_values are a subset of problem.frequencies
                print(f"  Warning: m0={m0:.4f} not found in problem.frequencies. Skipping calculation.")
                continue
            
            A = self.assemble_A_multi(problem, m0)
            b = self.assemble_b_multi(problem, m0)

            try:
                X = np.linalg.solve(A, b)
            except np.linalg.LinAlgError as e:
                print(f"  ERROR: Could not solve for m0={m0:.4f}: {e}. Storing NaN for coefficients.")
                # The `full_added_mass_matrix` and `full_damping_matrix` are already initialized with NaNs,
                # so no need to explicitly append NaNs here if X is not solved.
                continue # Skip to the next frequency

            # Compute hydrodynamic coefficients for this frequency
            hydro_coeffs = self.compute_hydrodynamic_coefficients(problem, X)
            
            # Ensure that hydro_coeffs['real'] and hydro_coeffs['imag'] are arrays of shape (num_modes,)
            # If compute_hydrodynamic_coefficients returns a scalar for a single mode case,
            # wrap it in an array here.
            current_added_mass = np.atleast_1d(hydro_coeffs['real'])
            current_damping = np.atleast_1d(hydro_coeffs['imag'])

            # Sanity check: Ensure the number of modes returned matches expected
            if current_added_mass.shape[0] != num_modes or current_damping.shape[0] != num_modes:
                raise ValueError(f"compute_hydrodynamic_coefficients returned {current_added_mass.shape[0]} modes "
                                f"for m0={m0:.4f}, but problem expects {num_modes} modes.")
            
            # Store results into the pre-allocated full matrices at the correct frequency index
            full_added_mass_matrix[freq_idx_in_problem, :] = current_added_mass
            full_damping_matrix[freq_idx_in_problem, :] = current_damping

            # --- Handle Potentials  ---
            calculated_potentials_for_this_freq_mode = {}
            for domain_name, domain in problem.geometry.domain_list.items():
                print(domain_name, domain.category)
                domain_name = domain.category # Or a more specific name like f'domain_{domain_idx}'
                num_harmonics_for_domain = domain.number_harmonics # Get this from the domain
                
                # Dummy potential values and coordinates
                dummy_potentials = (np.random.rand(num_harmonics_for_domain) + \
                                    1j * np.random.rand(num_harmonics_for_domain)).astype(complex)
                dummy_r_coords_dict = {f'r_h{k}': np.random.rand() for k in range(num_harmonics_for_domain)}
                dummy_z_coords_dict = {f'z_h{k}': np.random.rand() for k in range(num_harmonics_for_domain)}
                
                calculated_potentials_for_this_freq_mode[domain_name] = {
                    'potentials': dummy_potentials,
                    'r_coords_dict': dummy_r_coords_dict,
                    'z_coords_dict': dummy_z_coords_dict,
                }

            for mode_idx, mode_value in enumerate(problem.modes):
                
                # This is a placeholder for the actual calculation of potentials PER MODE
                current_mode_potentials = {}
                for domain_idx, domain in problem.geometry.domain_list.items():
                    domain_name = domain.category
                    num_harmonics_for_domain = domain.number_harmonics

                    dummy_potentials = (np.random.rand(num_harmonics_for_domain) + \
                                        1j * np.random.rand(num_harmonics_for_domain)).astype(complex)
                    dummy_r_coords_dict = {f'r_h{k}': np.random.rand() for k in range(num_harmonics_for_domain)}
                    dummy_z_coords_dict = {f'z_h{k}': np.random.rand() for k in range(num_harmonics_for_domain)}
                    
                    current_mode_potentials[domain_name] = {
                        'potentials': dummy_potentials,
                        'r_coords_dict': dummy_r_coords_dict,
                        'z_coords_dict': dummy_z_coords_dict,
                    }

                all_potentials_batch_data.append({
                    'frequency_idx': freq_idx_in_problem, # Use the index in problem.frequencies
                    'mode_idx': mode_idx,               # Use the index in problem.modes
                    'data': current_mode_potentials     # The domain-specific potentials
                })

        # After the loop, store the collected hydrodynamic coefficients
        results.store_hydrodynamic_coefficients(
            frequencies=problem.frequencies, # Pass the full problem frequencies
            modes=problem.modes,             # Pass the full problem modes
            added_mass_matrix=full_added_mass_matrix,
            damping_matrix=full_damping_matrix
        )
        
        # Store all potentials
        if all_potentials_batch_data:
            results.store_all_potentials(all_potentials_batch_data)

        return results
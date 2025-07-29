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
from functools import partial

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
            self.cache_list[problem] = self.build_problem_cache(problem)
    
    def _ensure_m_k_and_N_k_arrays(self, problem: 'MEEMProblem', m0):
        """
        Ensure that m_k_arr and N_k_arr are computed and cached for the given problem and m0.
        """
        cache = self.cache_list[problem]

        if cache.m_k_arr is None or cache.N_k_arr is None:
            domain_list = problem.domain_list
            domain_keys = list(domain_list.keys())
            NMK = [domain_list[idx].number_harmonics for idx in domain_keys]
            h = domain_list[0].h
            m_k_arr = np.array([cache.m_k_entry_func(k, m0, h) for k in range(NMK[-1])])
            N_k_arr = np.array([cache.N_k_func(k, m0, h, m_k_arr) for k in range(NMK[-1])])
            cache.set_precomputed_m_k_N_k(m_k_arr, N_k_arr)
    
    def assemble_A_multi(self, problem: 'MEEMProblem', m0) -> np.ndarray:
        """
        Assemble the system matrix A for a given problem using pre-computed blocks.
        """
        cache = self.cache_list[problem]
        A = cache.get_A_template()

        I_mk_vals = cache.get_closure("I_mk_vals")(m0, cache.m_k_arr, cache.N_k_arr)

        for row, col, calc_func in cache.m0_dependent_A_indices:
            A[row, col] = calc_func(problem, m0, cache.m_k_arr, cache.N_k_arr, I_mk_vals)

        return A
                
    # Now, the optimized assemble_b_multi method that uses the cache
    def assemble_b_multi(self, problem: 'MEEMProblem', m0) -> np.ndarray:
        """
        Assemble the right-hand side vector b for a given problem.
        """
        cache = self.cache_list[problem]
        b = cache.get_b_template()

        I_mk_vals = cache.get_closure("I_mk_vals")(m0, cache.m_k_arr, cache.N_k_arr)

        for row, calc_func in cache.m0_dependent_b_indices:
            b[row] = calc_func(problem, m0, cache.m_k_arr, cache.N_k_arr, I_mk_vals)

        return b

    def build_problem_cache(self, problem: 'MEEMProblem') -> ProblemCache:
        """
        Analyzes the problem and pre-computes m0-independent parts of A and b,
        and identifies indices for m0-dependent parts, storing them in a cache.
        """
        # 1. Initialization and Parameter Extraction
        cache = ProblemCache(problem)
        domain_list = problem.domain_list
        domain_keys = list(domain_list.keys())
        
        h = domain_list[0].h
        d = [domain_list[idx].di for idx in domain_keys]
        a = [domain_list[idx].a for idx in domain_keys]
        NMK = [domain.number_harmonics for domain in domain_list.values()]
        heaving = [domain_list[idx].heaving for idx in domain_keys]
        
        boundary_count = len(NMK) - 1
        size = NMK[0] + NMK[-1] + 2 * sum(NMK[1:len(NMK) - 1])

        A_template = np.zeros((size, size), dtype=complex)
        b_template = np.zeros(size, dtype=complex)

        # 2. Pre-compute m0-INDEPENDENT values
        I_nm_vals_precomputed = np.zeros((max(NMK), max(NMK), boundary_count - 1), dtype=complex)
        for bd in range(boundary_count - 1):
            for n in range(NMK[bd]):
                for m in range(NMK[bd + 1]):
                    I_nm_vals_precomputed[n, m, bd] = I_nm(n, m, bd, d, h)
        cache.set_I_nm_vals(I_nm_vals_precomputed)

        R_1n_func = np.vectorize(partial(R_1n, h=h, d=d, a=a))
        R_2n_func = np.vectorize(partial(R_2n, a=a, h=h, d=d))
        diff_R_1n_func = np.vectorize(partial(diff_R_1n, h=h, d=d, a=a), otypes=[complex])
        diff_R_2n_func = np.vectorize(partial(diff_R_2n, h=h, d=d, a=a), otypes=[complex])

        # 3. Define m0-DEPENDENT calculation closures
        def _calculate_I_mk_vals(m0, m_k_arr, N_k_arr):
            vals = np.zeros((NMK[boundary_count - 1], NMK[boundary_count]), dtype=complex)
            for m in range(NMK[boundary_count - 1]):
                for k in range(NMK[boundary_count]):
                    # FIX: Consistent argument order (h, d)
                    vals[m, k] = I_mk(m, k, boundary_count - 1, d, m0, h, m_k_arr, N_k_arr)
            return vals
        
        cache.set_closure("I_mk_vals", _calculate_I_mk_vals)
        cache.set_m_k_and_N_k_funcs(m_k_entry, N_k_multi)

        # 4. Assemble A_template and identify m0-dependent indices
        
        ## --- Potential Matching Blocks ---
        col_offset = 0
        row_offset = 0
        for bd in range(boundary_count):
            N = NMK[bd]
            M = NMK[bd + 1]

            if bd == (boundary_count - 1): # Final i-e boundary
                row_height = N
                left_block1 = p_diagonal_block(True, R_1n_func, bd, h, d, a, NMK)
                
                if bd > 0:
                    left_block2 = p_diagonal_block(True, R_2n_func, bd, h, d, a, NMK)
                    # FIX: Correct order is R1n block, then R2n block
                    block = np.concatenate([left_block1, left_block2], axis=1)
                else:
                    block = left_block1
                
                A_template[row_offset : row_offset + row_height, col_offset : col_offset + block.shape[1]] = block
                
                p_dense_e_col_start = col_offset + block.shape[1]
                for m_local in range(N):
                    for k_local in range(M):
                        g_row, g_col = row_offset + m_local, p_dense_e_col_start + k_local
                        calc_func = lambda p, m0, mk, Nk, Imk, m=m_local, k=k_local: \
                            p_dense_block_e_entry(m, k, bd, Imk, NMK, a, m0, h, mk, Nk)
                        cache.add_m0_dependent_A_entry(g_row, g_col, calc_func)
                col_offset += block.shape[1]

            else: # Internal i-i boundaries
                left_diag = d[bd] > d[bd+1]
                row_height = N if left_diag else M
                blocks = []
                
                # FIX: Correctly ordered R1n/R2n blocks
                if left_diag:
                    blocks.append(p_diagonal_block(True, R_1n_func, bd, h, d, a, NMK))
                    if bd > 0: blocks.append(p_diagonal_block(True, R_2n_func, bd, h, d, a, NMK))
                    blocks.append(p_dense_block(False, R_1n_func, bd, NMK, a, I_nm_vals_precomputed))
                    blocks.append(p_dense_block(False, R_2n_func, bd, NMK, a, I_nm_vals_precomputed))
                else: # right_diag
                    blocks.append(p_dense_block(True, R_1n_func, bd, NMK, a, I_nm_vals_precomputed))
                    if bd > 0: blocks.append(p_dense_block(True, R_2n_func, bd, NMK, a, I_nm_vals_precomputed))
                    blocks.append(p_diagonal_block(False, R_1n_func, bd, h, d, a, NMK))
                    blocks.append(p_diagonal_block(False, R_2n_func, bd, h, d, a, NMK))

                full_block = np.concatenate(blocks, axis=1)
                A_template[row_offset : row_offset + row_height, col_offset : col_offset + full_block.shape[1]] = full_block
                col_offset += 2*N if bd > 0 else N
            
            row_offset += row_height

        ## --- Velocity Matching Blocks ---
        col_offset = 0
        for bd in range(boundary_count):
            N = NMK[bd]
            M = NMK[bd + 1]

            if bd == (boundary_count - 1): # Final i-e boundary
                row_height = M
                v_dense_e_col_start = col_offset
                for m_local in range(M):
                    # First dense block (R1n)
                    for k_local in range(N):
                        g_row, g_col = row_offset + m_local, v_dense_e_col_start + k_local
                        calc_func = lambda p, m0, mk, Nk, Imk, m=m_local, k=k_local: \
                            v_dense_block_e_entry(m, k, bd, Imk, a, h, d)
                        cache.add_m0_dependent_A_entry(g_row, g_col, calc_func)
                    # Second dense block (R2n) if needed
                    if bd > 0:
                        # FIX: Correctly place the R2n block after the R1n block
                        r2n_col_start = v_dense_e_col_start + N
                        for k_local in range(N):
                            g_row, g_col = row_offset + m_local, r2n_col_start + k_local
                            calc_func = lambda p, m0, mk, Nk, Imk, m=m_local, k=k_local: \
                                v_dense_block_e_entry_R2(m, k, bd, Imk, a, h, d)
                            cache.add_m0_dependent_A_entry(g_row, g_col, calc_func)
                
                v_diag_e_col_start = col_offset + (2*N if bd > 0 else N)
                for k_local in range(M):
                    g_row, g_col = row_offset + k_local, v_diag_e_col_start + k_local
                    # FIX: Removed unused 'm' argument from lambda signature
                    calc_func = lambda p, m0, mk, Nk, Imk, k=k_local: \
                        v_diagonal_block_e_entry(m, k, bd, m0, mk, a, h)
                    cache.add_m0_dependent_A_entry(g_row, g_col, calc_func)
                col_offset += (2*N if bd > 0 else N)

            else: # Internal i-i boundaries
                left_diag = d[bd] <= d[bd+1]
                row_height = N if left_diag else M
                blocks = []
                
                # FIX: Correctly ordered diff_R1n/diff_R2n blocks
                if left_diag:
                    blocks.append(v_diagonal_block(True, diff_R_1n_func, bd, h, d, NMK, a))
                    if bd > 0: blocks.append(v_diagonal_block(True, diff_R_2n_func, bd, h, d, NMK, a))
                    blocks.append(v_dense_block(False, diff_R_1n_func, bd, I_nm_vals_precomputed, NMK, a))
                    blocks.append(v_dense_block(False, diff_R_2n_func, bd, I_nm_vals_precomputed, NMK, a))
                else: # right_diag
                    blocks.append(v_dense_block(True, diff_R_1n_func, bd, I_nm_vals_precomputed, NMK, a))
                    if bd > 0: blocks.append(v_dense_block(True, diff_R_2n_func, bd, I_nm_vals_precomputed, NMK, a))
                    blocks.append(v_diagonal_block(False, diff_R_1n_func, bd, h, d, NMK, a))
                    blocks.append(v_diagonal_block(False, diff_R_2n_func, bd, h, d, NMK, a))

                full_block = np.concatenate(blocks, axis=1)
                A_template[row_offset : row_offset + row_height, col_offset : col_offset + full_block.shape[1]] = full_block
                col_offset += 2*N if bd > 0 else N
            
            row_offset += row_height

        # 5. Assemble b_template and identify m0-dependent indices
        index = 0
        # Potential matching entries
        for bd in range(boundary_count):
            if bd == (boundary_count - 1):
                for n in range(NMK[-2]):
                    b_template[index] = b_potential_end_entry(n, bd, heaving, h, d, a)
                    index += 1
            else:
                num_entries = NMK[bd] if d[bd] > d[bd+1] else NMK[bd+1]
                for n in range(num_entries):
                    b_template[index] = b_potential_entry(n, bd, d, heaving, h, a)
                    index += 1

        # Velocity matching entries
        for bd in range(boundary_count):
            if bd == (boundary_count - 1):
                for n_local in range(NMK[-1]):
                    calc_func = lambda p, m0, mk, Nk, Imk, n=n_local: \
                        b_velocity_end_entry(n, bd, heaving, a, h, d, m0, NMK, mk, Nk)
                    cache.add_m0_dependent_b_entry(index, calc_func)
                    index += 1
            else:
                num_entries = NMK[bd] if d[bd] <= d[bd+1] else NMK[bd+1]
                for n in range(num_entries):
                    b_template[index] = b_velocity_entry(n, bd, heaving, a, h, d)
                    index += 1
                    
        # 6. Finalize and return cache
        cache.set_A_template(A_template)
        cache.set_b_template(b_template)
        
        return cache
    
    def solve_linear_system_multi(self, problem: MEEMProblem, m0) -> np.ndarray:
        """
        Solve the linear system A x = b for the given problem (multi-region, optimized).
        """
        cache = self.cache_list[problem]

        # ✅ Ensure m₀-dependent arrays are evaluated
        self._ensure_m_k_and_N_k_arrays(problem, m0)

        A = self.assemble_A_multi(problem, m0) # Now calls the optimized A assembly
        # Debug prints for zero rows and m0-dependent entries

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

    def compute_hydrodynamic_coefficients(self, problem, X, m0):
        """
        Compute the hydrodynamic coefficients (added mass and damping) for a given problem and solution vector X.
        """
        geometry = problem.geometry
        domain_keys = list(geometry.domain_list.keys())
        a = [geometry.domain_list[idx].a for idx in domain_keys]
        d = [
            domain.di[0] if isinstance(domain.di, list) else domain.di
            for domain in geometry.domain_list.values()
        ]
        h = geometry.domain_list[0].h
        NMK = [geometry.domain_list[idx].number_harmonics for idx in domain_keys]
        heaving = [geometry.domain_list[idx].heaving for idx in domain_keys]
        boundary_count = len(NMK) - 1

        size = NMK[0] + NMK[-1] + 2 * sum(NMK[1:len(NMK) - 1])
        c_vector = np.zeros((size - NMK[-1]), dtype=complex)
        # === Assemble c_vector ===``
        col = 0
        for n in range(NMK[0]):
            c_vector[n] = heaving[0] * int_R_1n(0, n, a, h, d)* z_n_d(n)
        col += NMK[0]

        for i in range(1, boundary_count):
            M = NMK[i]
            for m in range(M):
                c_vector[col + m] = heaving[i] * int_R_1n(i, m, a, h, d) * z_n_d(m)
                c_vector[col + M + m] = heaving[i] * int_R_2n(i, m, a, h, d) * z_n_d(m)
            col += 2 * M
        # === Assemble hydro_p_terms ===
        hydro_p_term_sum = np.zeros(boundary_count, dtype=complex)
        for i in range(boundary_count):
            hydro_p_term_sum[i] = heaving[i] * int_phi_p_i(i, h, d, a)

        # === Compute unscaled hydro coefficient ===
        hydro_coef = 2 * pi * (np.dot(c_vector, X[:-NMK[-1]]) + sum(hydro_p_term_sum))

        # === Convert to physical units ===
        hydro_coef_real = hydro_coef.real * h**3 * rho
        if m0 == np.inf: hydro_coef_imag = 0
        else: hydro_coef_imag = hydro_coef.imag * omega(m0,h,g) * h**3 * rho

        # === Nondimensional coefficients ===
        max_rad = a[0]
        for i in range(boundary_count - 1, 0, -1):
            if heaving[i]:
                max_rad = a[i]
                break

        hydro_coef_nondim = h**3/(max_rad**3 * pi)*hydro_coef
        # === Phase and excitation force ===
      
        return {
            "real": hydro_coef_real,
            "imag": hydro_coef_imag,
            "nondim_real": hydro_coef_nondim.real,
            "nondim_imag": hydro_coef_nondim.imag,
            "excitation_phase": excitation_phase(X, NMK, m0, a),
            "excitation_force": excitation_force(hydro_coef_imag, m0, h)
        }
    
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
        return Cs[-1][k] * Lambda_k(k, r, m0, a, m_k_arr) * Z_k_e(k, z, m0, h, NMK, m_k_arr)
    
    def calculate_potentials(self, problem, solution_vector: np.ndarray, m0, spatial_res, sharp) -> Dict[str, Any]:
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
        # Ensure m_k_arr and N_k_arr are computed and retrieved from the cache
        # =================== DEBUGGING START ===================
        print("\n--- Debugging: Entering calculate_potentials ---")
        print(f"  Received m0={m0}, spatial_res={spatial_res}, sharp={sharp}")
        # =======================================================
        self._ensure_m_k_and_N_k_arrays(problem, m0)
        cache = self.cache_list[problem]
        m_k_arr = cache.m_k_arr
        N_k_arr = cache.N_k_arr
        # =================== DEBUGGING ===================
        print(f"  Cache check: m_k_arr shape = {m_k_arr.shape}, N_k_arr shape = {N_k_arr.shape}")
        if m_k_arr is not None:
            print(f"  m_k_arr head: {m_k_arr[:3]}")
        # =================================================
        
        domain_list = problem.domain_list
        domain_keys = list(domain_list.keys())
        boundary_count = len(domain_keys) - 1

        NMK = [domain_list[idx].number_harmonics for idx in domain_keys]
        h = domain_list[0].h
        d = [domain_list[idx].di for idx in domain_keys if domain_list[idx].di is not None]
        print("d in new calculate_potentials:", d)
        a = [domain_list[idx].a for idx in domain_keys if domain_list[idx].a is not None]
        heaving = [domain_list[idx].heaving for idx in domain_keys]
        
        print("  Reformatting solution coefficients once...")
        Cs = self.reformat_coeffs(solution_vector, NMK, boundary_count)

        # --- 2. Create Meshgrid and Regions ---
        R, Z = make_R_Z(a, h, d, sharp, spatial_res)
        regions = []
        regions.append((R <= a[0]) & (Z < -d[0]))
        for i in range(1, boundary_count):
            regions.append((R > a[i-1]) & (R <= a[i]) & (Z < -d[i]))
        regions.append(R > a[-1])
        # =================== DEBUGGING ===================
        print(f"  Meshgrid created: R shape={R.shape}, Z shape={Z.shape}")
        print(f"    R range: [{np.min(R):.2f}, {np.max(R):.2f}], Z range: [{np.min(Z):.2f}, {np.max(Z):.2f}]")
        # =================================================

        # =================== DEBUGGING ===================
        print("  Region mask point counts:")
        for i, region_mask in enumerate(regions):
            print(f"    Region {i}: {np.sum(region_mask)} points")
        # =================================================

        # Initialize potential arrays
        phi = np.full_like(R, np.nan + np.nan*1j, dtype=complex) 
        phiH = np.full_like(R, np.nan + np.nan*1j, dtype=complex) 
        phiP = np.full_like(R, np.nan + np.nan*1j, dtype=complex) 

        # --- 3. Vectorized Calculation of Potentials ---
        print("  Calculating potentials with vectorized operations...")

        # Region 0 (Inner)
        if np.any(regions[0]):
            r_vals, z_vals = R[regions[0]], Z[regions[0]]
            n_vals = np.arange(NMK[0])
            R1n_vals = R_1n_vectorized(n_vals[:, None], r_vals[None, :], 0, h, d, a)
            Zn_vals = Z_n_i_vectorized(n_vals[:, None], z_vals[None, :], 0, h, d)
            phiH[regions[0]] = np.sum(Cs[0][:, None] * R1n_vals * Zn_vals, axis=0)
        print("    Done with Region 0.")

        # Intermediate Regions
        for i in range(1, boundary_count):
            if np.any(regions[i]):
                r_vals, z_vals = R[regions[i]], Z[regions[i]]
                m_vals = np.arange(NMK[i])
                R1n_vals = R_1n_vectorized(m_vals[:, None], r_vals[None, :], i, h, d, a)
                R2n_vals = R_2n_vectorized(m_vals[:, None], r_vals[None, :], i, a, h, d)
                Zm_vals = Z_n_i_vectorized(m_vals[:, None], z_vals[None, :], i, h, d)
                term1 = Cs[i][:NMK[i], None] * R1n_vals
                term2 = Cs[i][NMK[i]:, None] * R2n_vals
                phiH[regions[i]] = np.sum((term1 + term2) * Zm_vals, axis=0)
            print(f"    Done with Region {i}.")

        # Exterior Region
        if np.any(regions[-1]):
            r_vals, z_vals = R[regions[-1]], Z[regions[-1]]
            k_vals = np.arange(NMK[-1])
            Lambda_vals = Lambda_k_vectorized(k_vals[:, None], r_vals[None, :], m0, a, m_k_arr)
            Zk_vals = Z_k_e_vectorized(k_vals[:, None], z_vals[None, :], m0, h, m_k_arr, N_k_arr)
            phiH[regions[-1]] = np.sum(Cs[-1][:, None] * Lambda_vals * Zk_vals, axis=0)
        print("    Done with Exterior Region.")
        
        # --- 4. Calculate Particular Potential (phiP) ---
        print("  Calculating Particular Potential (phiP)...")
        phiP[regions[0]] = heaving[0] * phi_p_i(d[0], R[regions[0]], Z[regions[0]], h)
        for i in range(1, boundary_count):
            phiP[regions[i]] = heaving[i] * phi_p_i(d[i], R[regions[i]], Z[regions[i]], h)
        phiP[regions[-1]] = 0
        print("    Done with phiP calculation.")

        # Sum to get total potential phi
        phi = phiH + phiP
        print("--- Exiting Optimized calculate_potentials ---\n")

        return {"R": R, "Z": Z, "phiH": phiH, "phiP": phiP, "phi": phi}

    def visualize_potential(self, field, R, Z, title):
        plt.figure(figsize=(8, 6))
        plt.contourf(R, Z, field, levels=50, cmap='viridis')
        plt.colorbar()
        plt.title(title)
        plt.xlabel('Radial Distance (R)')
        plt.ylabel('Axial Distance (Z)')
        plt.show()
    
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

# package/src/openflash/meem_engine.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
from openflash.meem_problem import MEEMProblem
from openflash.problem_cache import ProblemCache
from openflash.multi_equations import *
from openflash.results import Results
from scipy import linalg
from openflash.multi_constants import rho as default_rho, g
from functools import partial
from openflash.body import SteppedBody
from openflash.geometry import ConcentricBodyGroup
from openflash.basic_region_geometry import BasicRegionGeometry

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
        self.cache_list = {} 

        for problem in problem_list:
            self.cache_list[problem] = self.build_problem_cache(problem)
    
    def _ensure_m_k_and_N_k_arrays(self, problem: 'MEEMProblem', m0):
        cache = self.cache_list[problem]
        if cache.m_k_arr is None or cache.cached_m0 != m0:
            domain_list = problem.domain_list
            domain_keys = list(domain_list.keys())
            NMK = [domain_list[idx].number_harmonics for idx in domain_keys]
            h = domain_list[0].h
            m_k_arr = np.array([cache.m_k_entry_func(k, m0, h) for k in range(NMK[-1])])
            N_k_arr = np.array([cache.N_k_func(k, m0, h, m_k_arr) for k in range(NMK[-1])])
            cache._set_precomputed_m_k_N_k(m_k_arr, N_k_arr, m0)
    
    def assemble_A_multi(self, problem: 'MEEMProblem', m0) -> np.ndarray:
        self._ensure_m_k_and_N_k_arrays(problem, m0)
        cache = self.cache_list[problem]
        A = cache._get_A_template()
        I_mk_vals = cache._get_closure("I_mk_vals")(m0, cache.m_k_arr, cache.N_k_arr)
        for row, col, calc_func in cache.m0_dependent_A_indices:
            A[row, col] = calc_func(problem, m0, cache.m_k_arr, cache.N_k_arr, I_mk_vals)
        return A
                
    def assemble_b_multi(self, problem: 'MEEMProblem', m0) -> np.ndarray:
        self._ensure_m_k_and_N_k_arrays(problem, m0)
        cache = self.cache_list[problem]
        b = cache._get_b_template()
        I_mk_vals = cache._get_closure("I_mk_vals")(m0, cache.m_k_arr, cache.N_k_arr)
        for row, calc_func in cache.m0_dependent_b_indices:
            b[row] = calc_func(problem, m0, cache.m_k_arr, cache.N_k_arr, I_mk_vals)
        return b

    def build_problem_cache(self, problem: 'MEEMProblem') -> ProblemCache:
        """
        Analyzes the problem and pre-computes m0-independent parts of A and b,
        and identifies indices for m0-dependent parts, storing them in a cache.
        """
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

        # Pre-compute m0-INDEPENDENT values
        I_nm_vals_precomputed = np.zeros((max(NMK), max(NMK), boundary_count - 1), dtype=complex)
        for bd in range(boundary_count - 1):
            for n in range(NMK[bd]):
                for m in range(NMK[bd + 1]):
                    I_nm_vals_precomputed[n, m, bd] = I_nm(n, m, bd, d, h)
        cache._set_I_nm_vals(I_nm_vals_precomputed)

        R_1n_func = np.vectorize(partial(R_1n, h=h, d=d, a=a))
        R_2n_func = np.vectorize(partial(R_2n, a=a, h=h, d=d))
        diff_R_1n_func = np.vectorize(partial(diff_R_1n, h=h, d=d, a=a), otypes=[complex])
        diff_R_2n_func = np.vectorize(partial(diff_R_2n, h=h, d=d, a=a), otypes=[complex])

        def _calculate_I_mk_vals(m0, m_k_arr, N_k_arr):
            vals = np.zeros((NMK[boundary_count - 1], NMK[boundary_count]), dtype=complex)
            for m in range(NMK[boundary_count - 1]):
                for k in range(NMK[boundary_count]):
                    vals[m, k] = I_mk(m, k, boundary_count - 1, d, m0, h, m_k_arr, N_k_arr)
            return vals
        
        cache._set_closure("I_mk_vals", _calculate_I_mk_vals)
        cache._set_m_k_and_N_k_funcs(m_k_entry, N_k_multi)

        ## --- Potential Matching Blocks ---
        col_offset = 0
        row_offset = 0
        for bd in range(boundary_count):
            N = NMK[bd]
            M = NMK[bd + 1]

            if bd == (boundary_count - 1): 
                row_height = N
                left_block1 = p_diagonal_block(True, R_1n_func, bd, h, d, a, NMK)
                
                if bd > 0:
                    left_block2 = p_diagonal_block(True, R_2n_func, bd, h, d, a, NMK)
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
                        cache._add_m0_dependent_A_entry(g_row, g_col, calc_func)
                col_offset += block.shape[1]

            else: 
                # Potential Match: Project onto SHORTER region (Standard MEEM)
                # d is depth from surface. Larger d means deeper bottom, so SMALLER height.
                # If d[bd] > d[bd+1], Region Left is shorter. Project on Left.
                project_on_left = d[bd] > d[bd+1]
                row_height = N if project_on_left else M
                blocks = []
                
                if project_on_left:
                    blocks.append(p_diagonal_block(True, R_1n_func, bd, h, d, a, NMK))
                    if bd > 0: blocks.append(p_diagonal_block(True, R_2n_func, bd, h, d, a, NMK))
                    blocks.append(p_dense_block(False, R_1n_func, bd, NMK, a, I_nm_vals_precomputed))
                    blocks.append(p_dense_block(False, R_2n_func, bd, NMK, a, I_nm_vals_precomputed))
                else: 
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
                    for k_local in range(N):
                        g_row, g_col = row_offset + m_local, v_dense_e_col_start + k_local
                        calc_func = lambda p, m0, mk, Nk, Imk, m=m_local, k=k_local: \
                            v_dense_block_e_entry(m, k, bd, Imk, a, h, d)
                        cache._add_m0_dependent_A_entry(g_row, g_col, calc_func)
                    if bd > 0:
                        r2n_col_start = v_dense_e_col_start + N
                        for k_local in range(N):
                            g_row, g_col = row_offset + m_local, r2n_col_start + k_local
                            calc_func = lambda p, m0, mk, Nk, Imk, m=m_local, k=k_local: \
                                v_dense_block_e_entry_R2(m, k, bd, Imk, a, h, d)
                            cache._add_m0_dependent_A_entry(g_row, g_col, calc_func)
                
                v_diag_e_col_start = col_offset + (2*N if bd > 0 else N)
                for k_local in range(M):
                    g_row, g_col = row_offset + k_local, v_diag_e_col_start + k_local
                    calc_func = lambda p, m0, mk, Nk, Imk, k=k_local: \
                        v_diagonal_block_e_entry(m, k, bd, m0, mk, a, h)
                    cache._add_m0_dependent_A_entry(g_row, g_col, calc_func)
                col_offset += (2*N if bd > 0 else N)

            else: # Internal i-i boundaries
                project_on_left = d[bd] <= d[bd+1]
                row_height = N if project_on_left else M
                blocks = []
                
                if project_on_left:
                    blocks.append(v_diagonal_block(True, diff_R_1n_func, bd, h, d, NMK, a))
                    if bd > 0: blocks.append(v_diagonal_block(True, diff_R_2n_func, bd, h, d, NMK, a))
                    blocks.append(v_dense_block(False, diff_R_1n_func, bd, I_nm_vals_precomputed, NMK, a))
                    blocks.append(v_dense_block(False, diff_R_2n_func, bd, I_nm_vals_precomputed, NMK, a))
                else: 
                    blocks.append(v_dense_block(True, diff_R_1n_func, bd, I_nm_vals_precomputed, NMK, a))
                    if bd > 0: blocks.append(v_dense_block(True, diff_R_2n_func, bd, I_nm_vals_precomputed, NMK, a))
                    blocks.append(v_diagonal_block(False, diff_R_1n_func, bd, h, d, NMK, a))
                    blocks.append(v_diagonal_block(False, diff_R_2n_func, bd, h, d, NMK, a))

                full_block = np.concatenate(blocks, axis=1)
                A_template[row_offset : row_offset + row_height, col_offset : col_offset + full_block.shape[1]] = full_block
                col_offset += 2*N if bd > 0 else N
            
            row_offset += row_height

        # Assemble b_template 
        index = 0
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

        for bd in range(boundary_count):
            if bd == (boundary_count - 1):
                for n_local in range(NMK[-1]):
                    calc_func = lambda p, m0, mk, Nk, Imk, n=n_local: \
                        b_velocity_end_entry(n, bd, heaving, a, h, d, m0, NMK, mk, Nk)
                    cache._add_m0_dependent_b_entry(index, calc_func)
                    index += 1
            else:
                num_entries = NMK[bd] if d[bd] <= d[bd+1] else NMK[bd+1]
                for n in range(num_entries):
                    b_template[index] = b_velocity_entry(n, bd, heaving, a, h, d)
                    index += 1
                    
        cache._set_A_template(A_template)
        cache._set_b_template(b_template)
        
        return cache
    
    def solve_linear_system_multi(self, problem: MEEMProblem, m0) -> np.ndarray:
        cache = self.cache_list[problem]
        self._ensure_m_k_and_N_k_arrays(problem, m0)
        A = self.assemble_A_multi(problem, m0)
        b = self.assemble_b_multi(problem, m0) 
        X = linalg.solve(A, b)
        return X
    
    def reformat_coeffs(self, x: np.ndarray, NMK, boundary_count) -> list[np.ndarray]:
        cs = []
        row = 0
        cs.append(x[:NMK[0]])
        row += NMK[0]
        for i in range(1, boundary_count):
            cs.append(x[row: row + NMK[i] * 2])
            row += NMK[i] * 2
        cs.append(x[row:])
        return cs

    def compute_hydrodynamic_coefficients(self, problem, X, m0, modes_to_calculate: Optional[np.ndarray] = None, rho: Optional[float] = None):
        """
        Computes the hydrodynamic coefficients (Added Mass and Damping) from the solution vector X.
        
        Args:
            rho (float, optional): Density of fluid. Defaults to value from multi_constants.
        """
        if rho is None:
            rho = default_rho

        geometry = problem.geometry
        domain_keys = list(geometry.domain_list.keys())
        a = [geometry.domain_list[idx].a for idx in domain_keys]
        d = [
            domain.di[0] if isinstance(domain.di, list) else domain.di
            for domain in geometry.domain_list.values()
        ]
        h = geometry.domain_list[0].h
        NMK = [geometry.domain_list[idx].number_harmonics for idx in domain_keys]
        boundary_count = len(NMK) - 1

        size = NMK[0] + NMK[-1] + 2 * sum(NMK[1:len(NMK) - 1])

        results_per_mode = []

        if modes_to_calculate is None:
            num_bodies = len(geometry.body_arrangement.bodies)
            modes_to_calculate = np.arange(num_bodies)

        body_to_regions = {}
        current_region = 0
        for b_i, body in enumerate(geometry.body_arrangement.bodies):
            if isinstance(body, SteppedBody):
                n_steps = len(body.a)
                body_to_regions[b_i] = list(range(current_region, current_region + n_steps))
                current_region += n_steps
            else:
                body_to_regions[b_i] = [current_region]
                current_region += 1

        for mode_index in modes_to_calculate:
            heaving = [0] * len(domain_keys)
            
            if mode_index in body_to_regions:
                for r_idx in body_to_regions[mode_index]:
                    if r_idx < len(heaving):
                        heaving[r_idx] = 1

            c_vector = np.zeros((size - NMK[-1]), dtype=complex)
            col = 0

            for n in range(NMK[0]):
                c_vector[n] = heaving[0] * int_R_1n(0, n, a, h, d) * z_n_d(n)
            col += NMK[0]

            for i in range(1, boundary_count):
                M = NMK[i]
                for m in range(M):
                    c_vector[col + m] = heaving[i] * int_R_1n(i, m, a, h, d) * z_n_d(m)
                    c_vector[col + M + m] = heaving[i] * int_R_2n(i, m, a, h, d) * z_n_d(m)
                col += 2 * M

            hydro_p_term_sum = np.zeros(boundary_count, dtype=complex)
            for i in range(boundary_count):
                hydro_p_term_sum[i] = heaving[i] * int_phi_p_i(i, h, d, a)

            hydro_coef = 2 * pi * (np.dot(c_vector, X[:-NMK[-1]]) + sum(hydro_p_term_sum))

            hydro_coef_real = hydro_coef.real * rho
            if m0 == np.inf:
                hydro_coef_imag = 0
            else:
                hydro_coef_imag = hydro_coef.imag * omega(m0, h, g) * rho
            
            results_per_mode.append({
                "mode": mode_index,
                "real": hydro_coef_real,      
                "imag": hydro_coef_imag,      
                "excitation_phase": excitation_phase(X, NMK, m0, a),
                "excitation_force": excitation_force(hydro_coef_imag, m0, h)
            })

        return results_per_mode
    
    def calculate_potentials(self, problem, solution_vector: np.ndarray, m0, spatial_res, sharp, R_range: Optional[np.ndarray] = None, Z_range: Optional[np.ndarray] = None) -> Dict[str, Any]:
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
        self._ensure_m_k_and_N_k_arrays(problem, m0)
        cache = self.cache_list[problem]
        m_k_arr = cache.m_k_arr
        N_k_arr = cache.N_k_arr
        
        # Get geometry parameters directly from the body arrangement and domains
        geometry = problem.geometry
        body_arrangement = geometry.body_arrangement
        domain_list = problem.domain_list
        
        # These are the correct physical parameters for meshgrid and particular solution
        a = body_arrangement.a
        d = body_arrangement.d
        heaving = body_arrangement.heaving
        
        # These are needed for the homogeneous solution and coefficient reformatting
        h = geometry.h
        domain_keys = list(domain_list.keys())
        boundary_count = len(domain_keys) - 1
        NMK = [domain_list[idx].number_harmonics for idx in domain_keys]
        
        # --- The rest of the function remains the same ---
        Cs = self.reformat_coeffs(solution_vector, NMK, boundary_count)

        # 2. Create Meshgrid and Regions
        # Now make_R_Z will receive the correct a and d lists
        R, Z = make_R_Z(a, h, d, sharp, spatial_res)
        regions = []
        regions.append((R <= a[0]) & (Z < -d[0]))
        for i in range(1, boundary_count):
            regions.append((R > a[i-1]) & (R <= a[i]) & (Z < -d[i]))
        regions.append(R > a[-1])

        # Initialize potential arrays
        phi = np.full_like(R, np.nan + np.nan*1j, dtype=complex) 
        phiH = np.full_like(R, np.nan + np.nan*1j, dtype=complex) 
        phiP = np.full_like(R, np.nan + np.nan*1j, dtype=complex) 

        # --- 3. Vectorized Calculation of Potentials ---

        # Region 0 (Inner)
        if np.any(regions[0]):
            r_vals, z_vals = R[regions[0]], Z[regions[0]]
            n_vals = np.arange(NMK[0])
            R1n_vals = R_1n_vectorized(n_vals[:, None], r_vals[None, :], 0, h, d, a)
            Zn_vals = Z_n_i_vectorized(n_vals[:, None], z_vals[None, :], 0, h, d)
            phiH[regions[0]] = np.sum(Cs[0][:, None] * R1n_vals * Zn_vals, axis=0)

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

        # Exterior Region
        if np.any(regions[-1]):
            r_vals, z_vals = R[regions[-1]], Z[regions[-1]]
            k_vals = np.arange(NMK[-1])
            Lambda_vals = Lambda_k_vectorized(k_vals[:, None], r_vals[None, :], m0, a, m_k_arr)
            Zk_vals = Z_k_e_vectorized(k_vals[:, None], z_vals[None, :], m0, h, m_k_arr, N_k_arr)
            phiH[regions[-1]] = np.sum(Cs[-1][:, None] * Lambda_vals * Zk_vals, axis=0)
        
        # --- 4. Calculate Particular Potential (phiP) ---
        phiP[regions[0]] = heaving[0] * phi_p_i(d[0], R[regions[0]], Z[regions[0]], h)
        for i in range(1, boundary_count):
            phiP[regions[i]] = heaving[i] * phi_p_i(d[i], R[regions[i]], Z[regions[i]], h)
        phiP[regions[-1]] = 0

        # Sum to get total potential phi
        phi = phiH + phiP

        return {"R": R, "Z": Z, "phiH": phiH, "phiP": phiP, "phi": phi}
    
    def visualize_potential(self, field, R, Z, title):
        fig, ax = plt.subplots(figsize=(8, 6))
        contour = ax.contourf(R, Z, field, levels=50, cmap='viridis')
        fig.colorbar(contour, ax=ax)
        ax.set_title(title)
        ax.set_xlabel('Radial Distance (R)')
        ax.set_ylabel('Axial Distance (Z)')
        return fig, ax
        
    def calculate_velocities(self, problem, solution_vector: np.ndarray, m0, spatial_res, sharp, R_range: Optional[np.ndarray] = None, Z_range: Optional[np.ndarray] = None) -> Dict[str, Any]:
        self._ensure_m_k_and_N_k_arrays(problem, m0)
        cache = self.cache_list[problem]
        m_k_arr, N_k_arr = cache.m_k_arr, cache.N_k_arr

        geometry = problem.geometry
        body_arrangement = geometry.body_arrangement
        domain_list = problem.domain_list

        body_a = body_arrangement.a
        body_d = body_arrangement.d
        body_heaving = body_arrangement.heaving
        
        h = geometry.h
        domain_keys = list(domain_list.keys())
        boundary_count = len(domain_keys) - 1
        NMK = [domain_list[idx].number_harmonics for idx in domain_keys]
        
        domain_a = [domain_list[idx].a for idx in domain_keys]
        domain_d = [domain_list[idx].di for idx in domain_keys]

        Cs = self.reformat_coeffs(solution_vector, NMK, boundary_count)

        R, Z = make_R_Z(body_a, h, body_d, sharp, spatial_res, R_range=R_range, Z_range=Z_range)
        
        regions = []
        regions.append(R <= body_a[0])
        for i in range(1, boundary_count):
            regions.append((R > body_a[i-1]) & (R <= body_a[i])) 
        regions.append(R > body_a[-1]) 
        
        vrH = np.full(R.shape, np.nan, dtype=complex)
        vzH = np.full(R.shape, np.nan, dtype=complex)

        if np.any(regions[0]):
            r, z = R[regions[0]], Z[regions[0]]
            n = np.arange(NMK[0])
            vrH[regions[0]] = np.sum(Cs[0][:, None] * diff_R_1n_vectorized(n[:, None], r[None, :], 0, h, domain_d, domain_a) * Z_n_i_vectorized(n[:, None], z[None, :], 0, h, domain_d), axis=0)
            vzH[regions[0]] = np.sum(Cs[0][:, None] * R_1n_vectorized(n[:, None], r[None, :], 0, h, domain_d, domain_a) * diff_Z_n_i_vectorized(n[:, None], z[None, :], 0, h, domain_d), axis=0)

        for i in range(1, boundary_count):
            if np.any(regions[i]):
                r, z = R[regions[i]], Z[regions[i]]
                m = np.arange(NMK[i])
                vr_term1 = Cs[i][:NMK[i], None] * diff_R_1n_vectorized(m[:, None], r[None, :], i, h, domain_d, domain_a)
                vr_term2 = Cs[i][NMK[i]:, None] * diff_R_2n_vectorized(m[:, None], r[None, :], i, h, domain_d, domain_a)
                vrH[regions[i]] = np.sum((vr_term1 + vr_term2) * Z_n_i_vectorized(m[:, None], z[None, :], i, h, domain_d), axis=0)
                vz_term1 = Cs[i][:NMK[i], None] * R_1n_vectorized(m[:, None], r[None, :], i, h, domain_d, domain_a)
                vz_term2 = Cs[i][NMK[i]:, None] * R_2n_vectorized(m[:, None], r[None, :], i, domain_a, h, domain_d)
                vzH[regions[i]] = np.sum((vz_term1 + vz_term2) * diff_Z_n_i_vectorized(m[:, None], z[None, :], i, h, domain_d), axis=0)

        if np.any(regions[-1]):
            r, z = R[regions[-1]], Z[regions[-1]]
            k = np.arange(NMK[-1])
            vrH[regions[-1]] = np.sum(Cs[-1][:, None] * diff_Lambda_k_vectorized(k[:, None], r[None, :], m0, domain_a, m_k_arr) * Z_k_e_vectorized(k[:, None], z[None, :], m0, h, m_k_arr, N_k_arr), axis=0)
            vzH[regions[-1]] = np.sum(Cs[-1][:, None] * Lambda_k_vectorized(k[:, None], r[None, :], m0, domain_a, m_k_arr) * diff_Z_k_e_vectorized(k[:, None], z[None, :], m0, h, m_k_arr, N_k_arr), axis=0)
        
        vrP = np.full(R.shape, 0.0, dtype=complex)
        vzP = np.full(R.shape, 0.0, dtype=complex)
        
        vrP[regions[0]] = body_heaving[0] * diff_r_phi_p_i(body_d[0], R[regions[0]], h)
        vzP[regions[0]] = body_heaving[0] * diff_z_phi_p_i(body_d[0], Z[regions[0]], h)
        for i in range(1, boundary_count):
            if body_heaving[i]:
                vrP[regions[i]] = body_heaving[i] * diff_r_phi_p_i(body_d[i], R[regions[i]], h)
                vzP[regions[i]] = body_heaving[i] * diff_z_phi_p_i(body_d[i], Z[regions[i]], h)

        vr = vrH + vrP
        vz = vzH + vzP
        
        for i in range(boundary_count):
            body_mask = (regions[i]) & (Z > -body_d[i])
            vr[body_mask] = np.nan
            vz[body_mask] = np.nan

        return {"R": R, "Z": Z, "vrH": vrH, "vzH": vzH, "vrP": vrP, "vzP": vzP, "vr": vr, "vz": vz}
    
    def run_and_store_results(self, problem_index: int) -> Results:
        original_problem = self.problem_list[problem_index]
        original_geometry = original_problem.geometry
        original_bodies = original_geometry.body_arrangement.bodies
        h = original_geometry.h
        
        original_domain_list = original_problem.domain_list
        original_domain_keys = list(original_domain_list.keys())
        NMK_list = [original_domain_list[idx].number_harmonics for idx in original_domain_keys]

        problem_modes = original_problem.modes
        omegas_to_run = original_problem.frequencies
        
        num_modes = len(problem_modes)
        num_freqs = len(omegas_to_run)

        results = Results(original_problem)

        full_added_mass_matrix = np.full((num_freqs, num_modes, num_modes), np.nan)
        full_damping_matrix = np.full((num_freqs, num_modes, num_modes), np.nan)
        all_potentials_batch_data = []

        for freq_idx, omega in enumerate(omegas_to_run):
            m0 = wavenumber(omega, h) 
            for i_idx, radiating_mode in enumerate(problem_modes):
                try:
                    temp_bodies = []
                    for body_j, original_body in enumerate(original_bodies):
                        if not isinstance(original_body, SteppedBody):
                            raise TypeError(
                                "run_and_store_results only supports SteppedBody objects. "
                                f"Found {type(original_body)}."
                            )
                        is_heaving = (body_j == radiating_mode)
                        temp_bodies.append(
                            SteppedBody(
                                a=original_body.a, 
                                d=original_body.d, 
                                slant_angle=original_body.slant_angle, 
                                heaving=is_heaving
                            )
                        )
                    temp_arrangement = ConcentricBodyGroup(temp_bodies)
                    temp_geometry = BasicRegionGeometry(temp_arrangement, h=h, NMK=NMK_list)
                    temp_problem = MEEMProblem(temp_geometry)
                    temp_problem.set_frequencies(np.array([omega]))
                    temp_engine = MEEMEngine(problem_list=[temp_problem])
                    X_i = temp_engine.solve_linear_system_multi(temp_problem, m0)
                    hydro_coeffs_col = temp_engine.compute_hydrodynamic_coefficients(
                        temp_problem, X_i, m0, modes_to_calculate=problem_modes
                    )
                    for coeff_dict in hydro_coeffs_col:
                        j_mode = coeff_dict['mode']
                        j_idx_result = np.where(problem_modes == j_mode)[0]
                        if j_idx_result.size == 0:
                            print(f"Warning: Mode {j_mode} not found in problem_modes. Skipping.")
                            continue
                        j_idx = j_idx_result[0]
                        full_added_mass_matrix[freq_idx, j_idx, i_idx] = coeff_dict['real']
                        full_damping_matrix[freq_idx, j_idx, i_idx] = coeff_dict['imag']

                    Cs = temp_engine.reformat_coeffs(X_i, NMK_list, len(NMK_list) - 1)
                    current_mode_potentials = {}
                    
                    domain_list = temp_problem.geometry.domain_list
                    if isinstance(domain_list, dict):
                        domain_iterable = domain_list.values()
                    else:
                        domain_iterable = domain_list

                    for domain in domain_iterable:
                        domain_idx = domain.index
                        domain_coeffs = Cs[domain_idx]
                        current_mode_potentials[domain.index] = {
                            "potentials": domain_coeffs,
                            "r_coords_dict": {f"r_h{k}": 0.0 for k in range(len(domain_coeffs))},
                            "z_coords_dict": {f"z_h{k}": 0.0 for k in range(len(domain_coeffs))} 
                        }

                    all_potentials_batch_data.append({
                        "frequency_idx": freq_idx,
                        "mode_idx": i_idx,
                        "data": current_mode_potentials,
                    })
                        
                except np.linalg.LinAlgError as e:
                    print(f"  ERROR: Could not solve for freq={omega:.4f}, mode={radiating_mode}: {e}. Storing NaN.")
                    full_added_mass_matrix[freq_idx, :, i_idx] = np.nan
                    full_damping_matrix[freq_idx, :, i_idx] = np.nan
                    continue 

        results.store_hydrodynamic_coefficients(
            frequencies=omegas_to_run,
            added_mass_matrix=full_added_mass_matrix,
            damping_matrix=full_damping_matrix,
        )

        if all_potentials_batch_data:
            results.store_all_potentials(all_potentials_batch_data)

        return results
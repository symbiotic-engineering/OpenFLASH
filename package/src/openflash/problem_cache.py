# package/src/openflash/problem_cache.py
import numpy as np
from typing import Callable, Dict, Any, Optional

from openflash.multi_equations import *

class ProblemCache:
    def __init__(self, problem):
        self.problem = problem
        self.A_template: Optional[np.ndarray] = None
        self.b_template: Optional[np.ndarray] = None
        self.m0_dependent_A_indices: list[tuple[int, int, Callable]] = []
        self.m0_dependent_b_indices: list[tuple[int, Callable]] = []

        self.m_k_entry_func: Optional[Callable] = None
        self.N_k_func: Optional[Callable] = None
        self.m_k_arr: Optional[np.ndarray] = None
        self.N_k_arr: Optional[np.ndarray] = None
        
        # --- FIX: Track the m0 value associated with the current cache ---
        self.cached_m0: Optional[float] = None 
        # -----------------------------------------------------------------

        self.I_nm_vals: Optional[np.ndarray] = None
        self.named_closures: Dict[str, Any] = {}

    def _set_A_template(self, A_template: np.ndarray):
        self.A_template = A_template

    def _set_b_template(self, b_template: np.ndarray):
        self.b_template = b_template

    def _add_m0_dependent_A_entry(self, row: int, col: int, calc_func: Callable):
        self.m0_dependent_A_indices.append((row, col, calc_func))

    def _add_m0_dependent_b_entry(self, row: int, calc_func: Callable):
        self.m0_dependent_b_indices.append((row, calc_func))

    def _set_m_k_and_N_k_funcs(self, m_k_entry_func: Callable, N_k_func: Callable):
        self.m_k_entry_func = m_k_entry_func
        self.N_k_func = N_k_func

    # --- FIX: Accept m0 as an argument to store it ---
    def _set_precomputed_m_k_N_k(self, m_k_arr: np.ndarray, N_k_arr: np.ndarray, m0: float):
        """
        Sets the pre-computed m_k and N_k arrays for a specific m0.
        """
        self.m_k_arr = m_k_arr
        self.N_k_arr = N_k_arr
        self.cached_m0 = m0
    # ------------------------------------------------

    def _set_I_nm_vals(self, I_nm_vals: np.ndarray):
        self.I_nm_vals = I_nm_vals

    def _get_A_template(self) -> np.ndarray:
        if self.A_template is None:
            raise ValueError("A_template has not been set.")
        return self.A_template.copy()

    def _get_b_template(self) -> np.ndarray:
        if self.b_template is None:
            raise ValueError("b_template has not been set.")
        return self.b_template.copy()

    def _set_closure(self, key: str, closure):
        self.named_closures[key] = closure

    def _get_closure(self, key: str):
        return self.named_closures.get(key, None)
    def _set_integration_constants(self, int_R1, int_R2, int_phi):
        self.int_R1_vals = int_R1
        self.int_R2_vals = int_R2
        self.int_phi_vals = int_phi
    
    def _get_integration_constants(self):
        if self.int_R1_vals is None:
             raise ValueError("Integration constants have not been set.")
        return self.int_R1_vals, self.int_R2_vals, self.int_phi_vals
    def refresh_forcing_terms(self, problem):
        """
        Re-calculates b_template and m0_dependent_b_indices based on the 
        current heaving configuration of the problem.
        This allows re-using the cache (and Matrix A) while changing the active mode.
        """
        domain_list = problem.domain_list
        domain_keys = list(domain_list.keys())
        
        # Extract geometry params (same as build_problem_cache)
        h = domain_list[0].h
        d = [domain_list[idx].di for idx in domain_keys]
        a = [domain_list[idx].a for idx in domain_keys]
        NMK = [domain.number_harmonics for domain in domain_list.values()]
        
        # Crucial: Get the NEW heaving flags
        heaving = [domain_list[idx].heaving for idx in domain_keys]
        
        boundary_count = len(NMK) - 1
        size = NMK[0] + NMK[-1] + 2 * sum(NMK[1:len(NMK) - 1])

        # 1. Reset b_template
        b_template = np.zeros(size, dtype=complex)
        
        index = 0
        for bd in range(boundary_count):
            if bd == (boundary_count - 1):
                for n in range(NMK[-2]):
                    b_template[index] = b_potential_end_entry(n, bd, heaving, h, d, a)
                    index += 1
            else:
                num_entries = NMK[bd + (d[bd] <= d[bd + 1])]
                for n in range(num_entries):
                    b_template[index] = b_potential_entry(n, bd, d, heaving, h, a)
                    index += 1
        
        self._set_b_template(b_template)

        # 2. Reset m0_dependent_b_indices
        self.m0_dependent_b_indices = [] # Clear old indices
        
        # Re-populate using the loop logic from build_problem_cache
        # Note: We must reset 'index' to match the velocity loop start position
        # The velocity loop starts after the potential loop.
        
        # Calculate offset where velocity equations start
        potential_eq_count = 0
        for bd in range(boundary_count):
            if bd == (boundary_count - 1):
                potential_eq_count += NMK[-2]
            else:
                potential_eq_count += NMK[bd + (d[bd] <= d[bd + 1])]
        
        index = potential_eq_count 

        for bd in range(boundary_count):
            if bd == (boundary_count - 1):
                for n_local in range(NMK[-1]):
                    # Closure to capture n_local and heaving state
                    calc_func = lambda p, m0, mk, Nk, Imk, n=n_local: \
                        b_velocity_end_entry(n, bd, heaving, a, h, d, m0, NMK, mk, Nk)
                    self._add_m0_dependent_b_entry(index, calc_func)
                    index += 1
            else:
                num_entries = NMK[bd + (d[bd] > d[bd + 1])]
                for n in range(num_entries):
                    # b_velocity_entry is not m0 dependent, so it goes into b_template?
                    # Wait, look at build_problem_cache in original file.
                    # b_velocity_entry IS put into b_template.
                    b_template[index] = b_velocity_entry(n, bd, heaving, a, h, d)
                    index += 1
        
        # Update the template again with the velocity entries added
        self._set_b_template(b_template)
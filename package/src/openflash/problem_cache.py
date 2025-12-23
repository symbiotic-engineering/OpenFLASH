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
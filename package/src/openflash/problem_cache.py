# problem_cache.py
import numpy as np
from typing import Callable, Dict, Any

from openflash.multi_equations import *

class ProblemCache:
    def __init__(self, problem):
        self.problem = problem
        self.A_template: np.ndarray = None
        self.b_template: np.ndarray = None
        self.m0_dependent_A_indices: list[tuple[int, int, Callable]] = []
        self.m0_dependent_b_indices: list[tuple[int, Callable]] = []

        # Attributes to store references to m_k_entry and N_k functions
        self.m_k_entry_func: Callable = None
        self.N_k_func: Callable = None
        self.m_k_arr: np.ndarray = None
        self.N_k_arr: np.ndarray = None
        
        # New attribute to store the m0-independent I_nm_vals matrix
        self.I_nm_vals: np.ndarray = None

        self.named_closures: Dict[str, Any] = {}

    def set_A_template(self, A_template: np.ndarray):
        self.A_template = A_template

    def set_b_template(self, b_template: np.ndarray):
        self.b_template = b_template

    def add_m0_dependent_A_entry(self, row: int, col: int, calc_func: Callable):
        """
        Add an m0-dependent A matrix entry.
        :param row: Row index of the entry.
        :param col: Column index of the entry.
        :param calc_func: A callable that takes (problem, m0, m_k_arr, N_k_arr) and returns the complex value.
        """
        self.m0_dependent_A_indices.append((row, col, calc_func))

    def add_m0_dependent_b_entry(self, row: int, calc_func: Callable):
        """
        Add an m0-dependent b vector entry.
        :param row: Row index of the entry.
        :param calc_func: A callable that takes (problem, m0, m_k_arr, N_k_arr) and returns the complex value.
        """
        self.m0_dependent_b_indices.append((row, calc_func))

    def set_m_k_and_N_k_funcs(self, m_k_entry_func: Callable, N_k_func: Callable):
        """
        Sets the references to the m_k_entry and N_k functions.
        """
        self.m_k_entry_func = m_k_entry_func
        self.N_k_func = N_k_func

    def set_precomputed_m_k_N_k(self, m_k_arr: np.ndarray, N_k_arr: np.ndarray):
        """
        Sets the pre-computed m_k and N_k arrays for a specific m0.
        """
        self.m_k_arr = m_k_arr
        self.N_k_arr = N_k_arr

    def set_I_nm_vals(self, I_nm_vals: np.ndarray):
        """
        Sets the pre-computed m0-independent I_nm_vals matrix.
        """
        self.I_nm_vals = I_nm_vals

    def get_A_template(self) -> np.ndarray:
        if self.A_template is None:
            raise ValueError("A_template has not been set.")
        return self.A_template.copy()

    def get_b_template(self) -> np.ndarray:
        if self.b_template is None:
            raise ValueError("b_template has not been set.")
        return self.b_template.copy()

    def set_closure(self, key: str, closure):
        self.named_closures[key] = closure

    def get_closure(self, key: str):
        return self.named_closures.get(key, None)
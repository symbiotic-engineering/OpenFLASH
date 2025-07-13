# problem_cache.py
import numpy as np
from typing import Callable
from openflash.multi_equations import *

# store the pre-computed m0-independent parts of matrices and vectors
# along with the specific indices where m0-dependent terms need to be inserted/updated.

class ProblemCache:
    def __init__(self, problem):
        self.problem = problem
        self.A_template = None
        self.b_template = None
        self.m0_dependent_A_indices = [] # Stores (row, col, func_to_calc_value) for m0-dependent A entries
        self.m0_dependent_b_indices = [] # Stores (row, func_to_calc_value) for m0-dependent b entries

        # Add attributes to store the *references* to the m_k_entry and N_k functions
        # These will be set by the MEEMEngine's _build_problem_cache method.
        self.m_k_entry_func: Callable = None
        self.N_k_func: Callable = None
        self.m_k_arr: np.ndarray = None
        self.N_k_arr: np.ndarray = None


    def set_A_template(self, A_template: np.ndarray):
        self.A_template = A_template

    def set_b_template(self, b_template: np.ndarray):
        self.b_template = b_template

    def add_m0_dependent_A_entry(self, row: int, col: int, calc_func: Callable):
        """
        Add an m0-dependent A matrix entry.
        :param row: Row index of the entry.
        :param col: Column index of the entry.
        :param calc_func: A callable that takes (problem, m0, m_k_arr, N_k_arr) and returns the complex value for A[row, col].It now expects the pre-computed m_k_arr and N_k_arr.
        """
        self.m0_dependent_A_indices.append((row, col, calc_func))

    def add_m0_dependent_b_entry(self, row: int, calc_func: Callable):
        """
        Add an m0-dependent b vector entry.
        :param row: Row index of the entry.
        :param calc_func: A callable that takes (problem, m0, m_k_arr, N_k_arr) and returns the complex value for b[row]. It now expects the pre-computed m_k_arr and N_k_arr.
        """
        self.m0_dependent_b_indices.append((row, calc_func))

    def set_m_k_and_N_k_funcs(self, m_k_entry_func: Callable, N_k_func: Callable):
        """
        Sets the references to the m_k_entry and N_k functions from multi_equations.
        These are the actual functions, not their computed values.
        """
        self.m_k_entry_func = m_k_entry_func
        self.N_k_func = N_k_func

    # Method to set the pre-computed arrays
    def set_precomputed_m_k_N_k(self, m_k_arr: np.ndarray, N_k_arr: np.ndarray):
        self.m_k_arr = m_k_arr
        self.N_k_arr = N_k_arr


    def get_A_template(self) -> np.ndarray:
        if self.A_template is None:
            raise ValueError("A_template has not been set.")
        return self.A_template.copy() # Return a copy to prevent accidental modification of the template

    def get_b_template(self) -> np.ndarray:
        if self.b_template is None:
            raise ValueError("b_template has not been set.")
        return self.b_template.copy() # Return a copy
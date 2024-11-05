# domain.py

from typing import List
import numpy as np
from equations import *
from constants import *

# Import multi-region functions
from multi_equations import *
from multi_constants import *

class Domain:
    """
    Represents a sub-region within the geometry, characterized by its own properties
    and methods to compute eigenfunctions and potentials.
    """

    def __init__(self, index: int, number_harmonics: int, height: float, top_BC, bottom_BC, category: str, params: dict):
        """
        Initialize the Domain object.

        :param index: Index of the domain in the geometry.
        :param number_harmonics: Number of harmonics.
        :param height: Height of the domain.
        :param top_BC: Top boundary condition.
        :param bottom_BC: Bottom boundary condition.
        :param category: Category/type of the domain (e.g., 'inner', 'outer', 'exterior', 'multi').
        :param params: Dictionary containing parameters like h, di, ai, m0, radial_width, slant, heave.
        """
        self.index = index
        self.number_harmonics = number_harmonics
        self.height = height
        self.top_BC = top_BC
        self.bottom_BC = bottom_BC
        self.category = category  # 'inner', 'outer', 'exterior', 'multi'
        self.params = params

        # Original base code parameters
        self.h = params.get('h', 1.0)
        self.di = params.get('di', 0.0)
        self.a1 = params.get('a1', 0.5)
        self.a2 = params.get('a2', 1.0)
        self.m0 = params.get('m0', 1.0)
        self.radial_width = params.get('radial_width', self.a2 - self.a1)
        self.m_k_vals = []  # For exterior domain eigenvalues

        # Set radial properties based on region characteristics
        self.r_properties = self.set_r_properties()
        self.heaving = params.get('heave', heaving[self.index] if self.index < len(heaving) else False)
        self.slant = params.get('slant', False)

        # For multi-region support
        self.ai = params.get('ai', a[self.index] if self.index < len(a) else None)
        self.scale = np.mean(a) if len(a) > 0 else 1.0

    def set_r_properties(self):
        """
        Define radial properties of the domain.

        :return: A dictionary with r conditions.
        """
        r_properties = {
            "r=0": self.category == 'inner',
            "r=infinity": self.category == 'exterior',
            "0<r<infinity": self.category in ['inner', 'outer', 'exterior', 'multi']
        }
        return r_properties

    def radial_eigenfunctions(self, r: float, n: int):
        """
        Compute radial eigenfunctions at a given radial coordinate.

        :param r: Radial coordinate.
        :param n: Mode number.
        :return: Value of the radial eigenfunctions at r.
        """
        if self.category == 'inner':
            R1n_val = R_1n_1(n, r)
            R2n_val = R_2n_2(n, r)
            return R1n_val, R2n_val
        elif self.category == 'outer':
            R1n_val = R_1n_2(n, r)
            R2n_val = R_2n_2(n, r)
            return R1n_val, R2n_val
        elif self.category == 'exterior':
            if not self.m_k_vals:
                self.m_k_vals = [m_k(k + 1) for k in range(self.number_harmonics)]
            Lambda_k_vals = [Lambda_k_r(k + 1, r) for k in range(self.number_harmonics)]
            return Lambda_k_vals
        elif self.category == 'multi':
            # For multi-region support
            R1n_val = R_1n(n, r, self.index)
            R2n_val = R_2n(n, r, self.index)
            return R1n_val, R2n_val
        else:
            raise ValueError("Unknown domain category.")

    def vertical_eigenfunctions(self, z: float, n: int):
        """
        Compute vertical eigenfunctions at a given vertical coordinate.

        :param z: Vertical coordinate.
        :param n: Mode number.
        :return: Value of the vertical eigenfunction at z.
        """
        if self.category == 'inner':
            Z_n_val = Z_n_i1(n, z)
            return Z_n_val
        elif self.category == 'outer':
            Z_n_val = Z_n_i2(n, z)
            return Z_n_val
        elif self.category == 'exterior':
            if not self.m_k_vals:
                self.m_k_vals = [m_k(k + 1) for k in range(self.number_harmonics)]
            Z_k_vals = [Z_n_e(k + 1, z) for k in range(self.number_harmonics)]
            return Z_k_vals
        elif self.category == 'multi':
            # For multi-region support
            Z_n_val = Z_n_i(n, z, self.index)
            return Z_n_val
        else:
            raise ValueError("Unknown domain category.")

    def particular_potential(self, r: float, z: float):
        """
        Compute the particular solution of the potential at given coordinates.

        :param r: Radial coordinate.
        :param z: Vertical coordinate.
        :return: Value of the particular potential at (r, z).
        """
        if self.category == 'inner':
            return phi_p_i1(r, z)
        elif self.category == 'outer':
            return phi_p_i2(r, z)
        elif self.category == 'exterior':
            return 0.0  # No particular potential in the exterior domain
        elif self.category == 'multi':
            # For multi-region support
            return phi_p_i(self.di, r, z)
        else:
            raise ValueError("Unknown domain category.")

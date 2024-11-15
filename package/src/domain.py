# domain.py

from typing import List
import numpy as np
from multi_equations import *
from multi_constants import h, d, a, m0, heaving

class Domain:
    """
    Represents a sub-region within the geometry, characterized by its own properties
    and methods to compute eigenfunctions and potentials.
    """

    def __init__(self, number_harmonics: int, height: float, radial_width: float, top_BC, bottom_BC, category: str, params: dict, index: int):
        """
        Initialize the Domain object.

        :param number_harmonics: Number of harmonics.
        :param height: Height of the domain.
        :param radial_width: Radial width of the domain.
        :param top_BC: Top boundary condition.
        :param bottom_BC: Bottom boundary condition.
        :param category: Category/type of the domain.
        :param params: Dictionary containing parameters like h, di, a, m0.
        :param index: Index of the domain in the multi-region setup.
        """
        self.number_harmonics = number_harmonics
        self.height = height
        self.radial_width = radial_width
        self.top_BC = top_BC
        self.bottom_BC = bottom_BC
        self.category = category  # 'inner', 'outer', 'exterior', 'multi'
        self.params = params
        self.index = index  # Index in the domain list

        self.h = params.get('h', h)
        self.di = params.get('di', d[index] if index < len(d) else 0.0)
        self.a = params.get('a', a[index] if index < len(a) else a[-1])
        self.m0 = params.get('m0', m0)
        self.scale = params.get('scale', np.mean(a))
        self.heaving = params.get('heaving', heaving[index] if index < len(heaving) else 0)
        self.slant = params.get('slant', False)
        self.m_k_vals = []  # For exterior domain eigenvalues

    def radial_eigenfunctions(self, r: float, n: int):
        """
        Compute radial eigenfunctions at a given radial coordinate.

        :param r: Radial coordinate.
        :param n: Mode number.
        :return: Value of the radial eigenfunctions at r.
        """
        if self.category in ['inner', 'multi']:
            R1n_val = R_1n(n, r, self.index)
            R2n_val = R_2n(n, r, self.index) if self.index > 0 else 0.0  # For i=0, R_2n is not defined
            return R1n_val, R2n_val
        elif self.category == 'exterior':
            if not self.m_k_vals:
                self.m_k_vals = [m_k(k + 1) for k in range(self.number_harmonics)]
            Lambda_k_vals = [Lambda_k(k + 1, r) for k in range(self.number_harmonics)]
            return Lambda_k_vals
        else:
            raise ValueError("Unknown domain category.")

    def vertical_eigenfunctions(self, z: float, n: int):
        """
        Compute vertical eigenfunctions at a given vertical coordinate.

        :param z: Vertical coordinate.
        :param n: Mode number.
        :return: Value of the vertical eigenfunction at z.
        """
        if self.category in ['inner', 'multi']:
            Z_n_val = Z_n_i(n, z, self.index)
            return Z_n_val
        elif self.category == 'exterior':
            if not self.m_k_vals:
                self.m_k_vals = [m_k(k + 1) for k in range(self.number_harmonics)]
            Z_k_vals = [Z_n_e(k + 1, z) for k in range(self.number_harmonics)]
            return Z_k_vals
        else:
            raise ValueError("Unknown domain category.")

    def particular_potential(self, r: float, z: float):
        """
        Compute the particular solution of the potential at given coordinates.

        :param r: Radial coordinate.
        :param z: Vertical coordinate.
        :return: Value of the particular potential at (r, z).
        """
        if self.category in ['inner', 'multi']:
            return phi_p_i(self.di, r, z)
        elif self.category == 'exterior':
            return 0.0  # No particular potential in the exterior domain
        else:
            raise ValueError("Unknown domain category.")

    
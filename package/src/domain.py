# domain.py

from typing import List
import numpy as np
from multi_equations import *
from multi_constants import *

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

    
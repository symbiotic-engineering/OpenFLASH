# domain.py
from __future__ import annotations

from typing import List, Dict, Union
import numpy as np
from openflash.multi_equations import *

class Domain:
    """
    Represents a sub-region within the geometry, characterized by its own properties
    and methods to compute eigenfunctions and potentials.
    """

    def __init__(self, number_harmonics: int, height: float, radial_width: float, top_BC, bottom_BC, category: str, params: dict, index: int, geometry: 'Geometry'):
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
        :param geometry: The Geometry object that this domain belongs to.

        """
        self.number_harmonics = number_harmonics
        self.height = height
        self.radial_width = radial_width
        self.top_BC = top_BC
        self.bottom_BC = bottom_BC
        self.category = category  # 'inner', 'outer', 'exterior', 'multi'
        self.params = params
        self.index = index  # Index in the domain list
        self.geometry = geometry
        

        self.h = params.get('h', geometry.z_coordinates.get('h'))
        self.di = self._get_di()
        self.a = self._get_a()
        self.m0 = params.get('m0')

        # Convert dict_values to NumPy array before calculating mean
        r_values = np.array(list(geometry.r_coordinates.values()))
        self.scale = params.get('scale', np.mean(r_values))

        self.heaving = self._get_heaving()
        self.slant = params.get('slant', False)
        self.m_k_vals = []  # For exterior domain eigenvalues]
        self.r_coords = self._get_r_coords()
        self.z_coords = self._get_z_coords()
        
    def _get_di(self) -> Union[float, None]:
        """Gets the di parameter based on category and index."""
        if self.category == 'inner':
            return self.params.get('di')
        elif self.category == 'outer':
            return self.params.get('di')
        elif self.category == 'exterior':
            return None # Exterior domain doesn't have di
        else:
            return self.params.get('di')

    def _get_a(self) -> Union[float, List[float], None]:
        """Gets the 'a' parameter based on category."""
        if self.category == 'inner':
            return self.params.get('a', self.geometry.r_coordinates.get('a1'))
        elif self.category == 'outer':
            return self.params.get('a', self.geometry.r_coordinates.get('a2'))
        elif self.category == 'exterior':
            return None # Exterior domain doesn't have a
        else:
            return self.params.get('a')

    def _get_heaving(self) -> Union[int, None]:
        """Gets the heaving parameter based on index."""
        return self.params.get('heaving')

    def _get_r_coords(self) -> Union[float, List[float], None]:
        """Gets the r coordinates based on category."""
        if self.category == 'inner':
            return 0 # r = 0 for inner
        elif self.category == 'outer':
            return [self.geometry.r_coordinates.get('a1'), self.geometry.r_coordinates.get('a2')]
        elif self.category == 'exterior':
            return np.inf # r -> infinity for exterior
        else:
            return self.geometry.r_coordinates.get('a1') # default to a1

    def _get_z_coords(self) -> Union[float, List[float], None]:
        """Gets the z coordinates."""
        return [0, self.h] # z from 0 to h
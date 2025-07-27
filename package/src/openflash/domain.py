from __future__ import annotations
from typing import Union, List
import numpy as np

class Domain:
    """
    Represents a region of the MEEM geometry with its own physical properties and boundary conditions.
    """

    def __init__(
        self,
        number_harmonics: int,
        height: float,
        radial_width: float,
        top_BC,
        bottom_BC,
        category: str,
        params: dict,
        index: int,
        geometry: Geometry
    ):
        self.number_harmonics = number_harmonics
        self.height = height
        self.radial_width = radial_width
        self.top_BC = top_BC
        self.bottom_BC = bottom_BC
        self.category = category  # 'inner', 'outer', or 'exterior'
        self.params = params
        self.index = index
        self.geometry = geometry

        self.h = params.get('h', geometry.z_coordinates.get('h'))
        self.di = self._get_di()
        self.a = self._get_a()
        self.scale = params.get('scale', np.mean(list(geometry.r_coordinates.values())))
        self.heaving = params.get('heaving')
        self.slant = params.get('slant', False)

        self.m_k_vals = []  # Exterior eigenvalues placeholder
        self.r_coords = self._get_r_coords()
        self.z_coords = self._get_z_coords()
        self.eigenfunction = None  # Will be set later based on configuration

    def _get_di(self) -> Union[float, None]:
        return self.params.get('di') if self.category != 'exterior' else None

    def _get_a(self) -> Union[float, None]:
        return self.params.get('a') if self.category != 'exterior' else None

    def _get_r_coords(self) -> Union[float, List[float]]:
        if self.category == 'inner':
            return 0
        elif self.category == 'outer':
            return [self.geometry.r_coordinates.get('a1'), self.geometry.r_coordinates.get('a2')]
        elif self.category == 'exterior':
            return np.inf
        return self.geometry.r_coordinates.get('a1')

    def _get_z_coords(self) -> List[float]:
        return [0, self.h]
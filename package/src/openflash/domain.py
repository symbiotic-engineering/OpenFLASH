from __future__ import annotations
from typing import Union, List, Dict, Optional
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
    

    def build_domain_params(
        NMK: List[int],
        a: List[float],
        d: List[float],
        heaving: List[Union[int, bool]],
        h: float, slant: Optional[List[Union[int, bool]]] = None
    ) -> List[Dict]:
        """
        Creates a structured list of domain parameters from simulation inputs.

        Args:
            NMK: List of harmonic counts for each domain.
            a: List of cylinder radii, must be strictly increasing.
            d: List of cylinder drafts.
            heaving: List of heaving states (0 or 1) for each cylinder.
            h: Total water depth.
            slant: Optional list of slant states (0 or 1) for each cylinder.

        Returns:
            A list of dictionaries, each defining a simulation domain.
        """
        boundary_count = len(NMK) - 1
        
        # If slant is not provided, default to all zeros.
        if slant is None:
            slant = [0] * boundary_count
            
        assert len(a) == boundary_count, "Length of 'a' must be one less than length of 'NMK'"
        assert len(d) == boundary_count, "Length of 'd' must match 'a'"
        assert len(heaving) == boundary_count, "Length of 'heaving' must match 'a'"
        for arr, name in zip([a, d, heaving], ['a', 'd', 'heaving']):
            assert len(arr) == boundary_count, f"{name} should have length len(NMK) - 1"

        for entry in heaving:
            assert entry in (0, 1), "heaving entries should be 0 or 1"

        left = 0
        for radius in a:
            assert radius > left, "a values should be increasing and > 0"
            left = radius

        for depth in d:
            assert 0 <= depth < h, "d must be nonnegative and less than h"

        for val in NMK:
            assert isinstance(val, int) and val > 0, "NMK entries must be positive integers"


        domain_params = []

        for idx in range(len(NMK)):
            if idx == 0:
                category = 'inner'
            elif idx == boundary_count:
                category = 'exterior'
            else:
                category = 'outer'

            # Assign boundary conditions based on domain category
            bottom_bc = 'Sea floor'
            if category == 'exterior':
                top_bc = 'Wave surface'
            else:  # 'inner' or 'outer'
                top_bc = 'Body'

            param = {
                'number_harmonics': NMK[idx],
                'height': h,
                'radial_width': a[idx] if idx < boundary_count else a[-1] * 1.5,
                'top_BC': top_bc,
                'bottom_BC': bottom_bc,
                'category': category,
            }

            if idx < boundary_count:
                param['a'] = a[idx]
                param['di'] = d[idx]
                param['heaving'] = heaving[idx]
                param['slant'] = slant[idx]

            domain_params.append(param)

        return domain_params

    def build_r_coordinates_dict(a: list[float]) -> dict[str, float]:
        """
        Given a list of radial boundary values, return a dict suitable for Geometry.r_coordinates.

        Parameters
        ----------
        a : list of float
            The radial boundary values [a1, a2, a3, ...]
        """
        return {f'a{i+1}': val for i, val in enumerate(a)}

    def build_z_coordinates_dict(h: float) -> dict[str, float]:
        """
        Given a height value, return a dict suitable for Geometry.z_coordinates.

        Parameters
        ----------
        h : float
            The height value

        Returns
        -------
        dict
            Dictionary mapping 'h' to the height value.
        """
        return {'h': h}
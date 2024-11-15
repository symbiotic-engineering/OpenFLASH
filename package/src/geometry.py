# geometry.py

from typing import Dict, List
from domain import Domain
import numpy as np
from multi_constants import h, d, a, heaving, m0

class Geometry:
    """
    Represents the physical geometry of the problem, including coordinates and domain parameters.
    """

    def __init__(self, r_coordinates: Dict[str, float], z_coordinates: Dict[str, float], domain_params: List[Dict]):
        """
        Initialize the Geometry object.

        :param r_coordinates: Dictionary of radial coordinates.
        :param z_coordinates: Dictionary of vertical coordinates.
        :param domain_params: List of dictionaries containing domain parameters.
        """
        self.r_coordinates = r_coordinates
        self.z_coordinates = z_coordinates
        self.domain_params = domain_params
        self.domain_list = self.make_domain_list()

    def make_domain_list(self) -> Dict[int, 'Domain']:
        """
        Creates a dictionary of Domain objects based on the domain parameters.

        :return: Dictionary of Domain objects with their index as keys.
        """
        domain_list = {}
        # Extract 'a' values from domain_params where 'a' exists
        a_values = [params['a'] for params in self.domain_params if 'a' in params]
        scale = np.mean(a_values) if a_values else 1.0

        for idx, params in enumerate(self.domain_params):
            # Prepare parameters to pass to Domain
            domain_params = {
                'h': self.z_coordinates.get('h', h),
                'di': params.get('di', d[idx] if idx < len(d) else 0.0),
                'a': params.get('a', a[idx] if idx < len(a) else a[-1]),
                'm0': params.get('m0', m0),
                'scale': scale,
                'heaving': params.get('heaving', heaving[idx] if idx < len(heaving) else 0),
                'slant': params.get('slant', False)
            }
            domain = Domain(
                number_harmonics=params.get('number_harmonics', 0),
                height=params.get('height', 0.0),
                radial_width=params.get('radial_width', 0.0),
                top_BC=params.get('top_BC', None),
                bottom_BC=params.get('bottom_BC', None),
                category=params.get('category', ''),
                params=domain_params,
                index=idx
            )
            domain_list[idx] = domain
        return domain_list

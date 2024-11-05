# geometry.py

from typing import Dict, List
from domain import Domain
from constants import h as h_single, d1, d2, a1, a2
from multi_constants import h as h_multi, d as d_multi, a as a_multi, heaving

class Geometry:
    """
    Represents the physical geometry of the problem, including coordinates and domain parameters.
    """

    def __init__(self, r_coordinates: Dict[str, float], z_coordinates: Dict[str, float], domain_params: List[Dict], multi_region=False):
        """
        Initialize the Geometry object.

        :param r_coordinates: Dictionary of radial coordinates.
        :param z_coordinates: Dictionary of vertical coordinates.
        :param domain_params: List of dictionaries containing domain parameters.
        :param multi_region: Boolean indicating if multi-region functionality is enabled.
        """
        self.r_coordinates = r_coordinates
        self.z_coordinates = z_coordinates
        self.domain_params = domain_params
        self.multi_region = multi_region
        self.domain_list = self.make_domain_list()
        

    def make_domain_list(self) -> Dict[int, 'Domain']:
        """
        Creates a dictionary of Domain objects based on the domain parameters.

        :return: Dictionary of Domain objects with their index as keys.
        """
        domain_list = {}
        for idx, params in enumerate(self.domain_params):
            if not self.multi_region:
                # Single-region setup
                domain_params = {
                    'h': self.z_coordinates.get('h', h_single),
                    'di': params.get('di', 0.0),
                    'a1': self.r_coordinates.get('a1', a1),
                    'a2': self.r_coordinates.get('a2', a2),
                    'm0': params.get('m0', 1.0),
                    'radial_width': params.get('radial_width', self.r_coordinates.get('a2', a2) - self.r_coordinates.get('a1', a1)),
                    'slant': params.get('slant', False),
                    'heave': params.get('heave', False),
                }
                domain = Domain(
                    index=idx,
                    number_harmonics=params.get('number_harmonics', 0),
                    height=params.get('height', h_single - (params.get('di') or 0.0)),
                    top_BC=params.get('top_BC', None),
                    bottom_BC=params.get('bottom_BC', None),
                    category=params.get('category', 'inner' if idx == 0 else 'outer'),
                    params=domain_params
                )
            else:
                # Multi-region setup
                domain_params = {
                    'h': h_multi,
                    'di': d_multi[idx],
                    'ai': a_multi[idx],
                    'm0': params.get('m0', 1.0),
                    'slant': params.get('slant', False),
                    'heave': params.get('heave', False),
                    'radial_width': params.get('radial_width', a_multi[idx] - a_multi[idx - 1] if idx > 0 else a_multi[idx]),
                }
                domain = Domain(
                    index=idx,
                    number_harmonics=params.get('number_harmonics', 0),
                    height=h_multi - d_multi[idx],
                    top_BC=params.get('top_BC', None),
                    bottom_BC=params.get('bottom_BC', None),
                    category='multi' if idx < len(self.domain_params) - 1 else 'exterior',
                    params=domain_params
                )
            domain_list[idx] = domain
        return domain_list

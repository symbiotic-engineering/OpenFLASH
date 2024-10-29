# geometry.py

from typing import Dict, List
from domain import Domain

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
        for idx, params in enumerate(self.domain_params):
            # Prepare parameters to pass to Domain
            domain_params = {
                'h': self.z_coordinates.get('h', 1.0),
                'di': params.get('di', 0.0),
                'a1': self.r_coordinates.get('a1', 0.5),
                'a2': self.r_coordinates.get('a2', 1.0),
                'm0': params.get('m0', 1.0),
            }
            domain = Domain(
                number_harmonics=params.get('number_harmonics', 0),
                height=params.get('height', 0.0),
                radial_width=params.get('radial_width', 0.0),
                top_BC=params.get('top_BC', None),
                bottom_BC=params.get('bottom_BC', None),
                category=params.get('category', ''),
                params=domain_params
            )
            domain_list[idx] = domain
        return domain_list

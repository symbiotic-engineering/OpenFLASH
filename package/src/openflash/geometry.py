from typing import Dict, List
import numpy as np
from openflash.domain import Domain

class Geometry:
    """
    Represents the physical geometry of the problem, including coordinates and domain parameters.
    """

    def __init__(self, r_coordinates: Dict[str, float], z_coordinates: Dict[str, float], domain_params: List[Dict]):
        """
        Initialize the Geometry object.

        Parameters
        ----------
        r_coordinates : dict
            Radial coordinates of the system.
        z_coordinates : dict
            Vertical coordinates (must contain 'h').
        domain_params : list of dict
            List of domain parameter dictionaries.
        """
        self.r_coordinates = r_coordinates
        self.z_coordinates = z_coordinates
        self.domain_params = domain_params
        self.domain_list = self.make_domain_list()

    def make_domain_list(self) -> Dict[int, Domain]:
        """
        Constructs the domain list using shared geometric properties.

        Returns
        -------
        dict
            Dictionary of domain index to Domain object.
        """
        domain_list = {}

        a_values = [params['a'] for params in self.domain_params if 'a' in params and params['a'] is not None]
        scale = np.mean(a_values) if a_values else 1.0

        h = self.z_coordinates.get('h')
        if h is None:
            raise ValueError("z_coordinates must contain key 'h'")

        for idx, params in enumerate(self.domain_params):
            category = params.get('category')
            di = params.get('di', None)
            a = params.get('a', None)

            if category != 'exterior':
                if di is None:
                    raise ValueError(f"domain_params[{idx}] missing required 'di'")
                if a is None:
                    raise ValueError(f"domain_params[{idx}] missing required 'a'")

            domain = Domain(
                number_harmonics=params.get('number_harmonics', 0),
                height=params.get('height', 0.0),
                radial_width=params.get('radial_width', 0.0),
                top_BC=params.get('top_BC'),
                bottom_BC=params.get('bottom_BC'),
                category=category,
                params={
                    'h': h,
                    'di': di,
                    'a': a,
                    'scale': scale,
                    'heaving': params.get('heaving'),
                    'slant': params.get('slant', False)
                },
                index=idx,
                geometry=self
            )

            domain_list[idx] = domain

        return domain_list

    @property
    def adjacency_matrix(self) -> np.ndarray:
        """
        Boolean matrix indicating adjacency between domains.

        Returns
        -------
        np.ndarray
            Matrix where entry (i, j) is True if domains i and j are adjacent.
        """
        n = len(self.domain_list)
        adj = np.zeros((n, n), dtype=bool)

        for i, domain_i in self.domain_list.items():
            for j, domain_j in self.domain_list.items():
                if i == j:
                    continue
                a_i = domain_i.a
                a_j = domain_j.a
                if a_i is not None and a_j is not None and np.isclose(a_i, a_j, atol=1e-6):
                    adj[i, j] = True

        return adj
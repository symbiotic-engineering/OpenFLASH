import numpy as np
from openflash.geometry import Geometry

class MEEMProblem:
    """
    Encapsulates the full configuration and computation targets for a MEEM scenario.
    """

    def __init__(self, geometry: Geometry):
        """
        Initialize a MEEMProblem instance.

        Parameters
        ----------
        geometry : Geometry
            The full system geometry, including all domains.
        """
        self.geometry = geometry
        self.domain_list = geometry.domain_list
        self.frequencies = np.array([])  # Angular frequencies ω (rad/s)
        self.modes = np.array([])        # Mode numbers (e.g. [0, 1, 2])

    def set_frequencies_modes(self, frequencies: np.ndarray, modes: np.ndarray):
        """
        Set the angular frequencies and motion modes for this problem.

        Parameters
        ----------
        frequencies : np.ndarray
            Array of angular frequencies (rad/s).
        modes : np.ndarray
            Array of mode indices.
        """
        assert isinstance(modes, np.ndarray), "modes must be a numpy array"
        assert np.all(modes > 0), "All mode numbers (m0) must be positive integers"
        assert np.issubdtype(modes.dtype, np.integer), "Mode numbers must be integers"

        self.frequencies = frequencies
        self.modes = modes

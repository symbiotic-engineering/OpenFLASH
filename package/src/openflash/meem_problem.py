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
        Set the angular frequencies and degrees of freedom for this problem.

        Parameters
        ----------
        frequencies : np.ndarray
            Array of angular frequencies or omega (rad/s).
        modes : np.ndarray
            Array of degrees of freedom indices.
        """
        assert isinstance(modes, np.ndarray), "modes must be a numpy array"
        assert np.all(frequencies > 0), "All frequencies must be positive"
        assert isinstance(frequencies, np.ndarray), "frequencies must be a numpy array"
        assert np.all(modes >= 0), "modes must be non-negative integers"

        self.frequencies = frequencies
        self.modes = modes

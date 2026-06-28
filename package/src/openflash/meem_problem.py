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

    def set_frequencies(self, frequencies: np.ndarray):
        """
        Set the angular frequencies for this problem.

        Parameters
        ----------
        frequencies : np.ndarray
            Array of angular frequencies or omega (rad/s).
        """
        assert np.all(frequencies > 0), "All frequencies must be positive"
        assert isinstance(frequencies, np.ndarray), "frequencies must be a numpy array"

        self.frequencies = frequencies

    @property
    def modes(self) -> np.ndarray:
        """
        Infers the active modes (degrees of freedom) from the
        heaving flags set on the geometry's bodies.
        
        This assumes the body index (0, 1, 2...) corresponds
        to the mode index.
        """
        # Assumes bodies are SteppedBody or CoordinateBody
        heaving_bodies = [
            i for i, body in enumerate(self.geometry.body_arrangement.bodies)
            if body.heaving
        ]
        return np.array(heaving_bodies)
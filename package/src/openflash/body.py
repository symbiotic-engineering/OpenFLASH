# body.py

from abc import ABC
from typing import Tuple
import numpy as np

class Body(ABC):
    """
    Abstract base class for a physical body.
    """
    heaving: bool

class SteppedBody(Body):
    """
    Represents a body defined by a series of concentric, vertical-walled steps.

    This is the primary implementation for the initial JOSS scope.

    Args:
        a (np.ndarray): Array of outer radii for each step.
        d (np.ndarray): Array of values (depth) for each step.
        slant_angle (np.ndarray): Array of slant angles for each step surface.
        heaving (bool, optional): Flag indicating if the entire body is heaving. Defaults to False.
    """
    def __init__(self, a: np.ndarray, d: np.ndarray, slant_angle: np.ndarray, heaving: bool = False):
        assert len(a) == len(d) == len(slant_angle), "Input arrays a, d, and slant_angle must have the same length."
        self.a = a
        self.d = d
        self.slant_angle = slant_angle
        self.heaving = heaving

class CoordinateBody(Body):
    """
    Represents a body defined by a series of (r, z) coordinates.

    This class is a placeholder for future functionality and requires a
    discretization method to be used in calculations.

    Args:
        r_coords (np.ndarray): Array of radial coordinates.
        z_coords (np.ndarray): Array of vertical coordinates (depth).
        heaving (bool, optional): Flag indicating if the body is heaving. Defaults to False.
    """
    def __init__(self, r_coords: np.ndarray, z_coords: np.ndarray, heaving: bool = False):
        assert len(r_coords) == len(z_coords), "r_coords and z_coords must be the same length."
        self.r_coords = r_coords
        self.z_coords = z_coords
        self.heaving = heaving

    def discretize(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        (Future Implementation) Converts r, z coordinates into stepped a, d, slant_angle arrays.
        """
        # NOTE: This is a placeholder discretization. A more robust method is needed.
        a = self.r_coords
        d = self.z_coords
        slant = np.gradient(d, a)  # Simple slope estimate
        return a, d, slant
# geometry.py

from abc import ABC, abstractmethod
from typing import List, Sequence
import numpy as np

from .body import Body, SteppedBody, CoordinateBody
from .domain import Domain

class BodyArrangement(ABC):
    """
    Abstract base class for any arrangement.
    """
    def __init__(self, bodies: Sequence[Body]):
        self.bodies = bodies
        # Count the number of bodies marked as heaving (heaving=True)
        heaving_count = sum(body.heaving for body in bodies)
        
        assert heaving_count <= 1, \
            f"Only 0 or 1 body can be marked as heaving. Found {heaving_count} heaving bodies."

    @property
    @abstractmethod
    def a(self) -> np.ndarray:
        """Array of characteristic radii."""
        pass

    @property
    @abstractmethod
    def d(self) -> np.ndarray:
        """Array of characteristic d."""
        pass

    @property
    @abstractmethod
    def slant_angle(self) -> np.ndarray:
        """Array of slant angles."""
        pass

    @property
    @abstractmethod
    def heaving(self) -> np.ndarray:
        """Array of heaving flags."""
        pass


class ConcentricBodyGroup(BodyArrangement):
    """
    A concrete arrangement of one or more concentric bodies.
    For JOSS, this class assumes all bodies are SteppedBody objects.
    """
    def __init__(self, bodies: Sequence[Body]):
        super().__init__(bodies)
        # For now, we only handle SteppedBody
        for body in self.bodies:
            if not isinstance(body, SteppedBody):
                raise TypeError("ConcentricBodyGroup currently only supports SteppedBody objects.")

    def _get_concatenated_property(self, prop_name: str) -> np.ndarray:
        """Helper to concatenate a property from all SteppedBody objects."""
        return np.concatenate([getattr(body, prop_name) for body in self.bodies])

    def _get_heaving_flags(self) -> np.ndarray:
        """Helper to create a heaving flag array based on each body."""
        flags = []
        for body in self.bodies:
            # --- THIS IS THE FIX ---
            if isinstance(body, SteppedBody):
                num_steps = len(body.a)
                flags.extend([body.heaving] * num_steps)
            # You could add 'elif isinstance(body, CoordinateBody):' here if needed
            # --- END FIX ---
        return np.array(flags, dtype=bool)

    @property
    def a(self) -> np.ndarray:
        return self._get_concatenated_property('a')

    @property
    def d(self) -> np.ndarray:
        return self._get_concatenated_property('d')

    @property
    def slant_angle(self) -> np.ndarray:
        return self._get_concatenated_property('slant_angle')

    @property
    def heaving(self) -> np.ndarray:
        return self._get_heaving_flags()


class Geometry(ABC):
    """
    Abstract base class for a complete problem geometry.

    A Geometry consists of a BodyArrangement and the total water depth, and
    it is responsible for creating the corresponding fluid domains.
    """
    def __init__(self, body_arrangement: BodyArrangement, h: float):
        self.body_arrangement = body_arrangement
        self.h = h
        self._fluid_domains: List[Domain] = []

    @property
    def fluid_domains(self) -> List[Domain]:
        if not self._fluid_domains:
            self._fluid_domains = self.make_fluid_domains()
        return self._fluid_domains

    # --- ADD THIS PROPERTY ---
    @property
    def domain_list(self) -> dict:
        """
        Returns a dictionary of domains keyed by index.
        Required for MEEMProblem/Results compatibility.
        """
        # If the property isn't overridden, create the dict from the list.
        if not self.fluid_domains:
            return {}
        # This check handles if a subclass (like BasicRegionGeometry)
        # has already provided a dict.
        if isinstance(self.fluid_domains, dict):
             return self.fluid_domains
        return {domain.index: domain for domain in self.fluid_domains}
    # --- END ADDITION ---

    @abstractmethod
    def make_fluid_domains(self) -> List[Domain]:
        """Creates the list of Domain objects from the BodyArrangement."""
        pass
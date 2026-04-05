# domain.py

from typing import Optional
import numpy as np

class Domain:
    """
    Represents a single, circular region of the fluid.

    This class stores the geometric boundaries and physical properties for a
    specific domain within the overall problem geometry. The upper boundary depth
    is determined from the geometry and cannot be set independently.
    """
    def __init__(self,
                 index: int,
                 NMK: int,
                 a_inner: float,
                 a_outer: float,
                 d_lower: float,
                 geometry_h: float,
                 heaving: Optional[bool] = None,
                 slant: bool = False,
                 category: str = "interior"):

        self.index = index
        self.number_harmonics = NMK
        self.a_inner = a_inner
        self.a_outer = a_outer
        self.d_lower = d_lower # Depth of the lower boundary (e.g., body or seafloor)
        self.d_upper = geometry_h # Depth of the upper boundary (e.g., free surface)
        self.heaving = heaving
        self.slant = slant
        self.category = category # e.g., 'interior', 'exterior'

        # --- Assertions ---
        assert isinstance(NMK, int) and NMK > 0, "NMK must be a positive integer."
        assert a_outer > a_inner >= 0, "Radii must be valid (a_outer > a_inner >= 0)."
        assert self.d_upper >= d_lower >= 0, "Depths must be valid (d_upper >= d_lower >= 0)."

    @property
    def h(self):
        """
        Return the total water depth (free surface height),
        which MEEMEngine expects as .h on domain objects.
        """
        return self.d_upper
    
    @property
    def di(self):
        """
        Lower boundary depth of the domain (used internally in MEEMEngine as .di).
        """
        return self.d_lower
    
    @property
    def a(self):
        """
        Alias for outer radius (used internally by MEEMEngine as .a).
        """
        return self.a_outer
    
    @staticmethod
    def are_adjacent(d1: "Domain", d2: "Domain", atol: float = 1e-6) -> bool:
        """
        (Future Implementation) Determines if two domains are radially adjacent.
        """
        # Check if d1 is inside d2
        if np.isfinite(d1.a_outer) and np.isclose(d1.a_outer, d2.a_inner, atol=atol):
            return True
        # Check if d2 is inside d1
        if np.isfinite(d2.a_outer) and np.isclose(d2.a_outer, d1.a_inner, atol=atol):
            return True
        return False
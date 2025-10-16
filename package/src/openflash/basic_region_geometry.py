# basic_region_geometry.py

from typing import List, Optional
import numpy as np

from .geometry import Geometry, ConcentricBodyGroup
from .body import SteppedBody
from .domain import Domain

class BasicRegionGeometry(Geometry):
    """
    A geometry where body radii are increasing from the center.

    This configuration results in a simple, non-overlapping series of circular
    fluid domains, where the mapping from bodies to domains is trivial.

    Args:
        body_arrangement (ConcentricBodyGroup): A group of concentric bodies.
        h (float): The total water depth.
        NMK (List[int]): List of the number of harmonics for each resulting domain.
    """
    def __init__(self, body_arrangement: ConcentricBodyGroup, h: float, NMK: List[int]):
        super().__init__(body_arrangement, h)
        self.NMK = NMK

        # --- Assertions ---
        # Verify that radii are strictly and increasing.
        all_radii = self.body_arrangement.a
        if not np.all(np.diff(all_radii) > 0):
            raise ValueError("Radii 'a' must be strictly increasing for BasicRegionGeometry. Use AnyRegionGeometry for other cases.")
        # Verify NMK has the correct length (one for each body segment + one for the exterior).
        if len(NMK) != len(all_radii) + 1:
            raise ValueError("Length of NMK must be one greater than the total number of body radii.")
        # FIX 1: Generate the domains and store the final dictionary ONCE during initialization.
        # The old line was: self._domain_list: List[Domain] = self.make_fluid_domains()
        domains_as_list = self.make_fluid_domains()
        self._domain_dict: dict = {domain.index: domain for domain in domains_as_list}

    @property
    def domain_list(self) -> dict:
        """
        Returns a dictionary of domains keyed by index.
        Required for MEEMEngine compatibility.
        """
        # FIX 2: Simply return the dictionary created during __init__. Do not recalculate.
        return self._domain_dict
    
    @classmethod
    def from_vectors(cls,
                     a: np.ndarray,
                     d: np.ndarray,
                     h: float,
                     NMK: List[int],
                     slant_angle: Optional[np.ndarray] = None,
                     body_map: Optional[List[int]] = None,
                     heaving_map: Optional[List[bool]] = None):
        """
        Method to create a BasicRegionGeometry from vector inputs.
        This is useful for users who prefer to define the geometry directly
        without explicitly creating Body objects. This version includes robust
        validation to prevent invalid body_map/heaving_map combinations and
        enforces the global monotonicity invariant required by BasicRegionGeometry.
        """
        if slant_angle is None:
            slant_angle = np.zeros_like(a)

        if body_map is None:
            body_map = [0] * len(a)

        # Determine number of bodies from the mapping
        num_bodies = max(body_map) + 1

        # Validate heaving_map length
        if heaving_map is None:
            heaving_map = [False] * num_bodies
        elif len(heaving_map) != num_bodies:
            raise ValueError(
                f"Length of heaving_map ({len(heaving_map)}) does not match inferred number of bodies ({num_bodies})"
            )

        # Build radii groups based on body_map and check contiguity + global monotonicity
        reconstructed = []
        last_value = -np.inf  # Start with negative infinity for strict monotonicity check

        for body_idx in range(num_bodies):
            # Extract indices for this body in original order
            indices = [j for j, idx in enumerate(body_map) if idx == body_idx]
            if not indices:
                raise ValueError(f"Body index {body_idx} is declared in body_map but has no assigned radii.")

            body_radii = a[indices]

            # Local monotonicity inside body is not required, but when flattened back,
            # the entire vector must be strictly increasing to satisfy BasicRegionGeometry rules.
            for r in body_radii:
                if r <= last_value:
                    raise ValueError(
                        "Radii 'a' must be strictly increasing after applying body_map. "
                        "Detected non-monotonic or backtracking group arrangement."
                    )
                last_value = r

            reconstructed.append(body_radii)

        # Now safe to construct bodies
        bodies = []
        for body_idx in range(num_bodies):
            indices = [j for j, idx in enumerate(body_map) if idx == body_idx]
            bodies.append(SteppedBody(
                a=a[indices],
                d=d[indices],
                slant_angle=slant_angle[indices],
                heaving=heaving_map[body_idx]
            ))

        arrangement = ConcentricBodyGroup(bodies)
        return cls(arrangement, h, NMK)

    def make_fluid_domains(self) -> List[Domain]:
        """
        Creates a list of fluid domains for the simple concentric case.
        """
        domains: List[Domain] = []
        last_outer_radius = 0.0

        arr = self.body_arrangement
        all_radii = arr.a
        all_d = arr.d
        all_heaving = arr.heaving
        all_slants = arr.slant_angle

        # Create interior domains under the bodies
        for i, (outer_r, d, is_heaving, is_slanted) in enumerate(zip(all_radii, all_d, all_heaving, all_slants)):
            domain = Domain(
                index=i,
                NMK=self.NMK[i],
                a_inner=last_outer_radius,
                a_outer=outer_r,
                d_lower=d,
                geometry_h=self.h, # pass the geometry's total depth
                heaving=is_heaving,
                slant=bool(is_slanted),
                category="interior"
            )
            domains.append(domain)
            last_outer_radius = outer_r

        # Create final exterior domain
        domains.append(Domain(
            index=len(all_radii),
            NMK=self.NMK[-1],
            a_inner=last_outer_radius,
            a_outer=np.inf,
            d_lower=0.0, # Seabed
            geometry_h=self.h,
            category="exterior"
        ))

        return domains
# test_basic_region_geometry.py

import pytest
import numpy as np
import sys
import os
from typing import List  # Added for type hinting

# make src importable (adjust path if your repository layout differs)
current_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(current_dir, '..', 'src'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from openflash.basic_region_geometry import BasicRegionGeometry
from openflash.body import SteppedBody, Body  # Added Body for type hinting
from openflash.geometry import ConcentricBodyGroup

# ------------------------------
# Helper utilities used in randomized tests
# ------------------------------
def contiguous_body_map(num_segments: int, num_bodies: int):
    """
    Create a body_map that maps contiguous segments to bodies.
    This guarantees that concatenating segments by body preserves the
    strictly increasing order of radii when `a` is strictly increasing.
    """
    assert 1 <= num_bodies <= num_segments
    # compute split sizes (distribute segments across bodies)
    base = num_segments // num_bodies
    extras = num_segments % num_bodies
    sizes = [base + (1 if i < extras else 0) for i in range(num_bodies)]
    bm = []
    cur = 0
    for body_idx, sz in enumerate(sizes):
        bm.extend([body_idx] * sz)
        cur += sz
    return bm

# Small positive epsilon to use when we want "near zero" radii but
# still respect Domain's assertion a_outer > a_inner >= 0
_EPS = 1e-6

# ------------------------------
# Deterministic parametrized valid initialization
# ------------------------------
@pytest.mark.parametrize(
    "radii_list, NMK",
    [
        ([np.array([1.0]), np.array([2.0])], [5, 6, 7]),
        ([np.array([0.5]), np.array([1.5]), np.array([3.0])], [2, 3, 4, 5]),
        ([np.array([1.0])], [10, 20]),
    ]
)
def test_basic_region_geometry_init_valid_param(radii_list, NMK):
    # FIX: Explicitly type hint as List[Body] to satisfy invariance
    bodies: List[Body] = [
        SteppedBody(a=r, d=np.array([1.0] * len(r)), slant_angle=np.zeros_like(r), heaving=False)
        for r in radii_list
    ]
    arrangement = ConcentricBodyGroup(bodies)
    geom = BasicRegionGeometry(arrangement, h=10.0, NMK=NMK)

    assert geom.NMK == NMK
    assert geom.h == 10.0
    assert isinstance(geom.body_arrangement, ConcentricBodyGroup)

# ------------------------------
# Deterministic parametrized invalid radii (not strictly increasing)
# ------------------------------
@pytest.mark.parametrize(
    "radii_list",
    [
        ([np.array([2.0]), np.array([1.0])]),
        ([np.array([1.0]), np.array([1.0])]),
    ]
)
def test_basic_region_geometry_init_invalid_radii_param(radii_list):
    # FIX: Explicitly type hint as List[Body]
    bodies: List[Body] = [
        SteppedBody(a=r, d=np.array([1.0] * len(r)), slant_angle=np.zeros_like(r), heaving=False)
        for r in radii_list
    ]
    arrangement = ConcentricBodyGroup(bodies)
    NMK = [1] * (len(radii_list) + 1)

    with pytest.raises(ValueError, match="Radii 'a' must be strictly increasing"):
        BasicRegionGeometry(arrangement, h=10.0, NMK=NMK)

# ------------------------------
# Deterministic parametrized invalid NMK length
# ------------------------------
@pytest.mark.parametrize(
    "radii_list, NMK",
    [
        ([np.array([1.0])], [5]),         # Should be length 2
        ([np.array([1.0, 2.0])], [1, 2]), # Should be length 3
    ]
)
def test_basic_region_geometry_init_invalid_nmk_length_param(radii_list, NMK):
    # FIX: Explicitly type hint as List[Body]
    bodies: List[Body] = [
        SteppedBody(a=r, d=np.array([1.0] * len(r)), slant_angle=np.zeros_like(r), heaving=False)
        for r in radii_list
    ]
    arrangement = ConcentricBodyGroup(bodies)

    with pytest.raises(ValueError, match="Length of NMK must be one greater"):
        BasicRegionGeometry(arrangement, h=10.0, NMK=NMK)

# ------------------------------
# Deterministic make_fluid_domains tests
# ------------------------------
@pytest.mark.parametrize(
    "a, d, NMK",
    [
        (np.array([1.0, 2.0]), np.array([2.0, 3.0]), [5, 6, 7]),
        (np.array([0.5, 1.5, 3.0]), np.array([1.0, 2.0, 3.0]), [2, 3, 4, 5]),
    ]
)
def test_make_fluid_domains_param(a, d, NMK):
    geom = BasicRegionGeometry.from_vectors(a, d, h=10.0, NMK=NMK)
    domains = geom.make_fluid_domains()

    assert len(domains) == len(a) + 1

    last_outer = 0.0
    for i, dom in enumerate(domains[:-1]):
        assert dom.index == i
        assert dom.category == "interior"
        assert dom.a_inner == last_outer
        assert dom.a_outer == a[i]
        last_outer = a[i]

    exterior = domains[-1]
    assert exterior.category == "exterior"
    assert exterior.a_inner == last_outer
    assert exterior.a_outer == np.inf

# ------------------------------
# Deterministic from_vectors with body_map and heaving_map
# ------------------------------
@pytest.mark.parametrize(
    "a, d, NMK, body_map, heaving_map, expected_num_bodies",
    [
        (np.array([1.0, 2.0]), np.array([2.0, 3.0]), [5, 6, 7], None, None, 1),
        (np.array([1.0, 2.0, 3.0]),
         np.array([1.0, 2.0, 3.0]),
         [2, 3, 4, 5],
         [0, 1, 1],
         [False, True], # One heaving body
         2),
        # FIXED: Changed heaving_map to have only one True to comply with the new assertion
        (np.array([0.5, 1.5, 3.0, 4.0]),
         np.array([0.5, 1.5, 2.5, 3.5]),
         [1, 2, 3, 4, 5],
         [0, 1, 2, 2],
         [True, False, False], # Changed from [True, True, True] to valid one heaving body
         3)
    ]
)
def test_from_vectors_body_and_heaving(a, d, NMK, body_map, heaving_map, expected_num_bodies):
    geom = BasicRegionGeometry.from_vectors(a, d, h=10.0, NMK=NMK,
                                            body_map=body_map, heaving_map=heaving_map)

    assert len(geom.body_arrangement.bodies) == expected_num_bodies

    for i, body in enumerate(geom.body_arrangement.bodies):
        if heaving_map is not None:
            assert body.heaving == heaving_map[i]
        else:
            assert body.heaving == False

    # --- FIX: Use the 'a' property of the arrangement instead of manually iterating the list ---
    # This solves the Pylance error regarding 'b.a' being inaccessible on base class 'Body'
    combined_radii = geom.body_arrangement.a
    np.testing.assert_array_equal(combined_radii, a)

# ------------------------------
# NEW TEST: Invalid heaving map (more than one heaving body)
# ------------------------------
def test_from_vectors_invalid_heaving_count():
    a = np.array([1.0, 2.0, 3.0])
    d = np.array([1.0, 2.0, 3.0])
    NMK = [5, 5, 5, 5]
    body_map = [0, 1, 2] # 3 separate bodies
    heaving_map_bad = [True, True, False] # Two heaving bodies (violates rule)
    
    with pytest.raises(AssertionError, match="Only 0 or 1 body can be marked as heaving"):
        BasicRegionGeometry.from_vectors(a, d, h=10.0, NMK=NMK,
                                         body_map=body_map, heaving_map=heaving_map_bad)

# ------------------------------
# Edge-case tests (near-zero positive radius, single-element, large NMK)
# ------------------------------
@pytest.mark.parametrize(
    "a, d, NMK",
    [
        # Use a tiny positive first radius instead of 0.0 to satisfy Domain assertion
        (np.array([_EPS, 1.0]), np.array([1.0, 2.0]), [2, 3, 4]),
        (np.array([2.0]), np.array([3.0]), [5, 6]),
        (np.array([1.0, 2.0]), np.array([1.0, 2.0]), [10000, 20000, 30000]),
    ]
)
def test_edge_cases(a, d, NMK):
    geom = BasicRegionGeometry.from_vectors(a, d, h=10.0, NMK=NMK)
    domains = geom.make_fluid_domains()

    assert len(domains) == len(a) + 1

    last_outer = 0.0
    for i, dom in enumerate(domains[:-1]):
        assert dom.index == i
        assert dom.a_inner == last_outer
        assert dom.a_outer == a[i]
        last_outer = a[i]

    exterior = domains[-1]
    assert exterior.a_outer == np.inf
    assert exterior.index == len(a)

# ------------------------------
# Randomized valid stress tests (guarantee strictly increasing radii and contiguous body_map)
# ------------------------------
@pytest.mark.parametrize("seed", range(10))
def test_randomized_stress_valid(seed):
    np.random.seed(seed)
    num_segments = np.random.randint(1, 6)  # 1-5 segments

    # Strictly increasing radii by cumulative sum of positive deltas
    deltas = np.random.uniform(0.1, 2.0, size=num_segments)
    a = np.cumsum(deltas)  # ensures a[0] > 0 and strictly increasing
    d = np.random.uniform(1.0, 5.0, size=num_segments)

    # NMK length must be num_segments + 1 and positive ints
    NMK = np.random.randint(1, 100, size=num_segments + 1).astype(int).tolist()

    # Choose a random number of bodies that partitions segments into contiguous blocks
    num_bodies = np.random.randint(1, num_segments + 1)
    body_map = contiguous_body_map(num_segments, num_bodies)
    
    # Ensure only 0 or 1 body is heaving
    heaving_map = [False] * num_bodies
    if num_bodies > 0 and np.random.rand() < 0.5: # 50% chance to set one body to heaving
        heaving_map[np.random.randint(0, num_bodies)] = True
    
    geom = BasicRegionGeometry.from_vectors(a, d, h=10.0, NMK=NMK,
                                            body_map=body_map, heaving_map=heaving_map)
    domains = geom.make_fluid_domains()

    # Basic checks and invariants
    assert len(domains) == len(a) + 1

    last_outer = 0.0
    for i, dom in enumerate(domains[:-1]):
        # non-overlap and monotonicity invariants
        assert dom.a_inner == last_outer
        assert dom.a_outer > last_outer
        last_outer = dom.a_outer

    # exterior domain invariants
    exterior = domains[-1]
    assert exterior.a_inner == last_outer
    assert exterior.a_outer == np.inf

# ------------------------------
# Randomized invalid tests (explicitly force non-monotonic maps or duplicate radii)
# ------------------------------
@pytest.mark.parametrize("seed", range(5))
def test_randomized_stress_invalid(seed):
    np.random.seed(seed)
    num_segments = np.random.randint(2, 6)  # need at least 2 to cause duplicate/ordering problems

    # Create strictly increasing a first, then force a duplicate to provoke ValueError
    deltas = np.random.uniform(0.1, 2.0, size=num_segments)
    a = np.cumsum(deltas)
    # force a duplicate to violate strict increase
    a[1] = a[0]

    d = np.random.uniform(1.0, 5.0, size=num_segments)
    NMK = np.random.randint(1, 100, size=num_segments + 1).tolist()

    # from_vectors should raise because radii are not strictly increasing
    with pytest.raises(ValueError, match="Radii 'a' must be strictly increasing"):
        BasicRegionGeometry.from_vectors(a, d, h=10.0, NMK=NMK)

    # Another invalid scenario: non-contiguous body_map that reorders segments.
    # Build a valid strictly increasing 'a' again and then create a non-contiguous map:
    deltas2 = np.random.uniform(0.1, 2.0, size=num_segments)
    a2 = np.cumsum(deltas2)
    d2 = np.random.uniform(1.0, 5.0, size=num_segments)
    NMK2 = np.random.randint(1, 100, size=num_segments + 1).tolist()

    # create a body_map that interleaves bodies (e.g., 0,1,0,1...)
    if num_segments >= 3:
        body_map_bad = [(i % 2) for i in range(num_segments)]
        # If num_segments is small, this map can cause concatenated radii to be non-monotonic.
        # from_vectors should detect and raise ValueError
        with pytest.raises(ValueError, match="Radii 'a' must be strictly increasing"):
            BasicRegionGeometry.from_vectors(a2, d2, h=10.0, NMK=NMK2, body_map=body_map_bad)

# ------------------------------
# Randomized extreme NMK stress tests (extremely large NMK counts but valid shapes)
# ------------------------------
@pytest.mark.parametrize("seed", range(5))
def test_randomized_extreme_nmk(seed):
    np.random.seed(seed)
    num_segments = np.random.randint(1, 6)

    # Strictly increasing radii
    deltas = np.random.uniform(0.1, 2.0, size=num_segments)
    a = np.cumsum(deltas)
    d = np.random.uniform(1.0, 5.0, size=num_segments)

    # Generate very large NMK values (but keep count correct)
    # Use values in the tens/hundreds of thousands to simulate pressure on code paths that store them
    NMK = np.random.randint(10_000, 200_000, size=num_segments + 1).astype(int).tolist()

    # Use contiguous body_map to preserve monotonicity
    num_bodies = np.random.randint(1, num_segments + 1)
    body_map = contiguous_body_map(num_segments, num_bodies)
    
    # Ensure only 0 or 1 body is heaving
    heaving_map = [False] * num_bodies
    if num_bodies > 0 and np.random.rand() < 0.5:
        heaving_map[np.random.randint(0, num_bodies)] = True

    geom = BasicRegionGeometry.from_vectors(a, d, h=10.0, NMK=NMK,
                                            body_map=body_map, heaving_map=heaving_map)
    domains = geom.make_fluid_domains()

    # Confirm invariants still hold with huge NMK numbers
    assert len(domains) == len(a) + 1
    last_outer = 0.0
    for dom in domains[:-1]:
        assert dom.a_inner == last_outer
        assert dom.a_outer > last_outer
        last_outer = dom.a_outer
    assert domains[-1].a_outer == np.inf
    
# ------------------------------
# NEW TEST: Heaving map length mismatch
# ------------------------------
def test_from_vectors_heaving_map_length_mismatch():
    """
    Tests that ValueError is raised when len(heaving_map) != number of unique bodies inferred
    from body_map (which is max(body_map) + 1).
    """
    a = np.array([1.0, 2.0])
    d = np.array([1.0, 1.0])
    NMK = [5, 5, 5]
    
    # body_map=[0, 1] implies 2 bodies (Body 0 and Body 1)
    body_map = [0, 1] 
    
    # heaving_map has length 1, but we expect length 2
    heaving_map_short = [False]
    
    with pytest.raises(ValueError, match=r"Length of heaving_map \(1\) does not match inferred number of bodies \(2\)"):
        BasicRegionGeometry.from_vectors(a, d, h=10.0, NMK=NMK,
                                         body_map=body_map, heaving_map=heaving_map_short)

# ------------------------------
# NEW TEST: Body declared but has no radii (skipped index)
# ------------------------------
def test_from_vectors_missing_body_radii():
    """
    Tests that ValueError is raised when a body index is implied by the range 
    of body_map (0 to max) but no segments are assigned to it.
    """
    a = np.array([1.0, 2.0])
    d = np.array([1.0, 1.0])
    NMK = [5, 5, 5]
    
    # body_map=[0, 2] implies max index is 2, so the code infers 3 bodies: 0, 1, and 2.
    # However, no segment is assigned to Body 1.
    body_map = [0, 2]
    
    # We provide a valid heaving map for 3 bodies to ensure we don't hit the length error first
    heaving_map = [False, False, False]

    with pytest.raises(ValueError, match="Body index 1 is declared in body_map but has no assigned radii"):
        BasicRegionGeometry.from_vectors(a, d, h=10.0, NMK=NMK,
                                         body_map=body_map, heaving_map=heaving_map)
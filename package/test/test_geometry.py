# test_geometry.py
import pytest
import numpy as np
import os as os
import sys as sys
from typing import Dict, List, cast

current_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(current_dir, '..', 'src'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
from openflash.body import SteppedBody, CoordinateBody, Body
from openflash.geometry import ConcentricBodyGroup, Geometry, BodyArrangement
from openflash.domain import Domain

# -------------------------
# Fixtures
# -------------------------
@pytest.fixture
def simple_stepped_body():
    a = np.array([1.0, 2.0])
    d = np.array([0.5, 1.0])
    slant = np.array([0.0, 0.1])
    # Set to False to ensure other tests don't fail by default
    return SteppedBody(a, d, slant, heaving=False)

@pytest.fixture
def concentric_group(simple_stepped_body):
    # This group only has one body, so it's valid
    return ConcentricBodyGroup([simple_stepped_body])

# -------------------------
# Helper Class for Mocking
# -------------------------
class MockBodyArrangement(BodyArrangement):
    """
    A minimal concrete implementation of BodyArrangement to satisfy type checking
    where the actual arrangement logic is not needed for the test.
    """
    @property
    def a(self) -> np.ndarray:
        return np.array([])

    @property
    def d(self) -> np.ndarray:
        return np.array([])

    @property
    def slant_angle(self) -> np.ndarray:
        return np.array([])

    @property
    def heaving(self) -> np.ndarray:
        return np.array([])

@pytest.fixture
def mock_arrangement():
    # FIX: Pass empty list [] because BodyArrangement.__init__ requires 'bodies'
    return MockBodyArrangement([])

# -------------------------
# ConcentricBodyGroup tests
# -------------------------
def test_concatenated_properties(concentric_group):
    # This test uses the simple_stepped_body fixture which is heaving=False
    body = concentric_group.bodies[0]
    np.testing.assert_array_equal(concentric_group.a, body.a)
    np.testing.assert_array_equal(concentric_group.d, body.d)
    np.testing.assert_array_equal(concentric_group.slant_angle, body.slant_angle)
    # Since fixture is heaving=False, heaving array should be all False
    np.testing.assert_array_equal(concentric_group.heaving, np.array([False, False]))

def test_invalid_body_type():
    with pytest.raises(TypeError):
        # CoordinateBody is not currently supported by ConcentricBodyGroup
        ConcentricBodyGroup([CoordinateBody(np.array([0,1]), np.array([0,1]))])

# -------------------------
# NEW TEST: Invalid heaving count
# -------------------------
def test_invalid_heaving_count_init():
    body1 = SteppedBody(np.array([1.0]), np.array([1.0]), np.array([0.0]), heaving=True)
    body2 = SteppedBody(np.array([2.0]), np.array([2.0]), np.array([0.0]), heaving=True)
    
    # This must raise an AssertionError because two bodies are heaving
    with pytest.raises(AssertionError, match="Only 0 or 1 body can be marked as heaving"):
        ConcentricBodyGroup([body1, body2])
        
# -------------------------
# Geometry abstract tests
# -------------------------
class DummyGeometry(Geometry):
    """Concrete implementation for testing Geometry base class."""
    def make_fluid_domains(self):
        # Create a simple domain per body step
        domains = []
        last_r = 0.0
        for i, (a, d, h_flag, sl) in enumerate(zip(
            self.body_arrangement.a,
            self.body_arrangement.d,
            self.body_arrangement.heaving,
            self.body_arrangement.slant_angle
        )):
            domains.append(Domain(
                index=i,
                NMK=1,
                a_inner=last_r,
                a_outer=a,
                d_lower=d,
                geometry_h=self.h,
                heaving=h_flag,
                slant=bool(sl),
                category="interior"
            ))
            last_r = a
        # Add exterior domain
        domains.append(Domain(
            index=len(self.body_arrangement.a),
            NMK=1,
            a_inner=last_r,
            a_outer=np.inf,
            d_lower=0.0,
            geometry_h=self.h,
            category="exterior"
        ))
        return domains

@pytest.fixture
def dummy_geometry(concentric_group):
    return DummyGeometry(concentric_group, h=5.0)

def test_fluid_domains_count(dummy_geometry, simple_stepped_body):
    # Should have one domain per step + one exterior
    expected_count = len(simple_stepped_body.a) + 1
    assert len(dummy_geometry.fluid_domains) == expected_count

def test_fluid_domain_properties(dummy_geometry, simple_stepped_body):
    # Use domain_list to get dictionary access for easy verification
    domains = dummy_geometry.domain_list
    # Access via index to verify dictionary structure
    for i in range(len(simple_stepped_body.a)):
        domain = domains[i]
        assert domain.a_outer == simple_stepped_body.a[i]
        assert domain.d_lower == simple_stepped_body.d[i]
        assert domain.heaving == False
    
    # Check exterior domain
    ext = domains[len(simple_stepped_body.a)]
    assert ext.category == "exterior"
    assert ext.a_outer == np.inf

# -------------------------
# Randomized stress test (fixed to comply with new heaving rule)
# -------------------------
def test_randomized_multiple_bodies():
    np.random.seed(42)
    num_bodies = 5
    bodies = []
    last_max_r = 0.0  # Keep track of last outer radius

    # Randomly select which body (0 to num_bodies-1) will be heaving
    # -1 means none are heaving
    heaving_index = np.random.choice(np.arange(num_bodies), size=1, replace=False)[0]
    
    for i in range(num_bodies):
        steps = np.random.randint(1, 5)
        # Generate increasing radii relative to last_max_r
        a = np.sort(np.random.rand(steps) * 10 + last_max_r + 0.1)  # shift to avoid overlap
        d = np.random.rand(steps) * 5
        slant = np.random.rand(steps) * 0.5
        
        # Only the selected index is True
        heaving = (i == heaving_index)
        
        bodies.append(SteppedBody(a, d, slant, heaving))
        last_max_r = a[-1]  # update last_max_r for next body

    group = ConcentricBodyGroup(bodies)
    geom = DummyGeometry(group, h=10.0)
    
    # FIX: Use domain_list to ensure we get a Dict[int, Domain]
    domains = geom.domain_list

    # Basic checks
    assert all(isinstance(d, Domain) for d in domains.values())
    # Check exterior domain
    max_index = max(domains.keys())
    assert domains[max_index].a_outer == np.inf
    # Ensure number of domains = sum of steps + 1 exterior
    expected_domains = sum(len(body.a) for body in bodies) + 1
    assert len(domains) == expected_domains

# ----------------------------------------------------------------
# NEW TESTS: Abstract Methods and Fluid Domains Logic Coverage
# ----------------------------------------------------------------

def test_body_arrangement_abstract_instantiation():
    """Test that BodyArrangement cannot be instantiated without implementing abstract methods."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class BodyArrangement"):
        BodyArrangement() # type: ignore

def test_geometry_abstract_instantiation(mock_arrangement):
    """Test that Geometry cannot be instantiated without implementing make_fluid_domains."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class Geometry"):
        # FIX: Added type: ignore because Pylance statically knows this is abstract
        # FIX: Passed mock_arrangement instead of None to satisfy type hint
        Geometry(mock_arrangement, h=10.0) # type: ignore

class DictReturningGeometry(Geometry):
    """
    A specific test subclass that returns a dictionary directly from make_fluid_domains.
    This covers the `if isinstance(self.fluid_domains, dict):` branch.
    """
    def make_fluid_domains(self) -> Dict[int, Domain]:
        # Return a dictionary directly
        return {
            0: Domain(0, 1, 0.0, 5.0, 10.0, self.h, category="interior"),
            1: Domain(1, 1, 5.0, np.inf, 0.0, self.h, category="exterior")
        }

def test_fluid_domains_logic_dict_branch(mock_arrangement):
    """Verifies that if make_fluid_domains returns a dict, it is returned as-is via domain_list."""
    # FIX: Passed mock_arrangement instead of None
    geom = DictReturningGeometry(mock_arrangement, h=20.0)
    # Check domain_list, which should return the dict directly
    domains = geom.domain_list
    
    assert isinstance(domains, dict)
    assert len(domains) == 2
    assert domains[0].category == "interior"
    assert domains[1].category == "exterior"

class ListReturningGeometry(Geometry):
    """
    A specific test subclass that returns a list from make_fluid_domains.
    This covers the `return {domain.index: domain ...}` branch in domain_list.
    """
    def make_fluid_domains(self) -> List[Domain]:
        return [
            Domain(0, 1, 0.0, 5.0, 10.0, self.h, category="interior"),
            Domain(1, 1, 5.0, np.inf, 0.0, self.h, category="exterior")
        ]

def test_fluid_domains_logic_list_branch(mock_arrangement):
    """Verifies that if make_fluid_domains returns a list, it is converted to a dict by domain_list."""
    # FIX: Passed mock_arrangement instead of None
    geom = ListReturningGeometry(mock_arrangement, h=20.0)
    
    # We must access .domain_list to get the dictionary conversion
    domains = geom.domain_list
    
    assert isinstance(domains, dict)
    assert len(domains) == 2
    assert domains[0].index == 0
    assert domains[1].index == 1

def test_fluid_domains_caching_behavior(mock_arrangement):
    """Verifies that the fluid_domains property caches the result (doesn't re-instantiate)."""
    # FIX: Passed mock_arrangement instead of None
    geom = ListReturningGeometry(mock_arrangement, h=20.0)
    
    domains_first_call = geom.fluid_domains
    domains_second_call = geom.fluid_domains
    
    # Ensure they are the exact same object in memory
    assert domains_first_call is domains_second_call

class EmptyGeometry(Geometry):
    """Subclass returning empty list to test empty case."""
    def make_fluid_domains(self):
        return []

def test_fluid_domains_empty(mock_arrangement):
    """Verifies behavior when no domains are created."""
    # FIX: Passed mock_arrangement instead of None
    geom = EmptyGeometry(mock_arrangement, h=10.0)
    # domain_list should return empty dict if fluid_domains is empty
    domains = geom.domain_list
    assert isinstance(domains, dict)
    assert len(domains) == 0
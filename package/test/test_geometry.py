# test_geometry.py
import pytest
import numpy as np
import os as os
import sys as sys
current_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(current_dir, '..', 'src'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
from openflash.body import SteppedBody, CoordinateBody
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
    domains = dummy_geometry.fluid_domains
    for i, domain in enumerate(domains[:-1]):
        assert domain.a_outer == simple_stepped_body.a[i]
        assert domain.d_lower == simple_stepped_body.d[i]
        # Fixture is heaving=False, so this should be False
        assert domain.heaving == False
        assert isinstance(domain.slant, bool)
    # Check exterior domain
    ext = domains[-1]
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
    domains = geom.fluid_domains

    # Basic checks
    assert all(isinstance(d, Domain) for d in domains)
    # Check exterior domain
    assert domains[-1].a_outer == np.inf
    # Ensure number of domains = sum of steps + 1 exterior
    expected_domains = sum(len(body.a) for body in bodies) + 1
    assert len(domains) == expected_domains
    # Ensure all interior radii are strictly increasing
    for i in range(len(domains) - 1):
        assert domains[i+1].a_inner >= domains[i].a_outer - 1e-12
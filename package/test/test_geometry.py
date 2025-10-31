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
    return SteppedBody(a, d, slant, heaving=True)

@pytest.fixture
def concentric_group(simple_stepped_body):
    return ConcentricBodyGroup([simple_stepped_body])

# -------------------------
# ConcentricBodyGroup tests
# -------------------------
def test_concatenated_properties(concentric_group):
    body = concentric_group.bodies[0]
    np.testing.assert_array_equal(concentric_group.a, body.a)
    np.testing.assert_array_equal(concentric_group.d, body.d)
    np.testing.assert_array_equal(concentric_group.slant_angle, body.slant_angle)
    np.testing.assert_array_equal(concentric_group.heaving, np.array([True, True]))

def test_invalid_body_type():
    with pytest.raises(TypeError):
        ConcentricBodyGroup([CoordinateBody(np.array([0,1]), np.array([0,1]))])

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
        assert domain.heaving == simple_stepped_body.heaving
        assert isinstance(domain.slant, bool)
    # Check exterior domain
    ext = domains[-1]
    assert ext.category == "exterior"
    assert ext.a_outer == np.inf

# -------------------------
# Randomized stress test (fixed)
# -------------------------
def test_randomized_multiple_bodies():
    np.random.seed(42)
    num_bodies = 5
    bodies = []
    last_max_r = 0.0  # Keep track of last outer radius

    for _ in range(num_bodies):
        steps = np.random.randint(1, 5)
        # Generate increasing radii relative to last_max_r
        a = np.sort(np.random.rand(steps) * 10 + last_max_r + 0.1)  # shift to avoid overlap
        d = np.random.rand(steps) * 5
        slant = np.random.rand(steps) * 0.5
        heaving = np.random.choice([True, False])
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

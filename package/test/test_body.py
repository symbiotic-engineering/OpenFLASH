import pytest
import numpy as np
import sys
import os

# make src importable (adjust path if your repository layout differs)
current_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(current_dir, '..', 'src'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
    
from openflash.body import SteppedBody, CoordinateBody
# -----------------------------
# SteppedBody Tests
# -----------------------------
@pytest.mark.parametrize(
    "a, d, slant, heaving",
    [
        (np.array([1.0, 2.0]), np.array([0.5, 1.0]), np.array([0.1, 0.2]), False),
        (np.array([0.5]), np.array([0.2]), np.array([0.05]), True),  # Single element
        (np.array([10.0, 20.0, 30.0]), np.array([5.0, 10.0, 15.0]), np.array([0.3, 0.4, 0.5]), True),
    ],
)
def test_stepped_body_valid(a, d, slant, heaving):
    body = SteppedBody(a=a, d=d, slant_angle=slant, heaving=heaving)
    assert np.array_equal(body.a, a)
    assert np.array_equal(body.d, d)
    assert np.array_equal(body.slant_angle, slant)
    assert body.heaving == heaving


@pytest.mark.parametrize(
    "a, d, slant",
    [
        (np.array([1.0, 2.0]), np.array([0.5]), np.array([0.1, 0.2])),  # d too short
        (np.array([1.0]), np.array([0.5, 1.0]), np.array([0.1])),      # d too long
        (np.array([1.0, 2.0]), np.array([0.5, 1.0]), np.array([0.1])), # slant too short
    ],
)
def test_stepped_body_invalid_lengths(a, d, slant):
    with pytest.raises(AssertionError):
        SteppedBody(a=a, d=d, slant_angle=slant)


def test_stepped_body_randomized_stress():
    rng = np.random.default_rng(seed=42)
    for _ in range(50):
        n = rng.integers(1, 20)
        a = rng.random(n) * 100
        d = rng.random(n) * 50
        slant = rng.random(n) * 10
        heaving = rng.choice([True, False])
        # Should not raise unless lengths mismatch
        body = SteppedBody(a=a, d=d, slant_angle=slant, heaving=heaving)
        assert len(body.a) == len(body.d) == len(body.slant_angle)


# -----------------------------
# CoordinateBody Tests
# -----------------------------
@pytest.mark.parametrize(
    "r, z, heaving",
    [
        (np.array([1.0, 2.0]), np.array([0.5, 0.7]), False),
        (np.array([5.0]), np.array([2.0]), True),  # Single point
    ],
)
def test_coordinate_body_valid(r, z, heaving):
    body = CoordinateBody(r_coords=r, z_coords=z, heaving=heaving)
    assert np.array_equal(body.r_coords, r)
    assert np.array_equal(body.z_coords, z)
    assert body.heaving == heaving


@pytest.mark.parametrize(
    "r, z",
    [
        (np.array([1.0, 2.0]), np.array([0.5])),     # mismatch
        (np.array([1.0]), np.array([0.5, 0.6])),     # mismatch
    ],
)
def test_coordinate_body_invalid_lengths(r, z):
    with pytest.raises(AssertionError):
        CoordinateBody(r_coords=r, z_coords=z)


def test_coordinate_body_discretize_shape():
    r = np.array([1.0, 2.0, 3.0])
    z = np.array([0.5, 1.0, 1.5])
    body = CoordinateBody(r_coords=r, z_coords=z)
    a, d, slant = body.discretize()
    assert len(a) == len(d) == len(slant) == len(r)


def test_coordinate_body_discretize_gradient_monotonic():
    r = np.array([1.0, 2.0, 4.0, 7.0])
    z = np.array([0.5, 1.5, 2.0, 4.0])
    body = CoordinateBody(r_coords=r, z_coords=z)
    a, d, slant = body.discretize()
    assert np.all(np.diff(a) >= 0)  # radius should remain non-decreasing


def test_coordinate_body_randomized_discretize():
    rng = np.random.default_rng(seed=10)
    for _ in range(30):
        n = rng.integers(2, 15)
        r = np.sort(rng.random(n) * 100)  # enforce sorted for monotonicity
        z = rng.random(n) * 30
        body = CoordinateBody(r_coords=r, z_coords=z)
        a, d, slant = body.discretize()
        assert len(a) == len(d) == len(slant)
        assert np.all(np.isfinite(slant))

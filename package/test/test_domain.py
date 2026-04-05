# test_domain.py

import pytest
import numpy as np
import pytest
import numpy as np
import os as os
import sys as sys
current_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(current_dir, '..', 'src'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
from openflash.domain import Domain

# --- Basic Tests ---

def test_domain_initialization_valid():
    d = Domain(index=0, NMK=5, a_inner=0.0, a_outer=1.0, d_lower=2.0, geometry_h=5.0)
    assert d.index == 0
    assert d.number_harmonics == 5
    assert d.a_inner == 0.0
    assert d.a_outer == 1.0
    assert d.d_lower == 2.0
    assert d.d_upper == 5.0
    assert d.heaving is None
    assert d.slant is False
    assert d.category == "interior"

def test_domain_initialization_invalid_NMK():
    with pytest.raises(AssertionError):
        Domain(index=0, NMK=0, a_inner=0.0, a_outer=1.0, d_lower=0.0, geometry_h=5.0)

def test_domain_initialization_invalid_radii():
    with pytest.raises(AssertionError):
        Domain(index=0, NMK=1, a_inner=1.0, a_outer=0.0, d_lower=0.0, geometry_h=5.0)
    with pytest.raises(AssertionError):
        Domain(index=0, NMK=1, a_inner=-0.1, a_outer=1.0, d_lower=0.0, geometry_h=5.0)

def test_domain_initialization_invalid_depths():
    with pytest.raises(AssertionError):
        Domain(index=0, NMK=1, a_inner=0.0, a_outer=1.0, d_lower=6.0, geometry_h=5.0)
    with pytest.raises(AssertionError):
        Domain(index=0, NMK=1, a_inner=0.0, a_outer=1.0, d_lower=-1.0, geometry_h=5.0)

# --- Adjacency Tests ---

def test_are_adjacent_true_outer_to_inner():
    d1 = Domain(index=0, NMK=1, a_inner=0.0, a_outer=1.0, d_lower=0.0, geometry_h=5.0)
    d2 = Domain(index=1, NMK=1, a_inner=1.0, a_outer=2.0, d_lower=0.0, geometry_h=5.0)
    assert Domain.are_adjacent(d1, d2)
    assert Domain.are_adjacent(d2, d1)

def test_are_adjacent_false():
    d1 = Domain(index=0, NMK=1, a_inner=0.0, a_outer=1.0, d_lower=0.0, geometry_h=5.0)
    d2 = Domain(index=1, NMK=1, a_inner=1.1, a_outer=2.0, d_lower=0.0, geometry_h=5.0)
    assert not Domain.are_adjacent(d1, d2)
    assert not Domain.are_adjacent(d2, d1)

def test_are_adjacent_with_infinite_outer():
    d1 = Domain(index=0, NMK=1, a_inner=0.0, a_outer=np.inf, d_lower=0.0, geometry_h=5.0)
    d2 = Domain(index=1, NMK=1, a_inner=10.0, a_outer=20.0, d_lower=0.0, geometry_h=5.0)
    assert not Domain.are_adjacent(d1, d2)

# --- Randomized Stress Tests ---

@pytest.mark.parametrize("num_domains", [10, 50, 100])
def test_randomized_domains(num_domains):
    np.random.seed(42)  # deterministic for tests
    # Generate sorted inner radii and random widths
    a_inner = np.sort(np.random.rand(num_domains))
    widths = np.random.rand(num_domains) * 0.5 + 0.1
    a_outer = a_inner + widths
    d_lower = np.random.rand(num_domains) * 5
    h = 10.0

    domains = [
        Domain(index=i, NMK=np.random.randint(1, 10), a_inner=a_inner[i], a_outer=a_outer[i], d_lower=d_lower[i], geometry_h=h)
        for i in range(num_domains)
    ]

    # Check all domains created correctly
    for i, d in enumerate(domains):
        assert d.a_outer > d.a_inner
        assert d.d_upper >= d.d_lower
        assert d.number_harmonics > 0

    # Check adjacency for consecutive domains
    for i in range(num_domains - 1):
        # Make domains exactly adjacent for testing
        domains[i+1].a_inner = domains[i].a_outer
        assert Domain.are_adjacent(domains[i], domains[i+1])


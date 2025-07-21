import numpy as np
import pytest
from unittest.mock import MagicMock
from openflash.meem_engine import MEEMEngine

def make_mock_problem():
    def make_domain(h_val, di_val, a_val, harmonics, heaving_val):
        domain = MagicMock()
        domain.h = h_val
        domain.di = di_val
        domain.a = a_val
        domain.number_harmonics = harmonics
        domain.heaving = heaving_val
        return domain

    # Slightly varying values to avoid symmetry / degeneracy
    domain0 = make_domain(h_val=2.0, di_val=1.0, a_val=0.5, harmonics=2, heaving_val=0.3)
    domain1 = make_domain(h_val=2.0, di_val=0.8, a_val=0.4, harmonics=2, heaving_val=0.2)
    domain2 = make_domain(h_val=2.0, di_val=0.6, a_val=0.3, harmonics=2, heaving_val=0.1)

    mock_domain_list = {
        0: domain0,
        1: domain1,
        2: domain2
    }

    mock_geometry = MagicMock()
    mock_geometry.num_regions = 3

    mock_problem = MagicMock()
    mock_problem.domain_list = mock_domain_list
    mock_problem.geometry = mock_geometry
    mock_problem.frequencies = [1.0]
    mock_problem.modes = [0]

    return mock_problem



def test_engine_initialization():
    problem = make_mock_problem()
    engine = MEEMEngine([problem])
    assert isinstance(engine.cache_list, dict)
    assert problem in engine.cache_list

def test_solve_linear_system_multi():
    problem = make_mock_problem()
    engine = MEEMEngine([problem])

    try:
        result = engine.solve_linear_system_multi(problem, m0=1.0)
        assert isinstance(result, np.ndarray)
    except Exception as e:
        pytest.fail(f"Exception occurred in solve_linear_system_multi: {e}")

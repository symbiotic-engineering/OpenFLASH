import os
import sys

import numpy as np
import pytest

current_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(current_dir, "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from openflash.body import SteppedBody
from openflash.geometry import ConcentricBodyGroup
from openflash.basic_region_geometry import BasicRegionGeometry
from openflash.meem_problem import MEEMProblem
from openflash.meem_engine import MEEMEngine
from openflash import multi_equations as me


def _assert_close(left, right):
    np.testing.assert_allclose(np.asarray(left), np.asarray(right), rtol=1e-12, atol=1e-12, equal_nan=True)


@pytest.mark.parametrize(
    "a,d,slant",
    [
        (1.5, 0.5, 0.0),
        (np.array(1.5), np.array(0.5), np.array(0.0)),
        (1.5, np.array(0.5), 0.0),
    ],
)
def test_stepped_body_scalar_input_variants(a, d, slant):
    body = SteppedBody(a=a, d=d, slant_angle=slant, heaving=True)
    np.testing.assert_array_equal(body.a, np.array([1.5]))
    np.testing.assert_array_equal(body.d, np.array([0.5]))
    np.testing.assert_array_equal(body.slant_angle, np.array([0.0]))
    assert body.heaving


@pytest.mark.parametrize(
    "a,d,slant",
    [
        ([1.0, 2.0], [0.5, 0.75], [0.0, 0.1]),
        (np.array([1.0, 2.0]), [0.5, 0.75], np.array([0.0, 0.1])),
        ([1.0, 2.0], np.array([0.5, 0.75]), np.array([0.0, 0.1])),
    ],
)
def test_stepped_body_vector_input_variants(a, d, slant):
    body = SteppedBody(a=a, d=d, slant_angle=slant, heaving=False)
    np.testing.assert_array_equal(body.a, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(body.d, np.array([0.5, 0.75]))
    np.testing.assert_array_equal(body.slant_angle, np.array([0.0, 0.1]))


def test_concentric_body_group_mixed_step_input_types():
    body_0 = SteppedBody(a=1.0, d=np.array(0.5), slant_angle=0.0, heaving=True)
    body_1 = SteppedBody(a=[2.0, 3.0], d=np.array([1.0, 1.5]), slant_angle=[0.0, 0.2], heaving=False)
    group = ConcentricBodyGroup([body_0, body_1])

    np.testing.assert_array_equal(group.a, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(group.d, np.array([0.5, 1.0, 1.5]))
    np.testing.assert_array_equal(group.slant_angle, np.array([0.0, 0.0, 0.2]))
    np.testing.assert_array_equal(group.heaving, np.array([True, False, False]))


@pytest.mark.parametrize(
    "a,d,slant,body_map,heaving_map",
    [
        ([1.0, 2.0], [0.5, 0.8], [0.0, 0.1], [0, 1], [True, False]),
        (np.array([1.0, 2.0]), np.array([0.5, 0.8]), np.array([0.0, 0.1]), [0, 1], np.array([True, False])),
        ([1.0, 2.0], np.array([0.5, 0.8]), None, None, None),
    ],
)
def test_basic_region_geometry_from_vectors_type_variants(a, d, slant, body_map, heaving_map):
    geom = BasicRegionGeometry.from_vectors(
        a=a,
        d=d,
        h=np.array(10.0),
        NMK=[3, 3, 3],
        slant_angle=slant,
        body_map=body_map,
        heaving_map=heaving_map,
    )
    np.testing.assert_array_equal(geom.body_arrangement.a, np.array([1.0, 2.0]))
    assert len(geom.domain_list) == 3


@pytest.mark.parametrize(
    "frequencies,expected",
    [
        (0.5, np.array([0.5])),
        (np.array(0.5), np.array([0.5])),
        ([0.5, 1.0], np.array([0.5, 1.0])),
        (np.array([0.5, 1.0]), np.array([0.5, 1.0])),
    ],
)
def test_meem_problem_set_frequencies_type_variants(frequencies, expected):
    geom = BasicRegionGeometry.from_vectors(a=[1.0], d=[0.5], h=10.0, NMK=[3, 3], heaving_map=[True])
    problem = MEEMProblem(geometry=geom)
    problem.set_frequencies(frequencies)
    np.testing.assert_array_equal(problem.frequencies, expected)


def test_meem_engine_with_mixed_instantiation_types():
    geom = BasicRegionGeometry.from_vectors(
        a=[1.0, 2.0],
        d=np.array([0.5, 0.8]),
        h=np.array(10.0),
        NMK=[3, 3, 3],
        body_map=[0, 1],
        heaving_map=[True, False],
    )
    problem = MEEMProblem(geometry=geom)
    problem.set_frequencies(np.array(0.7))

    engine = MEEMEngine(problem_list=[problem])
    A = engine.assemble_A_multi(problem, m0=0.4)
    b = engine.assemble_b_multi(problem, m0=0.4)

    assert isinstance(A, np.ndarray)
    assert isinstance(b, np.ndarray)
    assert A.shape[0] == A.shape[1]
    assert b.shape[0] == A.shape[0]


def test_multi_equations_scalar_input_equivalence_float_vs_scalar_array():
    _assert_close(me.omega(0.1, 100.0, 9.81), me.omega(np.array(0.1), np.array(100.0), np.array(9.81)))
    _assert_close(me.wavenumber(1.2, 100.0), me.wavenumber(np.array(1.2), np.array(100.0)))
    _assert_close(me.m_k_entry(1, 0.1, 100.0), me.m_k_entry(1, np.array(0.1), np.array(100.0)))


def test_multi_equations_vector_input_equivalence_list_vs_ndarray_and_mixed():
    h = 100.0
    m0 = 0.1
    a_list, a_arr = [5.0, 10.0, 15.0], np.array([5.0, 10.0, 15.0])
    d_list, d_arr = [0.0, 20.0, 50.0], np.array([0.0, 20.0, 50.0])
    heaving_list, heaving_arr = [1.0, 0.5, 0.0], np.array([1.0, 0.5, 0.0])
    m_k_arr = np.array([0.1, 0.2, 0.3, 0.4])
    N_k_arr = np.array([0.9, 1.0, 1.1, 1.2])

    method_calls = [
        (
            lambda d, hv, a: me.I_nm(1, 1, 0, d, h),
            (d_list, heaving_list, a_list),
            (d_arr, heaving_arr, a_arr),
            (d_arr, heaving_list, a_arr),
        ),
        (
            lambda d, hv, a: me.I_mk(1, 1, 1, d, m0, h, m_k_arr, N_k_arr),
            (d_list, heaving_list, a_list),
            (d_arr, heaving_arr, a_arr),
            (d_list, heaving_arr, a_arr),
        ),
        (
            lambda d, hv, a: me.b_potential_entry(1, 1, d, hv, h, a),
            (d_list, heaving_list, a_list),
            (d_arr, heaving_arr, a_arr),
            (d_arr, heaving_list, a_arr),
        ),
        (
            lambda d, hv, a: me.b_velocity_entry(1, 0, hv, a, h, d),
            (d_list, heaving_list, a_list),
            (d_arr, heaving_arr, a_arr),
            (d_list, heaving_arr, a_arr),
        ),
        (
            lambda d, hv, a: me.R_1n_vectorized(np.array([0, 1]), np.array([7.0, 7.0]), 1, h, d, a),
            (d_list, heaving_list, a_list),
            (d_arr, heaving_arr, a_arr),
            (d_arr, heaving_list, a_arr),
        ),
        (
            lambda d, hv, a: me.int_R_1n(1, 1, a, h, d),
            (d_list, heaving_list, a_list),
            (d_arr, heaving_arr, a_arr),
            (d_arr, heaving_arr, a_list),
        ),
        (
            lambda d, hv, a: me.make_R_Z(a, h, d, sharp=True, spatial_res=8)[0],
            (d_list, heaving_list, a_list),
            (d_arr, heaving_arr, a_arr),
            (d_list, heaving_arr, a_arr),
        ),
        (
            lambda d, hv, a: me.make_R_Z(a, h, d, sharp=True, spatial_res=8)[1],
            (d_list, heaving_list, a_list),
            (d_arr, heaving_arr, a_arr),
            (d_list, heaving_arr, a_arr),
        ),
    ]

    for method, args_list, args_array, args_mixed in method_calls:
        expected = method(*args_array)
        _assert_close(method(*args_list), expected)
        _assert_close(method(*args_mixed), expected)

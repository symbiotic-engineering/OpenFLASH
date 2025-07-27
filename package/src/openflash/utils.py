from typing import List, Dict, Union

def build_domain_params(
    NMK: List[int],
    a: List[float],
    d: List[float],
    heaving: List[Union[int, bool]],
    h: float
) -> List[Dict]:
    """
    Create structured domain parameters list for Geometry constructor.
    """
    boundary_count = len(NMK) - 1
    assert len(a) == boundary_count, "Length of 'a' must be one less than length of 'NMK'"
    assert len(d) == boundary_count, "Length of 'd' must match 'a'"
    assert len(heaving) == boundary_count, "Length of 'heaving' must match 'a'"
    for arr, name in zip([a, d, heaving], ['a', 'd', 'heaving']):
        assert len(arr) == boundary_count, f"{name} should have length len(NMK) - 1"

    for entry in heaving:
        assert entry in (0, 1), "heaving entries should be 0 or 1"

    left = 0
    for radius in a:
        assert radius > left, "a values should be increasing and > 0"
        left = radius

    for depth in d:
        assert 0 <= depth < h, "d must be nonnegative and less than h"

    for val in NMK:
        assert isinstance(val, int) and val > 0, "NMK entries must be positive integers"


    domain_params = []

    for idx in range(len(NMK)):
        if idx == 0:
            category = 'inner'
        elif idx == boundary_count:
            category = 'exterior'
        else:
            category = 'outer'

        param = {
            'number_harmonics': NMK[idx],
            'height': h,
            'radial_width': a[idx] if idx < boundary_count else a[-1] * 1.5,
            'top_BC': None,
            'bottom_BC': None,
            'category': category,
            'slant': 0,
        }

        if idx < boundary_count:
            param['a'] = a[idx]
            param['di'] = d[idx]
            param['heaving'] = heaving[idx]

        domain_params.append(param)

    return domain_params

def build_r_coordinates_dict(a: list[float]) -> dict[str, float]:
    """
    Given a list of radial boundary values, return a dict suitable for Geometry.r_coordinates.

    Parameters
    ----------
    a : list of float
        The radial boundary values [a1, a2, a3, ...]
    """
    return {f'a{i+1}': val for i, val in enumerate(a)}

def build_z_coordinates_dict(h: float) -> dict[str, float]:
    """
    Given a height value, return a dict suitable for Geometry.z_coordinates.

    Parameters
    ----------
    h : float
        The height value

    Returns
    -------
    dict
        Dictionary mapping 'h' to the height value.
    """
    return {'h': h}

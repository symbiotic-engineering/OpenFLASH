# package/test/test_multi_heave.py
"""
Compare OpenFLASH multi-body heave hydrodynamic coefficients against Capytaine.

Geometry: three concentric bodies (the inner one a single cylinder, the middle
one a two-step cylinder, the outer one a ring), all heaving. This mirrors
dev/python/multi_heave_dof_development.ipynb.

For every frequency we require < 10% error in:
  * every element of the added-mass matrix   A_ij(omega)
  * every element of the damping matrix      B_ij(omega)
  * the magnitude of the excitation force    |F_i(omega)|
  * the phase of the excitation force        angle(F_i(omega))
    (measured as the relative error the phase alone causes in F; see test)

Run with:   pytest package/test/test_multi_heave.py -v
"""
import itertools
import warnings

import numpy as np
import pytest

# importorskip: if a package is missing, the whole file is *skipped*
# (reported as "s") instead of crashing the test run with an ImportError.
of = pytest.importorskip("openflash")
cpt = pytest.importorskip("capytaine")
pygmsh = pytest.importorskip("pygmsh")
gmsh = pytest.importorskip("gmsh")
import xarray as xr  # noqa: E402
from capytaine.io.meshio import load_from_meshio  # noqa: E402

# --------------------------------------------------------------------------
# Problem definition (same inputs as the notebook)
# --------------------------------------------------------------------------
OMEGAS = np.linspace(0.1, 2, 25)
H = 50
A_LIST = [[1.5], [3.5, 5.5], [7.5]]
D_LIST = [[5], [3, 2], [1]]
HEAVING = [True, True, True]
TRUNCATION_ORDER = 500
MESH_SIZE_FACTOR = 0.35  # converged: errors shrink 0.5 -> 0.35 (contact walls removed)

N_DOFS = sum(HEAVING)
REL_TOL = 0.10  # 10 %


# --------------------------------------------------------------------------
# Solvers. Each returns a dict of plain numpy arrays:
#   added_mass, radiation_damping : (n_omega, n_dofs, n_dofs)
#   excitation_force (complex)    : (n_omega, n_dofs)
# --------------------------------------------------------------------------
def solve_openflash():
    bodies = [
        of.SteppedBody(
            a=np.array(a), d=np.array(d), slant_angle=np.zeros(len(a)), heaving=heave
        )
        for a, d, heave in zip(A_LIST, D_LIST, HEAVING)
    ]
    num_regions = sum(len(a) for a in A_LIST)
    geometry = of.BasicRegionGeometry(
        body_arrangement=of.ConcentricBodyGroup(bodies),
        h=H,
        NMK=[TRUNCATION_ORDER] * (num_regions + 1),
    )
    problem = of.MEEMProblem(geometry)
    problem.set_frequencies(OMEGAS)
    engine = of.MEEMEngine(problem_list=[problem])
    ds = engine.run_and_store_results(0).get_results()

    n_w = len(OMEGAS)
    mag = ds.excitation_force.values.reshape(n_w, N_DOFS)
    phase = ds.excitation_phase.values.reshape(n_w, N_DOFS)
    return {
        "added_mass": ds.added_mass.values.reshape(n_w, N_DOFS, N_DOFS),
        "radiation_damping": ds.damping.values.reshape(n_w, N_DOFS, N_DOFS),
        "excitation_force": mag * np.exp(1j * phase),
    }


def _build_capytaine_body(i):
    """Mesh body i with gmsh, cutting out the hole occupied by body i-1."""
    a_vals, d_vals = A_LIST[i], D_LIST[i]
    v_shift = 1e-4
    with pygmsh.occ.Geometry() as geom:
        gmsh.option.setNumber("Mesh.MeshSizeFactor", MESH_SIZE_FACTOR)
        cyls = None
        for a, d in zip(a_vals, d_vals):
            cyl = geom.add_cylinder([0, 0, v_shift], [0, 0, -d], a)
            cyls = cyl if cyls is None else geom.boolean_union(
                [cyls, cyl], delete_first=True, delete_other=True
            )
        if i > 0:
            # Slightly oversized cut avoids coincident-surface failures.
            hollow = geom.add_cylinder(
                [0, 0, v_shift + 1e-3], [0, 0, -(max(d_vals) + 2e-3)], A_LIST[i - 1][-1]
            )
            cyls = geom.boolean_difference(cyls, hollow, delete_first=True, delete_other=True)
        mesh = geom.generate_mesh()

    mesh_obj = load_from_meshio(mesh, f"mesh{i}")
    fb = cpt.FloatingBody(
        mesh=mesh_obj, lid_mesh=mesh_obj.generate_lid(z=-1e-2), name=f"fb{i}"
    ).keep_immersed_part()

    # Remove the vertical walls where this body touches its neighbours.
    hull = fb.mesh
    if i > 0:  # inner wall, shared with body i-1
        hull = _remove_contact_wall(hull, A_LIST[i - 1][-1], min(D_LIST[i - 1][-1], D_LIST[i][0]))
    if i < len(A_LIST) - 1:  # outer wall, shared with body i+1
        hull = _remove_contact_wall(hull, A_LIST[i][-1], min(D_LIST[i][-1], D_LIST[i + 1][0]))
    fb = cpt.FloatingBody(mesh=hull, lid_mesh=fb.lid_mesh, name=f"fb{i}")

    if HEAVING[i]:
        fb.add_translation_dof(name="Heave")
    return fb


def _remove_contact_wall(mesh, R, depth):
    """
    Delete the part of the vertical wall at radius R that lies above z = -depth.

    Adjacent bodies touch with zero gap, so each one's mesh has a wall sitting
    exactly on top of its neighbour's wall, with opposite normals and no water
    in between. Those panels are not wetted, and leaving them in makes the BEM
    ill-posed: results stop converging with mesh refinement and A_ij != A_ji.
    OpenFLASH (MEEM) has no fluid there either, so removing them is the
    like-for-like comparison.
    """
    c, n = mesh.faces_centers, mesh.faces_normals
    r = np.hypot(c[:, 0], c[:, 1])
    # Triangles on a faceted cylinder are slightly tilted, so allow |n_z| < 0.3.
    on_wall = (np.abs(n[:, 2]) < 0.3) & (np.abs(r - R) < 0.1)
    keep = mesh.extract_faces(np.flatnonzero(~on_wall))
    wall = mesh.extract_faces(np.flatnonzero(on_wall))
    # clipped() keeps the side opposite the plane normal, i.e. z < -depth.
    # This keeps the wetted part of a deeper body's wall below its neighbour.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        below = wall.clipped(cpt.Plane(normal=(0, 0, 1), point=(0, 0, -depth)))
    return keep.join_meshes(below) if below.nb_faces else keep


def solve_capytaine():
    body = cpt.FloatingBody.join_bodies(*[_build_capytaine_body(i) for i in range(len(A_LIST))])
    dofs = list(body.dofs.keys())
    test_matrix = xr.Dataset(coords={
        "omega": OMEGAS, "wave_direction": [0.0], "radiating_dof": dofs,
        "water_depth": H, "rho": of.rho, "g": of.g,
    })
    ds = cpt.BEMSolver().fill_dataset(test_matrix, body)

    # Pull out arrays with an explicit, known axis/DOF order.
    order = dict(radiating_dof=dofs, influenced_dof=dofs)
    rad_dims = ("omega", "radiating_dof", "influenced_dof")
    return {
        "added_mass": ds.added_mass.sel(**order).transpose(*rad_dims).values,
        "radiation_damping": ds.radiation_damping.sel(**order).transpose(*rad_dims).values,
        "excitation_force": ds.excitation_force.sel(wave_direction=0.0, influenced_dof=dofs)
                              .transpose("omega", "influenced_dof").values,
    }


# --------------------------------------------------------------------------
# Fixture: the expensive solves run ONCE and are shared by every test below.
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def results():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {"of": solve_openflash(), "cpt": solve_capytaine()}


def relative_error(test, ref):
    return np.abs(test - ref) / np.abs(ref)


def assert_within_tol(err, label, test, ref):
    """Fail with a per-frequency table showing every frequency that exceeds the tolerance."""
    __tracebackhide__ = True  # pytest: report the failure at the calling test, not here
    bad = np.flatnonzero(err >= REL_TOL)
    if bad.size:
        rows = "\n".join(
            f"  omega={OMEGAS[k]:.3f}  OpenFLASH={test[k]: .5e}  Capytaine={ref[k]: .5e}  err={err[k]:.2%}"
            for k in bad
        )
        pytest.fail(f"{label}: {bad.size}/{len(OMEGAS)} frequencies exceed {REL_TOL:.0%}\n{rows}")


# --------------------------------------------------------------------------
# Tests. parametrize turns each function into one test per matrix element /
# DOF, so a failure tells you exactly which coefficient is off.
# --------------------------------------------------------------------------
IJ = list(itertools.product(range(N_DOFS), range(N_DOFS)))
IJ_IDS = [f"{i+1}{j+1}" for i, j in IJ]
I_IDS = [f"dof{i+1}" for i in range(N_DOFS)]


@pytest.mark.parametrize("i,j", IJ, ids=IJ_IDS)
def test_added_mass(results, i, j):
    test = results["of"]["added_mass"][:, i, j]
    ref = results["cpt"]["added_mass"][:, i, j]
    assert_within_tol(relative_error(test, ref), f"A_{i+1}{j+1}", test, ref)


@pytest.mark.parametrize("i,j", IJ, ids=IJ_IDS)
def test_radiation_damping(results, i, j):
    test = results["of"]["radiation_damping"][:, i, j]
    ref = results["cpt"]["radiation_damping"][:, i, j]
    assert_within_tol(relative_error(test, ref), f"B_{i+1}{j+1}", test, ref)


@pytest.mark.parametrize("i", range(N_DOFS), ids=I_IDS)
def test_excitation_magnitude(results, i):
    test = np.abs(results["of"]["excitation_force"][:, i])
    ref = np.abs(results["cpt"]["excitation_force"][:, i])
    assert_within_tol(relative_error(test, ref), f"|F_{i+1}|", test, ref)


@pytest.mark.parametrize("i", range(N_DOFS), ids=I_IDS)
def test_excitation_phase(results, i):
    F_of = results["of"]["excitation_force"][:, i]
    F_cpt = results["cpt"]["excitation_force"][:, i]
    test, ref = np.angle(F_of), np.angle(F_cpt)
    # Phase error expressed as a relative error of the complex force:
    #   |F e^{i*dphi} - F| / |F| = 2|sin(dphi/2)|   (~= |dphi| in radians)
    # i.e. "how far off would F be if only the phase were wrong". We don't use
    # |dphi| / |phi_ref| because phi_ref -> 0 at low frequency, so a 0.01 deg
    # difference there would count as >20 % "error".
    # angle(F_of / F_cpt) also wraps dphi to (-pi, pi], so +179 vs -179 deg is 2 deg.
    dphi = np.angle(F_of / F_cpt)
    err = 2 * np.abs(np.sin(dphi / 2))
    assert_within_tol(err, f"phase(F_{i+1}) [rad]", test, ref)

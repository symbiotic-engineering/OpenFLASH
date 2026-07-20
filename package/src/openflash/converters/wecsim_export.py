import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np

from openflash.body import SteppedBody
from openflash.multi_constants import rho as openflash_default_rho


@dataclass
class WecSimExportConfig:
    rho: float = float(openflash_default_rho)
    g: float = 9.81
    water_depth: Optional[float] = None
    heading_deg: float = 0.0
    n_irf_omega: int = 1024
    n_irf_time: int = 1025
    irf_t_max: float = 60.0
    conjugate_excitation_for_wecsim: bool = True


def _get_body_count(results_obj, modes: np.ndarray) -> int:
    geometry = getattr(results_obj, "geometry", None)
    if geometry is not None and hasattr(geometry, "body_arrangement"):
        bodies = getattr(geometry.body_arrangement, "bodies", None)
        if bodies is not None:
            return int(len(bodies))
    if modes.size == 0:
        return 0
    return int(np.max(modes)) + 1


def _compute_irf(B: np.ndarray, w_source: np.ndarray, n_w: int, n_t: int, t_max: float):
    if n_w < 2:
        n_w = 2
    if n_t < 2:
        n_t = 2

    w_min = float(np.min(w_source))
    w_max = float(np.max(w_source))
    if np.isclose(w_min, w_max):
        w_max = w_min + 1.0

    ra_w = np.linspace(0.0, w_max, n_w)
    ra_t = np.linspace(0.0, float(t_max), n_t)

    n_dof = B.shape[0]
    ra_K = np.zeros((n_dof, n_dof, n_t), dtype=float)
    cos_wt = np.cos(np.outer(ra_t, ra_w))

    for i in range(n_dof):
        for j in range(n_dof):
            b_ij = B[i, j, :]
            b_interp = ra_w * np.interp(ra_w, w_source, b_ij, left=0.0, right=0.0)
            ra_K[i, j, :] = (2.0 / np.pi) * np.trapz(cos_wt * b_interp[np.newaxis, :], ra_w, axis=1)

    return ra_t, ra_w, ra_K


def _compute_body_hydrostatics(results_obj, nb: int):
    """
    Compute center of buoyancy [x, y, z] for each body from stepped geometry.

    Assumptions:
    - Axisymmetric geometry centered at x=y=0.
    - Each stepped segment spans annulus [r_in, r_out] with draft d below z=0.
    - Segment z-centroid is -d/2.
    """
    cob = np.full((nb, 3), np.nan, dtype=float)
    displaced_volume = np.zeros((nb,), dtype=float)
    geometry = getattr(results_obj, "geometry", None)
    if geometry is None or not hasattr(geometry, "body_arrangement"):
        return cob, displaced_volume

    bodies = getattr(geometry.body_arrangement, "bodies", None)
    if bodies is None:
        return cob, displaced_volume

    prev_outer_radius = 0.0
    for body_idx, body in enumerate(bodies[:nb]):
        if not isinstance(body, SteppedBody):
            continue

        body_volume = 0.0
        z_moment = 0.0
        for outer_radius, draft in zip(np.asarray(body.a, dtype=float), np.asarray(body.d, dtype=float)):
            r_in = float(prev_outer_radius)
            r_out = float(outer_radius)
            d = float(draft)
            if r_out <= r_in or d <= 0.0:
                prev_outer_radius = r_out
                continue

            seg_volume = np.pi * (r_out ** 2 - r_in ** 2) * d
            seg_z = -0.5 * d
            body_volume += seg_volume
            z_moment += seg_volume * seg_z
            prev_outer_radius = r_out

        if body_volume > 0.0:
            cob[body_idx, :] = np.array([0.0, 0.0, z_moment / body_volume], dtype=float)
            displaced_volume[body_idx] = body_volume

    return cob, displaced_volume


def _compute_body_linear_restoring_stiffness(results_obj, nb: int):
    """
    Compute per-body linear restoring stiffness normalized by rho*g.

    For axisymmetric stepped annular bodies, the heave hydrostatic restoring
    coefficient equals waterplane area. This returns per-body 6x6 matrices
    with K33 populated and all other terms set to zero.
    """
    khs_body = np.zeros((nb, 6, 6), dtype=float)
    geometry = getattr(results_obj, "geometry", None)
    if geometry is None or not hasattr(geometry, "body_arrangement"):
        return khs_body

    bodies = getattr(geometry.body_arrangement, "bodies", None)
    if bodies is None:
        return khs_body

    prev_outer_radius = 0.0
    for body_idx, body in enumerate(bodies[:nb]):
        if not isinstance(body, SteppedBody):
            continue

        radii = np.asarray(body.a, dtype=float)
        if radii.size == 0:
            continue

        outer_radius = float(np.max(radii))
        inner_radius = float(prev_outer_radius)
        prev_outer_radius = outer_radius

        if outer_radius <= inner_radius:
            continue

        waterplane_area = np.pi * (outer_radius ** 2 - inner_radius ** 2)
        khs_body[body_idx, 2, 2] = waterplane_area

    return khs_body


def _build_wecsim_canonical(results_obj, config: Optional[WecSimExportConfig] = None) -> Dict[str, np.ndarray]:
    if config is None:
        config = WecSimExportConfig()

    ds = results_obj.get_results()

    required_vars = ["added_mass", "damping"]
    for var in required_vars:
        if var not in ds:
            raise ValueError(f"Required dataset variable '{var}' is missing.")

    if "frequency" not in ds.coords:
        raise ValueError("Dataset is missing required 'frequency' coordinate.")

    w = np.asarray(ds.coords["frequency"].values, dtype=float)
    if w.ndim != 1 or w.size == 0:
        raise ValueError("Frequency coordinate must be a non-empty 1D array.")

    mode_i = np.asarray(ds.coords["mode_i"].values, dtype=int)
    mode_j = np.asarray(ds.coords["mode_j"].values, dtype=int)
    if mode_i.shape != mode_j.shape or not np.array_equal(mode_i, mode_j):
        raise ValueError("mode_i and mode_j coordinates must match for square coefficient matrices.")

    Nb = _get_body_count(results_obj, mode_i)
    n_dof = 6 * Nb
    Nh = 1
    Nf = w.size

    A_src = np.asarray(ds["added_mass"].values, dtype=float)
    B_src = np.asarray(ds["damping"].values, dtype=float)
    if A_src.shape != (Nf, mode_i.size, mode_i.size) or B_src.shape != (Nf, mode_i.size, mode_i.size):
        raise ValueError("added_mass and damping must have shape (Nf, Nm, Nm).")

    if float(config.rho) <= 0.0:
        raise ValueError("WecSimExportConfig.rho must be positive for normalized export.")
    if float(config.g) <= 0.0:
        raise ValueError("WecSimExportConfig.g must be positive for normalized export.")

    rho_g = float(config.rho) * float(config.g)
    added_mass_scale = float(config.rho)
    damping_scale = float(config.rho) * w
    damping_scale_3d = damping_scale[:, np.newaxis, np.newaxis]

    # Export normalized coefficients for BEMIO/WEC-Sim conventions.
    A_src_norm = A_src / added_mass_scale
    B_src_norm = np.divide(B_src, damping_scale_3d, out=np.zeros_like(B_src), where=damping_scale_3d != 0.0)

    A = np.zeros((n_dof, n_dof, Nf), dtype=float)
    B = np.zeros((n_dof, n_dof, Nf), dtype=float)

    for local_i, body_i in enumerate(mode_i):
        gi = int(body_i) * 6 + 2
        for local_j, body_j in enumerate(mode_j):
            gj = int(body_j) * 6 + 2
            A[gi, gj, :] = A_src_norm[:, local_i, local_j]
            B[gi, gj, :] = B_src_norm[:, local_i, local_j]

    ex_mag = np.zeros((n_dof, Nh, Nf), dtype=float)
    ex_phase = np.zeros((n_dof, Nh, Nf), dtype=float)
    if "excitation_force" in ds:
        ex_force_src = np.asarray(ds["excitation_force"].values, dtype=float)
        if ex_force_src.shape == (Nf, mode_i.size):
            ex_force_norm = ex_force_src / rho_g
            for local_i, body_i in enumerate(mode_i):
                gi = int(body_i) * 6 + 2
                ex_mag[gi, 0, :] = ex_force_norm[:, local_i]

    if "excitation_phase" in ds:
        ex_phase_src = np.asarray(ds["excitation_phase"].values, dtype=float)
        if ex_phase_src.shape == (Nf, mode_i.size):
            for local_i, body_i in enumerate(mode_i):
                gi = int(body_i) * 6 + 2
                ex_phase[gi, 0, :] = ex_phase_src[:, local_i]

    if config.conjugate_excitation_for_wecsim:
        ex_phase = -ex_phase

    ex_re = ex_mag * np.cos(ex_phase)
    ex_im = ex_mag * np.sin(ex_phase)

    A0 = A[:, :, 0]
    Ainf = A[:, :, -1]
    T = (2.0 * np.pi) / w
    theta = np.array([float(config.heading_deg)], dtype=float)

    water_depth = config.water_depth
    if water_depth is None:
        geometry = getattr(results_obj, "geometry", None)
        water_depth = getattr(geometry, "h", np.nan)

    ra_t, ra_w, ra_K = _compute_irf(
        B=B,
        w_source=w,
        n_w=config.n_irf_omega,
        n_t=config.n_irf_time,
        t_max=config.irf_t_max,
    )

    cob, displaced_volume = _compute_body_hydrostatics(results_obj, Nb)
    khs_body = _compute_body_linear_restoring_stiffness(results_obj, Nb)

    Khs = np.zeros((n_dof, n_dof), dtype=float)
    for b in range(Nb):
        start = 6 * b
        stop = start + 6
        Khs[start:stop, start:stop] = khs_body[b]

    return {
        "Nb": np.int32(Nb),
        "Nf": np.int32(Nf),
        "Nh": np.int32(Nh),
        "w": w,
        "T": T,
        "theta": theta,
        "A": A,
        "B": B,
        "A0": A0,
        "Ainf": Ainf,
        "excitation_re": ex_re,
        "excitation_im": ex_im,
        "excitation_mag": ex_mag,
        "excitation_phase": ex_phase,
        "ra_t": ra_t,
        "ra_w": ra_w,
        "ra_K": ra_K,
        "rho": np.float64(config.rho),
        "g": np.float64(config.g),
        "h": np.float64(water_depth),
        "mode_to_dof": np.array([int(m) * 6 + 2 for m in mode_i], dtype=np.int32),
        "mode_indices": mode_i.astype(np.int32),
        "CoB": cob,
        "disp_vol": displaced_volume,
        "linear_restoring_stiffness": Khs,
        "linear_restoring_stiffness_body": khs_body,
    }


def export_wecsim_hdf5(results_obj, file_path: str, config: Optional[WecSimExportConfig] = None):
    data = _build_wecsim_canonical(results_obj, config=config)
    if config is None:
        config = WecSimExportConfig()
    meta = {
        "a0_policy": "endpoint_min_omega",
        "ainf_policy": "endpoint_max_omega",
        "irf_method": "cosine_transform_of_damping",
        "heading_deg": float(data["theta"][0]),
        "excitation_conjugated_for_wecsim": bool(config.conjugate_excitation_for_wecsim),
    }

    def _write_text_dataset(h5: h5py.File, path: str, value: str):
        parent_path, dataset_name = path.rsplit("/", 1)
        parent = h5.require_group(parent_path)
        parent.create_dataset(dataset_name, data=np.array(value, dtype=h5py.string_dtype("utf-8")))

    def _write_bemio_compatible_groups(h5: h5py.File):
        def _as_col(x: np.ndarray) -> np.ndarray:
            return np.asarray(x).reshape(-1, 1)

        nb = int(data["Nb"])
        nh = int(data["Nh"])

        _write_text_dataset(h5, "/bem_data/code", "OPENFLASH")

        sim = h5.require_group("/simulation_parameters")
        sim.create_dataset("scaled", data=np.int32(0))
        sim.create_dataset("g", data=data["g"])
        sim.create_dataset("rho", data=data["rho"])
        sim.create_dataset("T", data=_as_col(data["T"]))
        sim.create_dataset("w", data=_as_col(data["w"]))
        sim.create_dataset("wave_dir", data=_as_col(data["theta"]))
        sim.create_dataset("water_depth", data=data["h"])

        # These coefficients are not currently split by physical source in OpenFLASH.
        ex_sc_zeros = np.zeros_like(data["excitation_re"])

        for body_idx in range(nb):
            bnum = body_idx + 1
            start = 6 * body_idx
            stop = start + 6
            body_path = f"/body{bnum}"

            _write_text_dataset(h5, f"{body_path}/properties/name", f"body{bnum}")
            props = h5.require_group(f"{body_path}/properties")
            props.create_dataset("body_number", data=np.int32(bnum))
            props.create_dataset("cb", data=_as_col(data["CoB"][body_idx, :]))
            props.create_dataset("cg", data=np.zeros((3, 1), dtype=float))
            props.create_dataset("disp_vol", data=data["disp_vol"][body_idx])
            props.create_dataset("dof", data=np.int32(6))
            props.create_dataset("dof_start", data=np.int32(start + 1))
            props.create_dataset("dof_end", data=np.int32(stop))

            hydro = h5.require_group(f"{body_path}/hydro_coeffs")

            # WEC-Sim's readBEMIOH5 applies reverseDimensionOrder after h5read.
            # Write with [dof, heading, freq] and [dof, radiating_dof, freq]
            # so loaded hydroData keeps frequency on the interpolation axis.
            hydro.create_dataset("linear_restoring_stiffness", data=data["linear_restoring_stiffness_body"][body_idx])

            ex_group = h5.require_group(f"{body_path}/hydro_coeffs/excitation")
            ex_group.create_dataset("mag", data=data["excitation_mag"][start:stop, :, :])
            ex_group.create_dataset("phase", data=data["excitation_phase"][start:stop, :, :])
            ex_group.create_dataset("re", data=data["excitation_re"][start:stop, :, :])
            ex_group.create_dataset("im", data=data["excitation_im"][start:stop, :, :])

            '''
            sc_group = h5.require_group(f"{body_path}/hydro_coeffs/excitation/scattering")
            sc_group.create_dataset("mag", data=ex_sc_zeros[start:stop, :, :])
            sc_group.create_dataset("phase", data=ex_sc_zeros[start:stop, :, :])
            sc_group.create_dataset("re", data=ex_sc_zeros[start:stop, :, :])
            sc_group.create_dataset("im", data=ex_sc_zeros[start:stop, :, :])

            fk_group = h5.require_group(f"{body_path}/hydro_coeffs/excitation/froude-krylov")
            fk_group.create_dataset("mag", data=ex_sc_zeros[start:stop, :, :])
            fk_group.create_dataset("phase", data=ex_sc_zeros[start:stop, :, :])
            fk_group.create_dataset("re", data=ex_sc_zeros[start:stop, :, :])
            fk_group.create_dataset("im", data=ex_sc_zeros[start:stop, :, :])

            ex_irf = h5.require_group(f"{body_path}/hydro_coeffs/excitation/impulse_response_fun")
            ex_irf.create_dataset("f", data=np.zeros((6, nh, data["ra_t"].size), dtype=float))
            ex_irf.create_dataset("t", data=_as_col(data["ra_t"]))
            ex_irf.create_dataset("w", data=_as_col(data["ra_w"]))
            '''

            am_group = h5.require_group(f"{body_path}/hydro_coeffs/added_mass")
            am_group.create_dataset("all", data=data["A"][start:stop, :, :])
            am_group.create_dataset("inf_freq", data=data["Ainf"][start:stop, :])

            rd_group = h5.require_group(f"{body_path}/hydro_coeffs/radiation_damping")
            rd_group.create_dataset("all", data=data["B"][start:stop, :, :])

            rd_irf = h5.require_group(f"{body_path}/hydro_coeffs/radiation_damping/impulse_response_fun")
            rd_irf.create_dataset("K", data=data["ra_K"][start:stop, :, :])
            rd_irf.create_dataset("t", data=_as_col(data["ra_t"]))
            rd_irf.create_dataset("w", data=_as_col(data["ra_w"]))

    with h5py.File(file_path, "w") as h5:
        _write_bemio_compatible_groups(h5)
        h5.attrs["metadata_json"] = json.dumps(meta)


def _cylindrical_wall_triangles(
    radius: float,
    z_bottom: float,
    z_top: float,
    n_theta: int,
    inward_normal: bool,
) -> List[np.ndarray]:
    triangles: List[np.ndarray] = []
    if radius <= 0.0:
        return triangles

    z_height = abs(float(z_top) - float(z_bottom))
    target_edge = 2.0 * float(radius) * np.sin(np.pi / float(n_theta))
    if target_edge <= 0.0:
        n_z = 1
    else:
        n_z = max(1, int(np.round(z_height / target_edge)))

    angles = np.linspace(0.0, 2.0 * np.pi, n_theta + 1)
    z_levels = np.linspace(z_bottom, z_top, n_z + 1)

    for k in range(n_theta):
        t0 = angles[k]
        t1 = angles[k + 1]

        for iz in range(n_z):
            zb = z_levels[iz]
            zt = z_levels[iz + 1]

            p0b = np.array([radius * np.cos(t0), radius * np.sin(t0), zb], dtype=float)
            p1b = np.array([radius * np.cos(t1), radius * np.sin(t1), zb], dtype=float)
            p0t = np.array([radius * np.cos(t0), radius * np.sin(t0), zt], dtype=float)
            p1t = np.array([radius * np.cos(t1), radius * np.sin(t1), zt], dtype=float)

            tri1 = np.array([p0b, p1b, p1t], dtype=float)
            tri2 = np.array([p0b, p1t, p0t], dtype=float)

            if inward_normal:
                tri1 = np.array([p0b, p1t, p1b], dtype=float)
                tri2 = np.array([p0b, p0t, p1t], dtype=float)

            triangles.append(tri1)
            triangles.append(tri2)

    return triangles


def _horizontal_cap_triangles(
    r_inner: float,
    r_outer: float,
    z: float,
    n_theta: int,
    upward_normal: bool,
) -> List[np.ndarray]:
    triangles: List[np.ndarray] = []
    if r_outer <= 0.0 or r_outer <= r_inner:
        return triangles

    # Use outer-radius chord length as the target in-plane edge size.
    target_edge = 2.0 * float(r_outer) * np.sin(np.pi / float(n_theta))
    radial_width = float(r_outer) - float(r_inner)
    if target_edge <= 0.0:
        n_r = 1
    else:
        n_r = max(1, int(np.round(radial_width / target_edge)))

    angles = np.linspace(0.0, 2.0 * np.pi, n_theta + 1)
    radii = np.linspace(r_inner, r_outer, n_r + 1)

    for ir in range(n_r):
        r0 = float(radii[ir])
        r1 = float(radii[ir + 1])

        # Innermost disk ring (r0=0) is meshed as a fan.
        if np.isclose(r0, 0.0):
            center = np.array([0.0, 0.0, z], dtype=float)
            for k in range(n_theta):
                t0 = angles[k]
                t1 = angles[k + 1]
                p0 = np.array([r1 * np.cos(t0), r1 * np.sin(t0), z], dtype=float)
                p1 = np.array([r1 * np.cos(t1), r1 * np.sin(t1), z], dtype=float)
                tri = np.array([center, p0, p1], dtype=float)
                if not upward_normal:
                    tri = np.array([center, p1, p0], dtype=float)
                triangles.append(tri)
            continue

        # Annular band is meshed by quad split into two triangles per theta segment.
        for k in range(n_theta):
            t0 = angles[k]
            t1 = angles[k + 1]
            o0 = np.array([r1 * np.cos(t0), r1 * np.sin(t0), z], dtype=float)
            o1 = np.array([r1 * np.cos(t1), r1 * np.sin(t1), z], dtype=float)
            i0 = np.array([r0 * np.cos(t0), r0 * np.sin(t0), z], dtype=float)
            i1 = np.array([r0 * np.cos(t1), r0 * np.sin(t1), z], dtype=float)

            tri1 = np.array([i0, o0, o1], dtype=float)
            tri2 = np.array([i0, o1, i1], dtype=float)
            if not upward_normal:
                tri1 = np.array([i0, o1, o0], dtype=float)
                tri2 = np.array([i0, i1, o1], dtype=float)

            triangles.append(tri1)
            triangles.append(tri2)

    return triangles


def _write_ascii_stl(file_path: str, solid_name: str, triangles: Sequence[np.ndarray]):
    with open(file_path, "w", encoding="ascii", newline="\n") as f:
        f.write(f"solid {solid_name}\n")
        for tri in triangles:
            f.write("  facet normal 0 0 0\n")
            f.write("    outer loop\n")
            for vertex in tri:
                f.write(
                    f"      vertex {vertex[0]:.12g} {vertex[1]:.12g} {vertex[2]:.12g}\n"
                )
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write(f"endsolid {solid_name}\n")


def export_to_stl(
    results_obj,
    output_dir: str,
    freeboard: float = 5.0,
    circumferential_segments: int = 36,
):
    """
    Export one ASCII STL per body containing closed meshes for stepped annular geometry.

    Mesh content per body:
    - Top cap at z=freeboard.
    - Bottom cap(s) at each section's draft.
    - Outer wall and inner wall (if annular).
    - Inter-section vertical walls only for exposed depth differences.
    """
    if circumferential_segments < 3:
        raise ValueError("circumferential_segments must be >= 3")

    geometry = getattr(results_obj, "geometry", None)
    if geometry is None or not hasattr(geometry, "body_arrangement"):
        raise ValueError("Results object does not provide a compatible geometry/body arrangement.")

    bodies = getattr(geometry.body_arrangement, "bodies", None)
    if bodies is None:
        raise ValueError("Geometry body arrangement has no bodies.")

    os.makedirs(output_dir, exist_ok=True)

    exported_files: List[str] = []
    prev_outer_radius = 0.0
    for body_idx, body in enumerate(bodies):
        if not isinstance(body, SteppedBody):
            continue

        radii = np.asarray(body.a, dtype=float)
        drafts = np.asarray(body.d, dtype=float)
        if radii.size == 0 or drafts.size == 0:
            continue

        outer_radius = float(np.max(radii))
        inner_radius = float(prev_outer_radius)
        prev_outer_radius = outer_radius

        if outer_radius <= 0.0 or np.max(drafts) <= 0.0:
            continue

        z_top = float(freeboard)
        r_bounds = np.concatenate(([inner_radius], radii))

        triangles: List[np.ndarray] = []

        # Top cap spanning the full radial footprint of the body.
        triangles.extend(
            _horizontal_cap_triangles(
                r_inner=inner_radius,
                r_outer=outer_radius,
                z=z_top,
                n_theta=int(circumferential_segments),
                upward_normal=True,
            )
        )

        # Bottom caps for each annular section.
        for sec_idx, draft in enumerate(drafts):
            z_bottom = -float(draft)
            triangles.extend(
                _horizontal_cap_triangles(
                    r_inner=float(r_bounds[sec_idx]),
                    r_outer=float(r_bounds[sec_idx + 1]),
                    z=z_bottom,
                    n_theta=int(circumferential_segments),
                    upward_normal=False,
                )
            )

        # Outer wall from freeboard down to outermost section draft.
        triangles.extend(
            _cylindrical_wall_triangles(
                radius=outer_radius,
                z_bottom=-float(drafts[-1]),
                z_top=z_top,
                n_theta=int(circumferential_segments),
                inward_normal=False,
            )
        )

        # Inner wall only if this body is annular.
        if inner_radius > 0.0 and inner_radius < outer_radius:
            triangles.extend(
                _cylindrical_wall_triangles(
                    radius=inner_radius,
                    z_bottom=-float(drafts[0]),
                    z_top=z_top,
                    n_theta=int(circumferential_segments),
                    inward_normal=True,
                )
            )

        # Inter-section walls for exposed portions only.
        for sec_idx in range(len(drafts) - 1):
            d_inner = float(drafts[sec_idx])
            d_outer = float(drafts[sec_idx + 1])
            if np.isclose(d_inner, d_outer):
                continue

            z_low = -max(d_inner, d_outer)
            z_high = -min(d_inner, d_outer)
            if z_high <= z_low:
                continue

            # If outer section is deeper, exposed face points inward.
            inward_normal = d_outer > d_inner
            triangles.extend(
                _cylindrical_wall_triangles(
                    radius=float(r_bounds[sec_idx + 1]),
                    z_bottom=z_low,
                    z_top=z_high,
                    n_theta=int(circumferential_segments),
                    inward_normal=inward_normal,
                )
            )

        file_name = f"body_{body_idx + 1}.stl"
        out_path = os.path.join(output_dir, file_name)
        _write_ascii_stl(out_path, f"body_{body_idx + 1}", triangles)
        exported_files.append(out_path)

    return exported_files

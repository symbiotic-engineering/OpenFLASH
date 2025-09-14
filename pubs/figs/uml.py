from graphviz import Digraph

dot = Digraph(comment="MEEM UML Diagram (Simplified Geometry-Domain Link)", format="png")
dot.attr(rankdir="TB", fontsize="10")  # Top-to-Bottom layout

# Define classes with constructors explicitly
classes = {
    "Geometry": [
        "+ __init__(r_coordinates: Dict[str, float], z_coordinates: Dict[str, float], domain_params: List[Dict])",
        "- r_coordinates: Dict[str, float]",
        "- z_coordinates: Dict[str, float]",
        "- domain_params: List[Dict]",
        "+ adjacency_matrix: np.ndarray",
        "+ make_domain_list(): Dict[int, Domain]"
    ],
    "Domain": [
        "+ __init__(number_harmonics: int, height: float, radial_width: float, top_BC, bottom_BC, category: str, params: dict, index: int, geometry: Geometry)",
        "- number_harmonics: int",
        "- height: float",
        "- radial_width: float",
        "- top_BC",
        "- bottom_BC",
        "- category: str",
        "- params: dict",
        "- index: int",
        "- geometry: Geometry",
        "+ build_domain_params(NMK, a, d, heaving, h, slant) List[Dict]",
        "+ build_r_coordinates_dict() dict[str, float]",
        "+ build_z_coordinates_dict() dict[str, float]"
    ],
    "MEEMProblem": [
        "+ __init__(geometry: Geometry)",
        "- geometry: Geometry",
        "+ set_frequencies_modes(frequencies: np.ndarray, modes: np.ndarray)"
    ],
    "MEEMEngine": [
        "+ __init__()",
        "- problem_list: List[MEEMProblem]",
        "+ _ensure_m_k_and_N_k_arrays(problem: MEEMProblem, m0)",
        "+ assemble_A_multi(problem: MEEMProblem, m0): ndarray",
        "+ assemble_b_multi(problem: MEEMProblem, m0): ndarray",
        "+ build_problem_cache(problem: MEEMProblem): ProblemCache",
        "+ calculate_potentials: Dict",
        "+ calculate_velocities(problem, solution_vector: np.ndarray, m0, spatial_res, sharp): Dict",
        "+ compute_hydrodynamic_coefficients(problem, X, m0)",
        "+ reformat_coeffs(x: np.ndarray, NMK, boundary_count) list[ndarray]",
        "+ run_and_store_results(problem_index: int): Results",
        "+ solve_linear_system_multi(problem: MEEMProblem, m0): ndarray",
        "+ visualize_potential(field, R, Z, title): tuple"
    ],
    "Results": [
        "+ __init__(geometry: Geometry, frequencies: np.ndarray, modes: np.ndarray)",
        "- geometry: Geometry",
        "- frequencies: np.ndarray",
        "- modes: np.ndarray",
        "+ display_results(): str",
        "+ export_to_netcdf(file_path: str)",
        "+ get_results(): xarray.Dataset",
        "+ store_all_potentials(all_potentials_batch: list[dict])",
        "+ store_hydrodynamic_coefficients(frequencies: np.ndarray, modes: np.ndarray, added_mass_matrix: np.ndarray, damping_matrix: np.ndarray)",
        "+ store_results(domain_index: int, radial_data: ndarray, vertical_data: ndarray)",
        "+ store_single_potential_field(potential_data: dict, frequency_idx: int = 0, mode_idx: int = 0)"
    ],
    "ProblemCache": [
        "+ __init__(problem: MEEMProblem)",
        "- problem",
        "+ _add_m0_dependent_A_entry(row: int, col: int, calc_func: Callable)",
        "+ _add_m0_dependent_b_entry(row: int, calc_func: Callable)",
        "+ _get_A_template(): ndarray",
        "+ _get_b_template(): ndarray",
        "+ _get_closure(key: str)",
        "+ _set_A_template(A_template: ndarray)",
        "+ _set_I_nm_vals(I_nm_vals: ndarray)",
        "+ _set_b_template(b_template: ndarray)",
        "+ _set_closure(key: str, closure)",
        "+ _set_m_k_and_N_k_funcs(m_k_entry_func: Callable, N_k_func: Callable)",
        "+ _set_precomputed_m_k_N_k(m_k_arr: ndarray, N_k_arr: ndarray)"
    ]
}

# Add class nodes
for cls, members in classes.items():
    members_str = "\\l".join(members) + "\\l"
    label = f"{{{cls}|{members_str}}}"
    dot.node(cls, shape="record", label=label)

# Relationships
dot.edge("Geometry", "Domain", arrowhead="diamond", label="1..*")  # Composition
dot.edge("MEEMProblem", "Geometry", label="1")
dot.edge("MEEMEngine", "MEEMProblem", label="*")
dot.edge("MEEMEngine", "ProblemCache", label="*")
dot.edge("MEEMEngine", "Results", label="*")
dot.edge("Results", "Geometry", label="1")

# Legend / Key node
legend = """<
<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="2">
  <TR><TD COLSPAN="2"><B>Legend</B></TD></TR>
  <TR><TD ALIGN="LEFT">1</TD><TD ALIGN="LEFT">Exactly one</TD></TR>
  <TR><TD ALIGN="LEFT">*</TD><TD ALIGN="LEFT">Many</TD></TR>
  <TR><TD ALIGN="LEFT">1..*</TD><TD ALIGN="LEFT">One to many</TD></TR>
  <TR><TD ALIGN="LEFT">Diamond</TD><TD ALIGN="LEFT">Composition</TD></TR>
  <TR><TD ALIGN="LEFT">Arrow</TD><TD ALIGN="LEFT">Association</TD></TR>
  <TR><TD ALIGN="LEFT">__init__</TD><TD ALIGN="LEFT">Constructor</TD></TR>
</TABLE>>"""
dot.node("Legend", legend, shape="none")

# Render
output_path = "openflash_uml_vertical"
dot.render(output_path, cleanup=True)

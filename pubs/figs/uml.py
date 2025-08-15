from graphviz import Digraph

dot = Digraph(comment="MEEM UML Diagram (Simplified Geometry-Domain Link)", format="png")
dot.attr(rankdir="TB", fontsize="10")  # Top-to-Bottom layout

# Define classes
classes = {
    "Geometry": [
        "- r_coordinates: Dict[str, float]",
        "- z_coordinates: Dict[str, float]",
        "- domain_params: List[Dict]",
        "+ adjacency_matrix: np.ndarray",
        "+ make_domain_list(): Dict[int, Domain]"
    ],
    "Domain": [
        "- number_harmonics: int",
        "- height: float",
        "- radial_width: float",
        "- top_BC",
        "- bottom_BC",
        "- category: str",
        "- params: dict",
        "- index: int",
        "- geometry: Geometry",
        "+ build_domain_params(...) List[Dict]",
        "+ build_r_coordinates_dict() dict[str, float]",
        "+ build_z_coordinates_dict() dict[str, float]"
    ],
    "MEEMProblem": [
        "- geometry: Geometry",
        "+ set_frequencies_modes(frequencies: np.ndarray, modes: np.ndarray)"
    ],
    "MEEMEngine": [
        "- problem_list: List[MEEMProblem]",
        "+ _ensure_m_k_and_N_k_arrays(problem: MEEMProblem, m0)",
        "+ assemble_A_multi(problem: MEEMProblem, m0): ndarray",
        "+ assemble_b_multi(problem: MEEMProblem, m0): ndarray",
        "+ build_problem_cache(problem: MEEMProblem): ProblemCache",
        "+ calculate_potentials(...): Dict",
        "+ calculate_velocities(...): Dict",
        "+ compute_hydrodynamic_coefficients(...)",
        "+ reformat_coeffs(...) list[ndarray]",
        "+ run_and_store_results(problem_index: int): Results",
        "+ solve_linear_system_multi(problem: MEEMProblem, m0): ndarray",
        "+ visualize_potential(field, R, Z, title): tuple"
    ],
    "Results": [
        "- geometry: Geometry",
        "- frequencies: np.ndarray",
        "- modes: np.ndarray",
        "+ display_results(): str",
        "+ export_to_netcdf(file_path: str)",
        "+ get_results(): xarray.Dataset",
        "+ store_all_potentials(all_potentials_batch: list[dict])",
        "+ store_hydrodynamic_coefficients(...)",
        "+ store_results(domain_index: int, radial_data: ndarray, vertical_data: ndarray)",
        "+ store_single_potential_field(...)"
    ],
    "ProblemCache": [
        "- problem",
        "+ add_m0_dependent_A_entry(row: int, col: int, calc_func: Callable)",
        "+ add_m0_dependent_b_entry(row: int, calc_func: Callable)",
        "+ get_A_template(): ndarray",
        "+ get_b_template(): ndarray",
        "+ get_closure(key: str)",
        "+ set_A_template(A_template: ndarray)",
        "+ set_I_nm_vals(I_nm_vals: ndarray)",
        "+ set_b_template(b_template: ndarray)",
        "+ set_closure(key: str, closure)",
        "+ set_m_k_and_N_k_funcs(m_k_entry_func: Callable, N_k_func: Callable)",
        "+ set_precomputed_m_k_N_k(m_k_arr: ndarray, N_k_arr: ndarray)"
    ]
}

# Add class nodes again
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

# Render
output_path = "openflash_uml_vertical"
dot.render(output_path, cleanup=True)

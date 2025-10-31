from graphviz import Digraph
import os

# Prevent Graphviz from trying to open the file after rendering
os.environ["GV_RENDER_NEATO_DOES_NOT_WORKAROUND_BUG_2312"] = "1"

# --- HELPER FUNCTION TO CREATE LEGEND SYMBOL ---
def create_symbol_image(filename, attributes):
    """Creates a small image of a graphviz symbol."""
    d = Digraph()
    d.attr('node', shape='point', style='invis', width='0')
    d.attr(rankdir='LR', splines='false', margin='0')
    d.edge('start', 'end', **attributes)
    d.render(os.path.splitext(filename)[0], format='png', cleanup=True)

# --- CREATE THE SYMBOL IMAGES NEEDED FOR THE LEGEND ---
create_symbol_image("legend_inheritance.png", {'arrowhead': 'empty'})
create_symbol_image("legend_association.png", {'arrowhead': 'vee'})
create_symbol_image("legend_dependency.png", {'arrowhead': 'vee', 'style': 'dashed'})
create_symbol_image("legend_aggregation.png", {'arrowhead': 'odiamond'})
create_symbol_image("legend_composition.png", {'arrowhead': 'diamond', 'style': 'filled'})


# --- MAIN DIAGRAM DEFINITION ---
dot = Digraph(comment="MEEM UML Diagram", format="png")
dot.attr(rankdir="TB", fontsize="10")  # Top-to-Bottom layout

# Define classes with constructors explicitly
classes = {
    "Geometry": [
        "<<Abstract>>",
        "+ __init__(body_arrangement: BodyArrangement, h: float)",
        "- body_arrangement: BodyArrangement",
        "- h: float",
        "- _fluid_domains: List[Domain]",
        "+ fluid_domains: List[Domain]",
        "+ make_fluid_domains(): List[Domain]"
    ],
    "BasicRegionGeometry": [
        "+ __init__(body_arrangement: ConcentricBodyGroup, h: float, NMK: List[int])",
        "- NMK: List[int]",
        "+ make_fluid_domains(): List[Domain]",
        "+ from_vectors(a: np.ndarray, d: np.ndarray, h: float, NMK: List[int]) -> BasicRegionGeometry"
    ],
    "AnyRegionGeometry": [
        "(future implementation)"
    ],
    "BodyArrangement": [
        "<<Abstract>>",
        "+ __init__(bodies: List[Body])",
        "- bodies: List[Body]",
        "+ a: np.ndarray",
        "+ d: np.ndarray",
        "+ slant_angle: np.ndarray",
        "+ heaving: np.ndarray"
    ],
    "ConcentricBodyGroup": [
        "+ __init__(bodies: List[SteppedBody])",
        "+ _get_concatenated_property(prop_name: str): np.ndarray",
        "+ _get_heaving_flags(): np.ndarray",
        "+ a: np.ndarray",
        "+ d: np.ndarray",
        "+ slant_angle: np.ndarray",
        "+ heaving: np.ndarray"
    ],
    "Body": [
        "<<Abstract>>",
        "- heaving: bool"
    ],
    "SteppedBody": [
        "+ __init__(a: np.ndarray, d: np.ndarray, slant_angle: np.ndarray, heaving: bool = False)",
        "- a: np.ndarray",
        "- d: np.ndarray",
        "- slant_angle: np.ndarray",
        "- heaving: bool"
    ],
    "CoordinateBody": [
        "+ __init__(r_coords: np.ndarray, z_coords: np.ndarray, heaving: bool = False)",
        "+ discretize(): Tuple[np.ndarray, np.ndarray, np.ndarray]"
    ],
    "Domain": [
        "+ __init__(index: int, NMK: int, a_inner: float, a_outer: float, d_lower: float, geometry_h: float, heaving: Optional[bool], slant: bool, category: str)",
        "- index: int",
        "- number_harmonics: int",
        "- a_inner: float",
        "- a_outer: float",
        "- d_lower: float",
        "- geometry_h: float",
        "- category: str",
        "+ are_adjacent(d1: Domain, d2: Domain) -> bool"
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

# Add class nodes using robust HTML-like labels
for cls, members in classes.items():
    # Escape special characters for HTML
    escaped_members = [m.replace('<', '&lt;').replace('>', '&gt;') for m in members]

    # Join members with HTML line breaks, aligned left
    members_str = '<BR ALIGN="LEFT"/>'.join(escaped_members)

    # Create the complete HTML table label for the node
    label = f'''<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
    <TR><TD><B>{cls}</B></TD></TR>
    <TR><TD ALIGN="LEFT">{members_str}</TD></TR>
    </TABLE>>'''

    dot.node(cls, shape="plain", label=label) # Use shape="plain" for HTML labels


# Relationships
# Geometry owns Domain (Composition: filled diamond)
dot.edge("Geometry", "Domain", arrowhead="diamond", style="filled", label="1..*")  

# MEEMProblem references Geometry (Association)
dot.edge("MEEMProblem", "Geometry", arrowhead="vee", label="1")

# MEEMEngine aggregates MEEMProblem (Aggregation: hollow diamond)
dot.edge("MEEMEngine", "MEEMProblem", arrowhead="odiamond", label="*")  

# MEEMEngine composes ProblemCache (Composition: filled diamond)
dot.edge("MEEMEngine", "ProblemCache", arrowhead="diamond", style="filled", label="*")  

# MEEMEngine depends on Results (Dependency: dashed arrow)
dot.edge("MEEMEngine", "Results", style="dashed", label="*")  

# Results references Geometry (Association)
dot.edge("Results", "Geometry", arrowhead="vee", label="1")

# Inheritance (is-a)
dot.edge("BasicRegionGeometry", "Geometry", arrowhead="empty")
dot.edge("AnyRegionGeometry", "Geometry", arrowhead="empty")
dot.edge("ConcentricBodyGroup", "BodyArrangement", arrowhead="empty")
dot.edge("SteppedBody", "Body", arrowhead="empty")
dot.edge("CoordinateBody", "Body", arrowhead="empty")

# Composition (has-a, strong ownership)
dot.edge("ConcentricBodyGroup", "Body", arrowhead="diamond", style="filled", label="1..*")

# Association (uses-a)
dot.edge("Geometry", "BodyArrangement", arrowhead="vee", label="1")

# --- Legend / Key node ---
legend = """<
<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="2" CELLPADDING="4">
  <TR><TD COLSPAN="2"><B>Legend</B></TD></TR>
  <TR><TD><IMG SRC="legend_inheritance.png"/></TD><TD ALIGN="LEFT">Inheritance ("is-a")</TD></TR>
  <TR><TD><IMG SRC="legend_composition.png"/></TD><TD ALIGN="LEFT">Composition ("Owns / Is part of")</TD></TR>
  <TR><TD><IMG SRC="legend_aggregation.png"/></TD><TD ALIGN="LEFT">Aggregation ("Has-a / Contains")</TD></TR>
  <TR><TD><IMG SRC="legend_association.png"/></TD><TD ALIGN="LEFT">Association ("uses-a")</TD></TR>
  <TR><TD><IMG SRC="legend_dependency.png"/></TD><TD ALIGN="LEFT">Dependency / Planned</TD></TR>
  <TR><TD COLSPAN="2" HEIGHT="1" BGCOLOR="black" BORDER="0"></TD></TR>
  <TR><TD ALIGN="LEFT">1, *, 1..*</TD><TD ALIGN="LEFT">Multiplicity</TD></TR>
</TABLE>>"""
dot.node("Legend", legend, shape="none")

# Render the final diagram
output_path = "openflash_uml_vertical"
dot.render(output_path, cleanup=True)

# Clean up the generated symbol images
for img in ["legend_inheritance.png", "legend_association.png", "legend_dependency.png",
            "legend_aggregation.png", "legend_composition.png"]:
    if os.path.exists(img):
        os.remove(img)

print(f"UML diagram successfully generated as '{output_path}.png'")
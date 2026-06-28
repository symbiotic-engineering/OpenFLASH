# __init__.py

# --- Core Public Classes ---
from .meem_engine import MEEMEngine
from .meem_problem import MEEMProblem
from .results import Results

# --- Geometry and Body Components ---
from .geometry import Geometry, ConcentricBodyGroup
from .body import Body, SteppedBody
from .basic_region_geometry import BasicRegionGeometry
from .domain import Domain
from .problem_cache import ProblemCache

# --- Key Utility Functions and Constants ---
from .multi_equations import *
from .multi_constants import *

# --- Define the Public API ---
__all__ = [
    # Core Classes
    "MEEMEngine",
    "MEEMProblem",
    "Results",
    "ProblemCache",

    # Geometry Components
    "Geometry",
    "Body",
    "SteppedBody",
    "ConcentricBodyGroup",
    "BasicRegionGeometry",
    "Domain",

    # Utilities
    "omega",
    "g",
]
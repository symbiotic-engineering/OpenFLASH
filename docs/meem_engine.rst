.. currentmodule:: package.meem_engine

MEEMEngine Class
================

The `MEEMEngine` class manages multiple `MEEMProblem` instances and performs actions such as solving systems of equations,
assembling matrices, and computing hydrodynamic coefficients.

Methods
-------

__init__(problem_list: List[MEEMProblem])
------------------------------------------
Initialize the `MEEMEngine` object.

:param problem_list: List of `MEEMProblem` instances.
:returns: None

assemble_A(problem: MEEMProblem) -> np.ndarray
------------------------------------------------
Assemble the system matrix `A` for a given problem.

:param problem: The `MEEMProblem` instance containing the domain information.
:returns: Assembled matrix `A` of shape (size, size), where `size` is the total number of harmonics across all domains.

assemble_A_multi(problem: MEEMProblem) -> np.ndarray
------------------------------------------------------
Assemble the system matrix `A` for a multi-domain problem with multiple boundary conditions.

:param problem: The `MEEMProblem` instance containing the domain list and domain parameters.
:returns: Assembled matrix `A` of shape (size, size), where `size` is the total number of harmonics across all domains.

assemble_b(problem: MEEMProblem) -> np.ndarray
------------------------------------------------
Assemble the right-hand side vector `b` for a given problem.

:param problem: The `MEEMProblem` instance.
:returns: Assembled vector `b` as a numpy array.

assemble_b_multi(problem: MEEMProblem) -> np.ndarray
------------------------------------------------------
Assemble the right-hand side vector `b` for a given multi-region problem.

:param problem: The `MEEMProblem` instance.
:returns: Assembled vector `b` as a numpy array.

compute_hydrodynamic_coefficients(problem: MEEMProblem, X: np.ndarray) -> Dict[str, any]
-----------------------------------------------------------------------------------------
Compute the hydrodynamic coefficients for a given problem and solution vector `X`.

:param problem: The `MEEMProblem` instance.
:param X: The solution vector `X` obtained from solving `A x = b`.
:returns: Dictionary containing the hydrodynamic coefficients and related values.

Detailed Method Descriptions
-----------------------------

__init__(problem_list: List[MEEMProblem])
------------------------------------------
This method initializes the `MEEMEngine` object with a list of `MEEMProblem` instances. It is used to set up the
problem set that the engine will manage.

assemble_A(problem: MEEMProblem) -> np.ndarray
------------------------------------------------
This method assembles the system matrix `A` for the given problem. The matrix is constructed based on the harmonics
and domain properties. It calculates matrix entries using the functions `equations.R_1n_1`, `equations.R_1n_2`,
`equations.Lambda_k_r`, etc., to populate the elements of the matrix.

assemble_A_multi(problem: MEEMProblem) -> np.ndarray
------------------------------------------------------
This method assembles the system matrix `A` for a multi-domain problem. The matrix is constructed similarly to
`assemble_A`, but with added complexity to account for multiple boundaries and regions.

assemble_b(problem: MEEMProblem) -> np.ndarray
------------------------------------------------
This method assembles the right-hand side vector `b` for the given problem. The vector is computed by integrating
various functions, including `phi_p_i1_i2_a1`, `Z_n_i1`, `phi_p_a2`, and others across the domains. It considers the
harmonics and boundary conditions to compute the vector entries.

assemble_b_multi(problem: MEEMProblem) -> np.ndarray
------------------------------------------------------
This method assembles the right-hand side vector `b` for a multi-domain problem. It calculates entries for boundary
conditions, velocity matching, and potential matching using the corresponding multi-domain equations.

compute_hydrodynamic_coefficients(problem: MEEMProblem, X: np.ndarray) -> Dict[str, any]
-----------------------------------------------------------------------------------------
This method computes the hydrodynamic coefficients for the given problem and solution vector `X`. The coefficients are
calculated using integrals of various functions over the domains. The method also calculates the real and imaginary
components of the coefficients and finds the maximum heaving radius for non-dimensionalizing the coefficient.

### Example Usage

```python````
from meem_engine import MEEMEngine
from meem_problem import MEEMProblem

# Define problems (Example)
problem_1 = MEEMProblem(...)
problem_2 = MEEMProblem(...)

# Create engine with a list of problems
engine = MEEMEngine([problem_1, problem_2])

# Assemble matrix for a single problem
matrix_A = engine.assemble_A(problem_1)

# Assemble matrix for a multi-domain problem
matrix_A_multi = engine.assemble_A_multi(problem_1)

# Assemble right-hand side vector for a problem
vector_b = engine.assemble_b(problem_1)

# Assemble right-hand side vector for a multi-domain problem
vector_b_multi = engine.assemble_b_multi(problem_1)

# Compute hydrodynamic coefficients for a solution
hydro_results = engine.compute_hydrodynamic_coefficients(problem_1, X)

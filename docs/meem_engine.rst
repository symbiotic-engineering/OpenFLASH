.. currentmodule:: package.meem_engine

MEEM_engine Module
===================

This module defines the `MEEMEngine` class responsible for managing and solving multiple `MEEMProblem` instances. It handles the numerical methods required to solve boundary value problems, assembles the necessary matrices, and visualizes the results.

.. automodule:: meem_engine
   :members:
   :undoc-members:

Class:
--------

.. autoclass:: meem_engine.MEEMEngine
   :members:
   :noindex:
   :undoc-members:
   :show-inheritance:

Attributes:
-----------
- `problem_list`: List[MEEMProblem] — A collection of `MEEMProblem` instances to be solved.

Methods:
--------
.. method:: __init__(problem_list: List[MEEMProblem])

   Initializes the MEEMEngine with a list of problems.

   :param problem_list: A list of MEEMProblem instances.
   :type problem_list: List[MEEMProblem]

.. method:: assemble_A(problem: MEEMProblem) -> np.ndarray

   Assembles the coefficient matrix A for the given problem.
   
   :param problem: The MEEMProblem instance.
   :type problem: MEEMProblem
   :returns: The assembled coefficient matrix A.

.. method:: assemble_b(problem: MEEMProblem) -> np.ndarray

   Assembles the right-hand side vector b for the given problem.
   
   :param problem: The MEEMProblem instance.
   :type problem: MEEMProblem
   :returns: The assembled right-hand side vector b.

.. method:: linear_solve_Axb(A: np.ndarray, b: np.ndarray) -> np.ndarray

   Solves the linear system Ax = b.
   
   :param A: Coefficient matrix A.
   :type A: np.ndarray
   :param b: Right-hand side vector b.
   :type b: np.ndarray
   :returns: Solution vector x.

.. method:: visualize_A(A: np.ndarray)

   Visualizes the non-zero entries of matrix A.

   :param A: Coefficient matrix A.
   :type A: np.ndarray

.. method:: perform_matching(matching_info: Dict[int, Dict[str, bool]]) -> bool

   Performs matching for all problems based on the provided matching information.
   
   :param matching_info: Matching information between domains.
   :type matching_info: Dict[int, Dict[str, bool]]
   :returns: True if all matchings are successful, False otherwise.

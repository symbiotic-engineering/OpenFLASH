.. currentmodule:: package.meem_engine

MEEM_engine Module
===================

This module defines the `MEEM_engine` class responsible for managing and solving multiple `MEEM_problem` instances. It handles the numerical methods required to solve boundary value problems, assembles the necessary matrices, and visualizes the results.

.. automodule:: meem_engine
   :members:
   :undoc-members:

Class:
--------

.. autoclass:: meem_engine.MEEM_engine
   :members:
   :noindex:
   :undoc-members:
   :show-inheritance:

Attributes:
-----------
- `problem_list`: Dict[str, MEEM_problem] — A collection of `MEEM_problem` instances to be solved.

Methods:
--------
.. method:: __init__(problem_list: Dict[str, MEEM_problem])
   :noindex:

   Initializes the MEEM_engine with a list of problems.
   
   :param problem_list: A dictionary of MEEM_problem instances.
   :type problem_list: Dict[str, MEEM_problem]

.. method:: linear_solve_Ax_b(matrix: Matrix, vector: Vector) -> Vector

   Solves the linear system Ax = b.
   
   :param matrix: Coefficient matrix A.
   :type matrix: Matrix
   :param vector: Known vector b.
   :type vector: Vector
   :returns: Vector x, which is the solution to Ax = b.

.. method:: assemble_A(problem: MEEM_problem) -> Matrix

   Assembles the coefficient matrix A for the given problem.
   
   :param problem: The MEEM_problem instance.
   :type problem: MEEM_problem
   :returns: The assembled coefficient matrix A.

.. method:: assemble_b(problem: MEEM_problem) -> Vector

   Assembles the known vector b for the given problem.
   
   :param problem: The MEEM_problem instance.
   :type problem: MEEM_problem
   :returns: The assembled known vector b.

.. method:: coupling_integral(domain1: Domain, domain2: Domain) -> float

   Computes the coupling integral between two domains.
   
   :param domain1: The first domain.
   :type domain1: Domain
   :param domain2: The second domain.
   :type domain2: Domain
   :returns: The value of the coupling integral.

.. method:: visualize()

   Visualizes the results of the solved problems.

.. method:: perform_matching(boundary_conditions: Dict[str, str]) -> bool

   Performs matching of boundary conditions across domains.
   
   :param boundary_conditions: A dictionary containing boundary condition matching data.
   :type boundary_conditions: Dict[str, str]
   :returns: True if matching is successful, otherwise False.

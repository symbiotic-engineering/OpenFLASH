.. currentmodule:: package.meem_problem

MEEMProblem Module
====================

This module defines the `MEEMProblem` class, which is responsible for managing individual matched eigenfunction problems. It aggregates domains and checks for boundary condition matches between them to ensure the correct mathematical problem setup.

.. automodule:: meem_problem
   :members:
   :undoc-members:

Class:
--------
.. autoclass:: meem_problem.MEEMProblem
   :members:
   :noindex:
   :undoc-members:
   :show-inheritance:

Attributes:
-----------
- `domain_list`: List[Domain] — Populated from the Geometry class, this list holds Domain instances to be checked for boundary condition matching.

Methods:
--------
.. method:: __init__(geometry: Geometry)
   :noindex:

   Initializes the MEEMProblem instance with a geometry, loading domain information from the Geometry object.
   
   :param geometry: An instance of the Geometry class containing domain information.
   :type geometry: Geometry

.. method:: match_domains() -> Dict[int, Dict[str, bool]]

   Checks boundary condition matching between consecutive domains in the `domain_list`.
   
   :returns: A dictionary with information about the matching status of each domain pair, keyed by the index. The dictionary includes a boolean value for each boundary condition check (e.g., 'top_match', 'bottom_match').

.. method:: perform_matching(matching_info: Dict[int, Dict[str, bool]]) -> bool

   Takes matching information from `match_domains` and verifies if all domains match according to the specified boundary conditions.
   
   :param matching_info: A dictionary containing the matching status between domain pairs.
   :returns: True if all domains match successfully, False otherwise.
   
   If matching fails, the method prints the index at which matching failed.

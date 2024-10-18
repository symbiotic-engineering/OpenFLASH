.. currentmodule:: package.meem_problem

MEEM_problem Module
====================

This module defines the `MEEM_problem` class, which is responsible for managing individual matched eigenfunction problems. It aggregates domains and checks for boundary condition matches between them.

.. automodule:: meem_problem
   :members:
   :undoc-members:

Classes:
--------
.. autoclass:: meem_problem.MEEM_problem
   :members:
   :noindex:
   :undoc-members:
   :show-inheritance:

Attributes:
-----------
- `domain_list`: Dict[str, Domain] — A list of Domain instances populated from the Geometry class.

Methods:
--------
.. method:: __init__(geometry: Geometry)
   :noindex:

   Initializes the MEEM_problem with a geometry instance.
   
   :param geometry: An instance of the Geometry class containing domain information.
   :type geometry: Geometry

.. method:: match_domains() -> Dict[str, Dict[str, bool]]

   Checks boundary condition matching between the domains in the domain list.
   
   :returns: A dictionary indicating the matching results for each domain.

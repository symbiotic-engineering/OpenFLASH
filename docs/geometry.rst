.. currentmodule:: package.geometry

Geometry Module
================

This module defines the `Geometry` class responsible for creating domain objects based on specified coordinates.

.. automodule:: geometry
   :members:
   :undoc-members:

Class:
--------

.. autoclass:: geometry.Geometry
   :members:
   :noindex:
   :undoc-members:
   :show-inheritance:

Attributes:
-----------
- `r_coordinates`: Dict[float] — Radial coordinates.
- `z_coordinates`: Dict[float] — Vertical coordinates.
- `domain_params`: List[Dict] — Parameters for domain creation.

Methods:
--------
.. method:: __init__(r_coordinates, z_coordinates, domain_params)
   :noindex:
   
   Initializes the Geometry class with the given coordinates and parameters.
   
   :param r_coordinates: The radial coordinates.
   :type r_coordinates: Dict[float]
   :param z_coordinates: The vertical coordinates.
   :type z_coordinates: Dict[float]
   :param domain_params: The parameters for creating domains.
   :type domain_params: List[Dict]

.. method:: make_domain_list() -> Dict[str, Domain]
   
   Creates a list of domain objects.
   
   :returns: A dictionary of domain objects.

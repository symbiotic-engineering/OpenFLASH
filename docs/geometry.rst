.. currentmodule:: package.geometry

Geometry Module
===============

This module defines the `Geometry` class, which is responsible for creating and managing domain objects within the specified coordinates of a given geometry. Domains within the geometry are characterized by unique radial and vertical coordinates and parameterized to enable computation of eigenfunctions and potentials within each region.

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
- `r_coordinates`: Dict[str, float] — Dictionary of radial coordinates specifying positions within the geometry.
- `z_coordinates`: Dict[str, float] — Dictionary of vertical coordinates that define height positions within the geometry.
- `domain_params`: List[Dict] — A list of dictionaries where each dictionary contains parameters for initializing a `Domain` object, including harmonics, boundary conditions, and physical properties.

Methods:
--------
.. method:: __init__(r_coordinates, z_coordinates, domain_params)
   :noindex:
   
   Initializes the Geometry class with the given coordinates and parameters.
   
   :param r_coordinates: The radial coordinates defining the radial positions within the geometry.
   :type r_coordinates: Dict[str, float]
   :param z_coordinates: The vertical coordinates defining height positions.
   :type z_coordinates: Dict[str, float]
   :param domain_params: A list of dictionaries, each containing parameters for creating a `Domain`.
   :type domain_params: List[Dict]

.. method:: make_domain_list() -> Dict[int, Domain]
   
   Creates a dictionary of `Domain` objects, where each key is an integer index mapping to a `Domain`.
   
   :returns: A dictionary of `Domain` objects by index.

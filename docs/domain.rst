.. currentmodule:: package.domain

Domain Module
=============

This module defines the `Domain` class which represents the characteristics of a physical domain.

.. automodule:: domain
   :members:
   :undoc-members:

Classes:
--------

.. autoclass:: domain.Domain
   :members:
   :undoc-members:
   :show-inheritance:

Attributes:
-----------
- `number_harmonics`: int — Number of harmonics in the domain.
- `height`: float — Height of the domain.
- `radial_width`: float — Radial width of the domain.
- `top_BC`: float — Top boundary condition.
- `bottom_BC`: float — Bottom boundary condition.
- `category`: str — Category of the domain.

Methods:
--------
.. method:: __init__(number_harmonics, height, radial_width, top_BC, bottom_BC)

   Initializes the Domain class with specified parameters.
   
   :param number_harmonics: The number of harmonics.
   :type number_harmonics: int
   :param height: Height of the domain.
   :type height: float
   :param radial_width: Radial width of the domain.
   :type radial_width: float
   :param top_BC: Top boundary condition.
   :type top_BC: float
   :param bottom_BC: Bottom boundary condition.
   :type bottom_BC: float

.. method:: radial_eigenfunctions(r: float)

   Calculates radial eigenfunctions for a given radial coordinate.
   
   :param r: Radial coordinate.
   :type r: float
   :returns: Eigenfunction value.

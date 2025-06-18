.. _geometry-module:

===============
Geometry Module
===============

.. automodule:: geometry
   :no-members:
   :no-undoc-members: 
   :show-inheritance:

Overview
========

This module (`geometry.py`) defines the :class:`Geometry` class, which is responsible for encapsulating the dimensions and spatial configurations of the hydrodynamic problem. It manages the radial and vertical coordinates, and more importantly, orchestrates the creation and organization of :class:`~domain.Domain` objects, which represent distinct sub-regions. The :class:`Geometry` class serves as the central hub for defining the problem's spatial setup.

.. _geometry-class:

The Geometry Class
==================

.. autoclass:: Geometry
   :members:
   :undoc-members: 
   :show-inheritance:
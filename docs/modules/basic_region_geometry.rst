.. _basic_region_geometry-module:

============================
Basic Region Geometry Module
============================

.. automodule:: openflash.basic_region_geometry

.. _basic_region_geometry-overview:

Conceptual Overview
===================

The ``BasicRegionGeometry`` class is the primary **concrete implementation** of the abstract :class:`~openflash.geometry.Geometry` class. It is designed for the most common use case: a set of simple, concentric bodies whose radii are strictly increasing from the center outwards.

Its main responsibility is to take a physical description of the bodies (via a :class:`~openflash.geometry.ConcentricBodyGroup`) and automatically partition the fluid volume into a series of non-overlapping :class:`~openflash.domain.Domain` objects. This process is crucial as it translates the user-defined physical problem into the structured set of sub-regions required by the ``MEEMEngine`` for solving.

Primary Usage (Object-Oriented)
===============================

The standard way to use this class is by first defining your physical objects and then passing them to the constructor.

.. code-block:: python

   from openflash import SteppedBody, ConcentricBodyGroup, BasicRegionGeometry

   # 1. Define the physical bodies
   body1 = SteppedBody(a=np.array([5.0]), d=np.array([20.0]), heaving=True)
   body2 = SteppedBody(a=np.array([10.0]), d=np.array([10.0]), heaving=False)

   # 2. Group the bodies into an arrangement
   arrangement = ConcentricBodyGroup(bodies=[body1, body2])

   # 3. Define other parameters
   h = 100.0  # Total water depth
   NMK = [30, 30, 30] # Harmonics for inner, middle, and outer domains

   # 4. Create the Geometry object
   # This object will automatically generate the fluid domains internally.
   geometry = BasicRegionGeometry(
       body_arrangement=arrangement,
       h=h,
       NMK=NMK
   )

Alternative Usage (Vector-Based)
================================

For convenience, the classmethod :meth:`~from_vectors` allows you to create a ``BasicRegionGeometry`` instance directly from NumPy arrays without explicitly creating ``SteppedBody`` objects. This can be useful for scripting or when working with data from other sources.

.. code-block:: python

   from openflash import BasicRegionGeometry
   import numpy as np

   # Define geometry using simple arrays
   a_vals = np.array([5.0, 10.0])
   d_vals = np.array([20.0, 10.0])
   heaving_flags = [True, False]
   h = 100.0
   NMK = [30, 30, 30]

   # Create the geometry object directly from vectors
   geometry = BasicRegionGeometry.from_vectors(
       a=a_vals,
       d=d_vals,
       h=h,
       NMK=NMK,
       heaving_map=heaving_flags
   )


.. _basic_region_geometry-api:

API Reference
=============

.. autoclass:: openflash.basic_region_geometry.BasicRegionGeometry
   :members: from_vectors, domain_list, make_fluid_domains
   :undoc-members:
   :show-inheritance:
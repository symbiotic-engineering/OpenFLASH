.. _geometry-module:

===============
Geometry Module
===============

.. automodule:: openflash.geometry
   :no-members:

.. _geometry-overview:

Conceptual Overview
===================

The Geometry module provides the classes necessary to define the physical layout of the hydrodynamic problem. It acts as the bridge between the high-level description of physical objects (the :ref:`body-module`) and the low-level fluid sub-regions (the :ref:`domain-module`) used by the solver.

The conceptual hierarchy is as follows:

1.  **Body Objects**: You start by defining one or more physical structures using classes like ``SteppedBody``. Each ``Body`` has its own physical properties (radii, depths, heaving status).

2.  **BodyArrangement**: These individual ``Body`` objects are then grouped into a ``BodyArrangement``. This class organizes the collection of bodies. For most use cases, you will use the concrete ``ConcentricBodyGroup`` class.

3.  **Geometry**: Finally, a ``Geometry`` object is created from a ``BodyArrangement`` and the total water depth (`h`). The primary role of a ``Geometry`` object is to process this physical layout and generate the corresponding list of fluid ``Domain`` objects that the ``MEEMEngine`` can solve.

In summary: **Bodies -> Arrangement -> Geometry -> Domains**

.. _geometry-api:

API Reference
=============

The module contains three key classes that work together to define the problem's spatial configuration.

The Geometry Class
------------------

.. autoclass:: openflash.geometry.Geometry
   :members: fluid_domains, make_fluid_domains
   :undoc-members:
   :show-inheritance:

   The ``Geometry`` class is an **abstract base class**. You will not use this class directly, but rather one of its concrete implementations, such as ``openflash.basic_region_geometry.BasicRegionGeometry``. It establishes the core responsibility of turning a physical layout into a set of solvable fluid domains.

---

The BodyArrangement Class
-------------------------

.. autoclass:: openflash.geometry.BodyArrangement
   :members:
   :undoc-members:
   :show-inheritance:

   Like ``Geometry``, the ``BodyArrangement`` class is an **abstract base class**. It defines the required interface for any class that organizes a collection of ``Body`` objects.

---

The ConcentricBodyGroup Class
-----------------------------

.. autoclass:: openflash.geometry.ConcentricBodyGroup
   :members:
   :undoc-members:
   :show-inheritance:

   This is the primary **concrete class** you will use to group your ``SteppedBody`` objects for a standard concentric cylinder problem. It takes a list of ``Body`` objects and automatically concatenates their properties (like radii and depths) into single arrays that can be used by a ``Geometry`` object.
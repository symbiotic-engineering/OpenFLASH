.. _body-module:

===========
Body Module
===========

.. automodule:: openflash.body

.. _body-overview:

Conceptual Overview
===================

The Body module provides the classes used to define the physical structures in the hydrodynamic simulation. These classes serve as the fundamental building blocks of the problem geometry.

The user's first step in setting up a simulation is to create one or more ``Body`` objects. Each object represents a distinct physical component with its own geometric properties (like radii and depths) and dynamic properties (like whether it is heaving).

Once defined, these ``Body`` objects are collected into a :class:`~openflash.geometry.BodyArrangement` (typically a :class:`~openflash.geometry.ConcentricBodyGroup`), which is then used to initialize a :class:`~openflash.geometry.Geometry` object.

The most important and commonly used class in this module is the ``SteppedBody``.

.. _body-api:

API Reference
=============

Body (Abstract Base Class)
--------------------------

.. autoclass:: openflash.body.Body
   :members:
   :undoc-members:
   :show-inheritance:

   This is an **abstract base class** that defines the basic interface for all body types. You will not create instances of this class directly.

---

SteppedBody
-----------

.. autoclass:: openflash.body.SteppedBody
   :members:
   :undoc-members:
   :show-inheritance:

   This is the primary class for defining bodies with concentric, vertical-walled steps. A single ``SteppedBody`` can be composed of one or more steps, allowing for complex, tiered structures to be defined as a single entity.

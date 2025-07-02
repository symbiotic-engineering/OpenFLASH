.. _domain-module:

==============
Domain Module
==============

.. automodule:: domain
   :members:
   :undoc-members:
   :show-inheritance:

.. _domain-overview:

.. figure:: _static/domain_table.png
   :alt: Table of domains
   :align: center
   :width: 100%

   **Figure 1**: This table illustrates the characteristics of the domains, including exterior and an arbitrary number of concentric cylindrical interior domains.

.. figure:: _static/domain_drawing.png
   :alt: Example of how domains would look
   :align: center
   :width: 100%

   **Figure 2**: The package can model any number of concentric cylindrical domains :math:`i_1, \ldots, i_M` between exterior domains :math:`e`.

This module defines the `Domain` class, which represents the characteristics of a physical domain.

Overview
========

The `domain.py` module defines the :class:`Domain` class, which represents a discrete sub-region within the overall fluid geometry. In multi-domain problems, the total computational space is divided into these individual domains, each with its own characteristics, boundary conditions, and parameters. The :class:`Domain` class manages these properties and provides methods to access relevant geometric and physical attributes.


The Domain Class
================

.. autoclass:: Domain
   :members:
   :undoc-members: 
   :show-inheritance:
   :noindex:

   .. automethod:: __init__

   .. automethod:: _get_di

   .. automethod:: _get_a

   .. automethod:: _get_heaving

   .. automethod:: _get_r_coords

   .. automethod:: _get_z_coords
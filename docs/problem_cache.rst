.. _problem-cache-module:

====================
Problem Cache Module
====================

.. automodule:: problem_cache
   :no-members:       
   :no-undoc-members:

.. _problem-cache-overview:

Overview
========

The `problem_cache.py` module introduces the :class:`ProblemCache` class, a critical component designed to optimize the performance of the MEEM engine. This class serves as a storage for pre-computed, frequency-independent parts of the system matrix (A) and right-hand side vector (b), along with storing indices and calculation functions for terms that depend on the incident wave number (:math:`m_0`). By caching these components, MEEM Engine can efficiently re-evaluate the system for different frequencies without re-computing the entire problem from scratch, significantly reducing computation time.

.. _problem-cache-class:

The ProblemCache Class
======================

.. autoclass:: ProblemCache
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex: 
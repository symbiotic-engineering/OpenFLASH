.. _introduction:

==============
Introduction
==============

Welcome to the documentation for **OpenFLASH**.

This project provides a robust and efficient computational framework for solving complex hydrodynamic problems involving multiple regions. Leveraging the power of MEEM, it aims to accurately model fluid dynamics for applications in engineering, marine hydrodynamics, and related fields.

Key Features:
-------------
* **Multi-Domain Approach**: Divides the region into cylindrical domains, allowing for flexible geometric configurations and handling of multiple bodies with different depths. Learn more about defining domains in the :doc:`domain` module and setting up the overall geometry in :doc:`geometry`.
* **Semi-Analytical Method & Eigenfunction Expansion**: OpenFLASH employs a semi-analytical approach. The core of this method lies in the use of analytical eigenfunctions, enabling highly accurate and efficient solutions by reducing the computational burden compared to fully numerical methods. The core mathematical formulations are detailed in the :doc:`multi_equations` module.
* **Performance Optimization**: Employs caching mechanisms (see :doc:`problem_cache`) to store frequency-independent computations, enabling rapid evaluation across a range of wave frequencies and significantly reducing solving time.
    * We have profiled OpenFLASH against **Capytaine**, a leading open-source boundary element method (BEM) software. We found that **MEEM** performs significantly faster than Capytaine, achieving **10x faster computation times**.
* **Modular Design**: The codebase is structured with distinct modules for clear separation of concerns, facilitating maintainability and extensibility. Key modules include those for geometric definition, domain properties, and problem-solving.

Target Audience:
----------------
This documentation is intended for:
* Researchers and students.
* Hydrodynamicists or ocean engineers.
* Developers interested in understanding, contributing to, or extending the codebase.

Getting Started:
----------------
To begin using the OpenFLASH, we recommend the following steps:

1.  **Project Setup**: Ensure you have followed the installation instructions (`installation.rst`).
2.  **Application Walkthrough**: Explore a practical example of how to use the solver by following the :doc:`app_walk` and tutorial_walk.ipynb guide.
3.  **API Reference**: For detailed information on specific classes, functions, and their parameters, refer to the individual module documentation listed in the sidebar (e.g., :doc:`domain`, :doc:`multi_equations`, :doc:`geometry`, :doc:`problem_cache`).
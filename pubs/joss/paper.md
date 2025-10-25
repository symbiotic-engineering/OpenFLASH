# OpenFLASH: A Python Package for Matched Eigenfunctions Methods

Kapil Khanal, Rebecca McCabe, Hope Best, Ruiyang Jiang, Maha Haji

## Summary

OpenFLASH is a Python package designed to quickly solve boundary value problems using the matched eigenfunction expansion method, particularly in scenarios involving multiple concentric cylindrical regions. This method can reduce the runtime by an order of magnitude compared to traditional Boundary Element Method (BEM) solvers, making it more fitting for design optimization studies of floating structures such as wave energy converters (WEC).

It provides a modular framework for:

  * Defining complex geometries 
  * Setting up multi-domain problems with appropriate boundary and matching conditions 
  * Performing numerical computations to determine the hydrodynamic coefficients 
  * Analyzing the resulting potential and velocity fields 

The package includes tools for:

  * Data storage using `xarray` 
  * A comprehensive testing suite using `unittest` 
  * An interactive simulation application built with Streamlit (`docs/app.py`) for real-time parameter adjustment and visualization 
  * Extensive documentation built with Sphinx 

OpenFLASH implements a sequential workflow from geometry definition to result visualization.  It is a computationally efficient alternative to BEM solvers that is well-suited for applications in floating body dynamics where boundary value problems in cylindrical coordinates arise. 

## Statement of Need

Semi-analytical methods, such as the matched eigenfunction expansion method, offer an efficient and often accurate alternative to purely numerical techniques like Boundary Element Methods (BEM) for certain classes of boundary value problems.  These methods are particularly advantageous when dealing with geometries that can be decomposed into simpler sub-regions where analytical solutions (in eigenfunction expansions) are known or can be derived.  By matching these solutions at the interfaces between sub-regions, a system of linear equations can be formed and solved for the unknown expansion coefficients. 

Wave energy converters (WEC) hold significant promise for transforming the oscillatory motion of waves into usable energy, offering high predictability and enhanced energy security that complements other renewable sources like wind and solar power.  However, the optimization of WECs has been hindered by the substantial computational costs associated with modeling their hydrodynamic interactions in waves.  This project aims to address this challenge by developing `open_flash`, an open-source and computationally efficient software tool for modeling WECs using semi-analytical methods. 

OpenFLASH aims to provide a robust and user-friendly Python implementation of this methodology, specifically tailored for problems involving connected cylindrical domains.  The package is designed to handle multi-domain problems, including exterior domains extending to infinity and interior domains with specific radial extents, each with defined top and bottom boundary conditions.  The computational workflow begins with defining the geometry and problem parameters, followed by assembling and solving the linear system, calculating hydrodynamic coefficients and potentials, storing the results, and finally visualizing them.  This specialization can lead to more efficient problem setup and solution, particularly in fields like marine hydrodynamics and the burgeoning field of wave energy technology.  The package addresses the need for a tool that bridges the gap between analytical derivations and numerical computation for this important class of problems.  Furthermore, it provides tools for managing, testing, interactively visualizing, documenting, and outlining its computational process. 

## Functionality

OpenFLASH offers the following key functionalities:

  * **Modular Geometry Definition**: The `Geometry` class represents the physical setup of the problem, managing radial (`r_coordinates`) and vertical (`z_coordinates`) dimensions and parameters for individual sub-regions (domains).  It is designed to flexibly support any number of concentric cylinders, making it well-suited for modeling a wide range of Wave Energy Converter (WEC) configurations.  The class is responsible for creating a list of `Domain` objects based on the provided input. 
  * **Domain Representation**: The `Domain` class represents a sub-region within the defined geometry.  Each domain possesses properties such as height, radial width, and specific boundary conditions at its top and bottom.  The class also calculates the spatial coordinates relevant to the domain. 
  ![Domain class](pubs/joss/figures/domain.png)
  * **Problem Setup**: The `MEEMProblem` class sets up the computational problem by defining the relevant frequencies and modes of analysis.  It links the `Geometry` object to the problem definition and maintains a list of the `Domain` objects necessary for the subsequent computations. 
  * **MEEM Computation Engine**: The `MEEMEngine` class is the core of the package, responsible for managing the matched eigenfunction expansion method.  It assembles the system matrix (A) and the right-hand side vector (b) based on the geometry, boundary conditions, and matching conditions.  It then solves the resulting linear system (Ax=b) to determine the unknown coefficients of the eigenfunction expansions.  Additionally, it calculates hydrodynamic coefficients and the potential fields within each domain. 
  ![MEEMEngine class](pubs/joss/figures/meem_engine.png)
  * **Problem Cache for Efficient Computation**: The `ProblemCache` class is designed to enhance the computational efficiency of the `MEEMEngine` class significantly.  Solving for the unknown expansion coefficients involves assembling and solving a system of linear equations, Ax = b.  Many terms within the system matrix A and the right-hand side vector b are independent of m0.  Recomputing these constant terms for every m0 would be computationally expensive.  The ProblemCache addresses this by pre-computing and storing the A matrix and b vector templates, specifically the parts that are independent of m0.  It then identifies and stores the indices of the m0-dependent terms, along with references to the functions required to calculate their values.  This approach allows quick updates of only the necessary m0-dependent entries for each new m0 value, rather than reassembling the entire system.  Furthermore, the `ProblemCache` stores precomputed arrays for mk and Nk.  By providing these pre-computed arrays and the functions to calculate the m0-dependent terms, the `ProblemCache` reduces overall computation time. 
  * **Results Management**: The `Results` class provides a structured way to store and organize the output of the MEEM computations using the `xarray` library, adhering to conventions similar to those used in the `Capytaine` library.  It offers methods for storing various results, including velocity potentials and hydrodynamic coefficients, and facilitates accessing and exporting these results to NetCDF (`.nc`) files. 
  * **Visualization**: The package includes visualization capabilities, as demonstrated in the Streamlit application. 
  * **Documentation**: The package utilizes Sphinx to generate comprehensive documentation.  The documentation includes a tutorial that guides users through the process of using the package and explains its capabilities.  The documentation is built in HTML format for easy accessibility. 
  * **Interactive Simulation and Visualization**: A Streamlit application (`docs/app.py`) provides a graphical user interface for interacting with OpenFLASH.  Users can define problem parameters through the GUI, run simulations, and visualize the resulting potential and velocity fields in real-time.  This interactive tool enhances the usability and accessibility of the package. 
  * **Testing Suite**: The package includes a comprehensive suite of unit tests (`tests` directory) using the `unittest` framework.  These tests cover the core functionalities, ensuring the reliability and correctness of the code across different modules. 

## Examples

The `tests` directory contains several unit tests that demonstrate the usage of different modules within OpenFLASH.  The `docs` directory contains the `app.py` script, which provides a fully functional interactive application built with Streamlit.  This application allows users to define multi-domain problems by inputting parameters such as water height, body dimensions, heaving states, and slant vectors for each region.  Users can also control the number of harmonics and spatial resolution.  The app then computes the hydrodynamic coefficients and generates plots of the real and imaginary parts of the total, homogeneous, and particular potentials, as well as the radial and vertical velocity potentials.  The `docs` directory also contains the Sphinx documentation, including a tutorial on how to use the package. 

## Impact

OpenFLASH provides a specialized and powerful tool for researchers and engineers working on boundary value problems in domains with connected cylindrical geometries, with a particular emphasis on advancing the field of wave energy conversion. Its modular design and focus on the matched eigenfunction expansion method offer several key benefits for WEC research:

  * **Accessibility for WEC Researchers**: The translation of hydrodynamic models into a user-friendly Python package, coupled with comprehensive documentation, clear tutorials, and optimized code, will make complex WEC simulations accessible to a wider range of researchers, lowering the barrier to entry in this important field. 
  * **Accelerating WEC Innovation Through Efficient Modeling**: OpenFLASH offers a computationally efficient, semi-analytical modeling approach that enables rapid analysis and optimization of Wave Energy Converter (WEC) configurations. By streamlining design exploration and reducing development bottlenecks, it supports faster technology deployment, cost reduction, and innovation, contributing to the advancement of renewable energy and the growth of the Blue Economy. 
  * **Open-Source and Community-Driven**: As an open-source project, OpenFLASH encourages community contributions, further enhancing its capabilities and ensuring its long-term sustainability and utility for the wave energy research community. The focus on clear documentation and user-friendly implementation promotes collaboration and wider adoption. 

The initial development of hydrodynamic models lays the groundwork for this package, and the ongoing work to refine the code structure, optimize usability, incorporate diverse WEC geometries, and expand documentation will ensure that OpenFLASH becomes a valuable asset for the wave energy research community. 

## Acknowledgements

We acknowledge the foundational work on matched eigenfunction methods as presented in the cited references. The development of OpenFLASH has been supported by the resources and expertise within the SEA LAB, led by Professor Maha N. Haji. We also acknowledge the contributions of the following students to the development of this software: Kapil Khanal, Rebecca McCabe, Hope Best, Ruiyang Jiang, Yinghui Bimali, En Lo, and Collin Treacy. We also acknowledge the ongoing collaborations with Jessica Nguyen, Clint Chester Reyes, and Brittany Lydon for their work on validating and extending the capabilities of OpenFLASH to elliptical and Cartesian coordinate systems.

## References

  * Chatjigeorgiou, I. K. (2018). *Analytical Methods in Marine Hydrodynamics*. Cambridge University Press. [https://doi.org/10.1017/9781316838983](https://doi.org/10.1017/9781316838983) 
  * Chau, F. P., & Yeung, R. W. (2010). *Inertia and Damping of Heaving Compound Cylinders*. Presented at the 25th International Workshop on Water Waves and Floating Bodies, Harbin, China. [https://www.academia.edu/73219479/Inertia\_and\_Damping\_of\_Heaving\_Compound\_Cylinders\_Fun](https://www.academia.edu/73219479/Inertia_and_Damping_of_Heaving_Compound_Cylinders_Fun) 
  * Chau, F. P., & Yeung, R. W. (2013). Inertia, Damping, and Wave Excitation of Heaving Coaxial Cylinders. In *ASME 2012 31st International Conference on Ocean, Offshore and Arctic Engineering* (pp. 803–813). American Society of Mechanical Engineers Digital Collection. [https://doi.org/10.1115/OMAE2012-83987](https://doi.org/10.1115/OMAE2012-83987) 
  * Yeung, R. W. (1981). Added mass and damping of a vertical cylinder in finite-depth waters. *Appl. Ocean Res.*, *3*(3), 119–133. [https://doi.org/10.1016/0141-1187(81)90101-2](https://doi.org/10.1016/0141-1187\(81\)90101-2) 
  * Son, D., Belissen, V., & Yeung, R. W. (2016). Performance validation and optimization of a dual coaxial-cylinder ocean-wave energy extractor. *Renew. Energy*, *92*, 192–201. [https://doi.org/10.1016/j.renene.2016.01.032](https://doi.org/10.1016/j.renene.2016.01.032) 
  * Kokkinowrachos, K., Mavrakos, S., & Asorakos, S. (1986). Behaviour of vertical bodies of revolution in waves. *Ocean Eng.*, *13*(6), 505–538. [https://doi.org/10.1016/0029-8018(86)90037-5](https://www.google.com/search?q=https://doi.org/10.1016/0029-8018\(86\)90037-5)
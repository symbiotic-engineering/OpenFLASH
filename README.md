# open-flash

matlab: see `hydro/matlab/run_MEEM.m` for the symbolic and numeric code, see `test/matlab/` for some scripts to get results.

python: see `package/src/` for some helper functions.

`test/time_comparison` for time comparisons of BEM (Capytaine), and `dev/` for various matlab experiments.

## About The Open-Flash Project

The Open-Flash project is a Python package designed for solving boundary value problems using eigenfunction expansion methods. It provides a modular framework for defining complex geometries, setting up multi-domain problems, performing numerical computations, and analyzing results, particularly in fields like fluid dynamics.

When referencing this work, please reference our `docs/citations.rst`

**License:**

This project is licensed under the MIT License. See the `LICENSE` file for details.

**How to Run the Python Code:**

1.  **Install the package:**
    ```bash
    pip install git+[https://github.com/symbiotic-engineering/semi-analytical-hydro.git](https://github.com/symbiotic-engineering/semi-analytical-hydro.git)
    ```
2.  **Install dependencies:** Navigate to the project directory (if you cloned it) and run:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Explore the `docs/` and `package/test` directory:** These directories contain scripts and notebooks demonstrating how to use the open-flash framework for different problems. Run these examples to understand the workflow.
4.  **Refer to the documentation in the `docs/` directory:** The documentation provides detailed information on the different modules and classes within the `open-flash` package.

## References

The following publications are relevant to this package:

1. I. K. Chatjigeorgiou, *Analytical Methods in Marine Hydrodynamics*. Cambridge: Cambridge University Press, 2018. doi: 10.1017/9781316838983.
2. F. P. Chau and R. W. Yeung, “Inertia and Damping of Heaving Compound Cylinders,” presented at the 25th International Workshop on Water Waves and Floating Bodies, Harbin, China, Jan. 2010. Accessed: Sep. 27, 2023. [Online]. Available: https://www.academia.edu/73219479/Inertia_and_Damping_of_Heaving_Compound_Cylinders_Fun
3. F. P. Chau and R. W. Yeung, “Inertia, Damping, and Wave Excitation of Heaving Coaxial Cylinders,” presented at the ASME 2012 31st International Conference on Ocean, Offshore and Arctic Engineering, American Society of Mechanical Engineers Digital Collection, Aug. 2013, pp. 803–813. doi: 10.1115/OMAE2012-83987.
4. R. W. Yeung, “Added mass and damping of a vertical cylinder in finite-depth waters,” *Appl. Ocean Res.*, vol. 3, no. 3, pp. 119–133, Jul. 1981, doi: 10.1016/0141-1187(81)90101-2.
5. D. Son, V. Belissen, and R. W. Yeung, “Performance validation and optimization of a dual coaxial-cylinder ocean-wave energy extractor,” *Renew. Energy*, vol. 92, pp. 192–201, Jul. 2016, doi: 10.1016/j.renene.2016.01.032.
6. K. Kokkinowrachos, S. Mavrakos, and S. Asorakos, “Behaviour of vertical bodies of revolution in waves,” *Ocean Eng.*, vol. 13, no. 6, pp. 505–538, Jan. 1986, doi: 10.1016/0029-8018(86)90037-5.
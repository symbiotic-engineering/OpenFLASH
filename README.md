# OpenFLASH ⚡️
Open-source Flexible Library for Analytical and Semi-analytical Hydrodynamics 

## About The OpenFLASH Project

The OpenFLASH project is a Python package designed for solving hydrodynamic boundary value problems using eigenfunction expansion methods. It provides a modular framework for defining complex geometries, setting up multi-domain problems, performing numerical computations, and analyzing results, particularly for linear potential flow hydrodynamics. It can be significantly faster than Boundary Element Method calculations although is restricted to certain geometries (currently axisymmetric compound cylinders).  

When referencing this work, please reference our `docs/citations.rst`

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Installation

Option 1: via pypi (recommended for users)

```bash
pip install open-flash
```


Option 2: via git (recommended for developers)
1.  **Clone the repository:**
```bash
git clone https://github.com/symbiotic-engineering/OpenFLASH.git
cd OpenFLASH
```
2.  **Install the package:**
```bash
pip install -e .
```
3.  **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Usage

Please see our [documentation](https://symbiotic-engineering.github.io/OpenFLASH/) and [tutorial notebook](https://symbiotic-engineering.github.io/OpenFLASH/tutorial_walk.html). The documentation provides detailed instructions and API reference for the different modules and classes within the `openflash` package.

If you prefer not to utilize the package programatically, the model can also be run with an [interactive web app](http://symbiotic-engineering.github.io/OpenFLASH/app_streamlit.html) (the site may take several seconds to load).

## Theory
Please see our [equations documentation](https://symbiotic-engineering.github.io/OpenFLASH/multi_equations.html) and [references](https://symbiotic-engineering.github.io/OpenFLASH/citations.html) for mathematical background and derivations as well as validation information.

## MATLAB
We also have a MATLAB code version, although the Python package is intended as primary for future development. MATLAB only supports bodies consisting of 2 concentric cylinders, rather than the arbitrary N concentric cylinders in the Python package.

See `matlab/src/run_MEEM.m` for the symbolic and numeric code, see `matlab/test/` for some scripts to get results, and `matlab/dev` for various matlab experiments.


<!-- ## References

The following publications are relevant to this package:

1. I. K. Chatjigeorgiou, *Analytical Methods in Marine Hydrodynamics*. Cambridge: Cambridge University Press, 2018. doi: 10.1017/9781316838983.
2. F. P. Chau and R. W. Yeung, “Inertia and Damping of Heaving Compound Cylinders,” presented at the 25th International Workshop on Water Waves and Floating Bodies, Harbin, China, Jan. 2010. Accessed: Sep. 27, 2023. [Online]. Available: https://www.academia.edu/73219479/Inertia_and_Damping_of_Heaving_Compound_Cylinders_Fun
3. F. P. Chau and R. W. Yeung, “Inertia, Damping, and Wave Excitation of Heaving Coaxial Cylinders,” presented at the ASME 2012 31st International Conference on Ocean, Offshore and Arctic Engineering, American Society of Mechanical Engineers Digital Collection, Aug. 2013, pp. 803–813. doi: 10.1115/OMAE2012-83987.
4. R. W. Yeung, “Added mass and damping of a vertical cylinder in finite-depth waters,” *Appl. Ocean Res.*, vol. 3, no. 3, pp. 119–133, Jul. 1981, doi: 10.1016/0141-1187(81)90101-2.
5. D. Son, V. Belissen, and R. W. Yeung, “Performance validation and optimization of a dual coaxial-cylinder ocean-wave energy extractor,” *Renew. Energy*, vol. 92, pp. 192–201, Jul. 2016, doi: 10.1016/j.renene.2016.01.032.
6. K. Kokkinowrachos, S. Mavrakos, and S. Asorakos, “Behaviour of vertical bodies of revolution in waves,” *Ocean Eng.*, vol. 13, no. 6, pp. 505–538, Jan. 1986, doi: 10.1016/0029-8018(86)90037-5. -->
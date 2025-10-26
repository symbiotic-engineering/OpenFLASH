# OpenFLASH ⚡️
Open-source Flexible Library for Analytical and Semi-analytical Hydrodynamics 

## About The OpenFLASH Project

The OpenFLASH project is a Python package designed for solving hydrodynamic boundary value problems using eigenfunction expansion methods. It provides a modular framework for defining complex geometries, setting up multi-domain problems, performing numerical computations, and analyzing results, particularly for linear potential flow hydrodynamics. It can be significantly faster than Boundary Element Method calculations although is restricted to certain geometries (currently axisymmetric compound cylinders).  

When referencing this work, please reference our `citation.cff`.

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

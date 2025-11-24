.. _installation:

============
Installation
============

This section provides detailed instructions on how to set up **OpenFLASH** on your local machine.

Prerequisites
-------------
Before you begin, ensure you have the following software installed on your system:

* **Python**: Version 3.9 or higher. You can download the latest version from `python.org <https://www.python.org/downloads/>`_.
* **pip**: The Python package installer, which typically comes bundled with Python installations.
* **Anaconda**: Popular Python distribution for scientific computing that simplifies environment management.

Recommended Setup: Virtual Environments
---------------------------------------
It is **strongly recommended** to use a virtual environment to manage the project's dependencies. This isolates the OpenFLASH's packages from your system-wide Python installation, preventing potential conflicts with other Python projects.

Choose one of the following methods to set up your virtual environment:

.. tabs::

   .. tab:: Using `venv` (Python's built-in)

      1.  **Create a virtual environment**:
          Open your terminal or command prompt. Navigate to your desired location (e.g., where you plan to clone the openFlASH repository). Then, run the following command:

          .. code-block:: bash

              python3 -m venv openflash_project_env

          This creates a new directory named `openflash_project_env` containing the virtual environment files.

      2.  **Activate the virtual environment**:

          * **macOS / Linux**:
              .. code-block:: bash

                  source openflash_project_env/bin/activate

          * **Windows (Command Prompt)**:
              .. code-block:: batch

                  openflash_project_env\Scripts\activate.bat

          * **Windows (PowerShell)**:
              .. code-block:: powershell

                  .\openflash_project_env\Scripts\Activate.ps1

          Your terminal prompt should change to indicate that the virtual environment is active (e.g., `(openflash_project_env)` will appear at the beginning of your prompt).

   .. tab:: Using `conda`

      1.  **Create a Conda environment**:
          If you have Anaconda installed, you can create a dedicated environment for the project:

          .. code-block:: bash

              conda create -n openflash_project_env python=3.9  # You can specify your preferred Python version

      2.  **Activate the Conda environment**:

          .. code-block:: bash

              conda activate openflash_project_env

          Your terminal prompt will change to show the active environment (e.g., `(openflash_project_env)`).

Installation via PyPI (pip)
---------------------------
**OpenFLASH** can be installed directly from PyPI using `pip`.

1.  **Activate your virtual environment**:
    Ensure your chosen virtual environment (created with `venv` or `conda`) is active.

2.  **Install OpenFLASH**:
    Run the following command in your activated environment:

    .. code-block:: bash

        pip install open-flash

    This will download and install the latest stable version of OpenFLASH and its dependencies.

Installing via Conda
---------------------

You can install **OpenFLASH** directly from the `sea-lab` channel on Anaconda.org by running:

.. code-block:: bash

    conda install sea-lab::open-flash

This will install **OpenFLASH** and all necessary dependencies into your current conda environment.

.. note::

   Ensure your conda environment is activated before running the command.

Verification (Optional)
-----------------------
To quickly verify that your installation was successful and core dependencies are available, you can open a Python interpreter within your activated environment and try importing some modules:

.. code-block:: python

    >>> import numpy
    >>> import scipy
    >>> import matplotlib
    >>> print("All core dependencies imported successfully!")
    >>> exit()

Troubleshooting
---------------
* ``Command python3 not found`` or similar errors: Ensure Python is correctly installed and added to your system's PATH. On some systems, ``python`` might refer to Python 2, and ``python3`` to Python 3.
* ``pip install -r requirements.txt`` fails:
    * Check your internet connection.
    * For specific compilation errors related to scientific packages (e.g., ``scipy``), you may need to install system-level build tools (like ``build-essential`` on Linux or Xcode Command Line Tools on macOS) or refer to the official documentation of the problematic package.
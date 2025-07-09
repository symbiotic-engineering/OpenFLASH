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

Installation via PyPI (pip)
---------------------------
**OpenFLASH** can also be installed directly from PyPI using `pip`. This is the recommended method for users who do not intend to modify the OpenFLASH source code.

1.  **Activate your virtual environment**:
    Ensure your chosen virtual environment (created with `venv` or `conda`) is active.

2.  **Install OpenFLASH**:
    Run the following command in your activated environment:

    .. code-block:: bash

        pip install open-flash

    This will download and install the latest stable version of OpenFLASH and its dependencies.

Installing via Conda
---------------------

You can install **OpenFLASH** directly from the `hopeonthestack` channel on Anaconda.org by running:

.. code-block:: bash

    conda install hopeonthestack::open-flash

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
* **`Command 'python3' not found` or similar errors**: Ensure Python is correctly installed and added to your system's PATH. On some systems, `python` might refer to Python 2, and `python3` to Python 3.
* **`pip install -r requirements.txt` fails**:
    * Check your internet connection.
    * For specific compilation errors related to scientific packages (e.g., `scipy`), you may need to install system-level build tools (like `build-essential` on Linux or Xcode Command Line Tools on macOS) or refer to the official documentation of the problematic package.
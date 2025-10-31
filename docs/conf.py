# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../package/src/openflash'))
sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'OpenFLASH'
copyright = '2025, SEA Lab'
author = 'SEA Lab'

# The full version, including alpha/beta/rc tags
from importlib.metadata import version as pkg_version, PackageNotFoundError

try:
    release = pkg_version("open-flash")
except PackageNotFoundError:
    release = "0+unknown"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'nbsphinx', 
              'sphinx.ext.mathjax', 'sphinx_design', 'sphinx.ext.viewcode', 
              'sphinx_tabs.tabs', 'sphinx_copybutton', 'sphinx_last_updated_by_git']

# Ensure Jupyter notebooks are copied as part of the build process
nbsphinx_allow_errors = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_logo = '_static/SEALab_Logo_Light_202101_120ht.png'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_context = {
    "display_github": True,
    "github_user": "symbiotic-engineering",
    "github_repo": "OpenFLASH",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

html_theme_options = {
    'version_selector': True,
}
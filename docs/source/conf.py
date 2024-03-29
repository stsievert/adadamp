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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = "adadamp"
copyright = "2020, Scott Sievert"
author = "Scott Sievert"

intersphinx_mapping = {'torch': ('https://pytorch.org/docs/stable', None),
                       'numpy': ('https://numpy.org/doc/stable/', None)}

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
#  extensions = ["sphinx.ext.autodoc", "numpydoc", "sphinx.ext.imgmath"]
#  imgmath_image_format = "svg"
extensions = ["sphinx.ext.autodoc", "sphinx.ext.intersphinx", "sphinx.ext.napoleon", "sphinx.ext.autosummary"]
autosummary_generate = True
autosummary_generate_overwrite = True

napoleon_numpy_docstring = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Lifted from https://github.com/dask/dask-jobqueue/pull/301/files
#
# Temporary work-around for spacing problem between parameter and parameter
# type in the doc, see https://github.com/numpy/numpydoc/issues/215. The bug
# has been fixed in sphinx (https://github.com/sphinx-doc/sphinx/pull/5976) but
# through a change in sphinx basic.css except rtd_theme does not use basic.css.
# In an ideal world, this would get fixed in this PR:
# https://github.com/readthedocs/sphinx_rtd_theme/pull/747/files
def setup(app):
    app.add_stylesheet("basic.css")

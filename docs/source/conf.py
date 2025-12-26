# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
project = 'causers'
copyright = '2025, James Nordlund'
author = 'James Nordlund'
version = '0.5.1'
release = '0.5.1'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',      # REQ-002: Extract docstrings from Python modules
    'sphinx.ext.napoleon',     # Parse NumPy/Google-style docstrings
    'sphinx.ext.viewcode',     # Link to source code
    'sphinx.ext.intersphinx',  # Cross-reference external docs (Polars, NumPy)
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Autodoc configuration ---------------------------------------------------
autodoc_member_order = 'bysource'        # Preserve source file order
autodoc_typehints = 'description'        # Show type hints in description
autodoc_class_signature = 'separated'    # Separate class signature from docstring
napoleon_google_docstring = False        # Disable Google style
napoleon_numpy_docstring = True          # Enable NumPy style (matches existing docstrings)
napoleon_include_init_with_doc = True    # Include __init__ docstrings

# -- Intersphinx configuration -----------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'polars': ('https://docs.pola.rs/api/python/stable', None),
    'numpy': ('https://numpy.org/doc/stable', None),
}

# -- HTML output options -----------------------------------------------------
html_theme = 'sphinx_rtd_theme'          # REQ-003: RTD theme
html_static_path = ['_static']
html_title = 'causers Documentation'
html_short_title = 'causers'
html_logo = None                         # Optional: add logo later
html_favicon = None                      # Optional: add favicon later

# -- Extension options -------------------------------------------------------
# Do NOT enable doctest auto-execution (REQ-031)
# doctest is NOT in extensions list

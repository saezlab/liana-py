# Configuration file for the Sphinx documentation builder.
import sys
from pathlib import Path
from datetime import datetime

import liana  # noqa: E402

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

project = 'liana'
copyright = f'{datetime.now():%Y}, Saezlab'
author = 'Daniel Dimitrov'
version = liana.__version__

# -- General configuration
extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosectionlabel',
    'numpydoc',
    'nbsphinx',
    'IPython.sphinxext.ipython_console_highlighting'
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']
autosummary_generate = True
templates_path = ['_templates']

# -- Options for HTML output
master_doc = 'index'

html_theme = 'furo'
html_static_path = ["_static"]
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#2980B9",
        "color-brand-content": "#2980B9",
    },
    "dark_css_variables": {
        "color-brand-primary": "#2980B9",
        "color-brand-content": "#2980B9",
    },
}
html_context = dict(
    display_github=True,
    github_user='saezlab',
    github_repo='liana-py',
    github_version='main',
    conf_py_path='/docs/source/',
)
html_show_sphinx = False
html_logo = '_static/logo.png'
html_favicon = '_static/logo.png'
html_css_files = ['custom.css']

# -- Options for EPUB output
epub_show_urls = 'footnote'
nbsphinx_execute = 'never'

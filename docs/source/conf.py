# Configuration file for the Sphinx documentation builder.
import os
import sys
from pathlib import Path
from datetime import datetime

HERE = Path(__file__).parent
sys.path.insert(0, f"{HERE.parent.parent}")

def read(rel_path: str) -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path)) as fp:
        return fp.read()

def get_version(rel_path: str) -> str:
    for line in read(rel_path).splitlines():
        if line.startswith("version ="):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")

project = 'liana'
copyright = f'{datetime.now():%Y}, Saezlab'
author = 'Daniel Dimitrov'
release = get_version("../../pyproject.toml")
version = release

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

html_theme = 'sphinx_rtd_theme'
html_static_path = ["_static"]
html_theme_options = dict(
    logo_only=True,
    display_version=True,
)
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

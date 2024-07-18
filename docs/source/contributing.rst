Contributing Guide
==================

Scanpy provides extensive `developer documentation <https://scanpy.readthedocs.io/en/stable/dev/index.html>`_, most of which applies to this repo, too. This document will not reproduce the entire content from there. Instead, it aims at summarizing the most important information to get you started on contributing.

We assume that you are already familiar with git and with making pull requests on GitHub. If not, please refer to the `developer documentation <https://scanpy.readthedocs.io/en/stable/dev/index.html>`_.

Installing Dev Dependencies
----------------------------

In addition to the packages needed to *use* this package, you need additional Python packages to *run tests* and *build the documentation*. It's easy to install them using pip:

.. code-block:: bash

    git clone https://github.com/saezlab/liana-py
    cd liana
    pip install -e ".[full]"

Code-style
----------

This package uses `pre-commit <https://pre-commit.com/>`_ to enforce consistent code-styles. On every commit, pre-commit checks will either automatically fix issues with the code or raise an error message.

To enable pre-commit locally, simply run

.. code-block:: bash

    pre-commit install

Most editors also offer an *autoformat on save* feature. Consider enabling this option for `black <https://black.readthedocs.io/en/stable/integrations/editors.html>`_ and `prettier <https://prettier.io/docs/en/editors.html>`_.

Writing Tests
-------------

This package uses `pytest <https://docs.pytest.org/en/8.2.x/>`_ for automated testing. Please write `tests <https://scanpy.readthedocs.io/en/latest/dev/testing.html#writing-tests>`_ for every function added to the package.

Most IDEs integrate with pytest and provide a GUI to run tests. Alternatively, you can run all tests from the command line by executing

.. code-block:: bash

    pytest

in the root of the repository. Continuous integration will automatically run the tests on all pull requests and upon merge into the main branch.

Publishing a Release
--------------------

Updating the Version Number
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before making a release, you need to update the version number. Please adhere to `Semantic Versioning <https://semver.org/>`_, in brief

    Given a version number MAJOR.MINOR.PATCH, increment the:

    1.  MAJOR version when you make incompatible API changes,
    2.  MINOR version when you add functionality in a backwards compatible manner, and
    3.  PATCH version when you make backwards compatible bug fixes.

    Additional labels for pre-release and build metadata are available as extensions to the MAJOR.MINOR.PATCH format.

We use `bump2version <https://pypi.org/project/bump2version/>`_ to automatically update the version number in all places and automatically create a git tag. Run one of the following commands in the root of the repository

.. code-block:: bash

    bump2version patch
    bump2version minor
    bump2version major

Once you are done, run

.. code-block:: bash

    git push --tags

to publish the created tag on GitHub.

Building and Publishing the Package on PyPI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Python packages are not distributed as source code, but as *distributions*. The most common distribution format is the so-called *wheel*. To build a *wheel*, run

.. code-block:: bash

    python -m build

This command creates a *source archive* and a *wheel*, which are required for publishing your package to `PyPI <https://pypi.org/project/liana/>`_. These files are created directly in the root of the repository.

Before uploading them to `PyPI <https://pypi.org/project/liana/>`_, you can check that your *distribution* is valid by running:

.. code-block:: bash

    twine check dist/*

and finally publishing it with:

.. code-block:: bash

    twine upload dist/*

Provide your username and password when requested and then go check out your package on `PyPI <https://pypi.org/project/liana/>`_!

For more information, refer to the `python packaging tutorial <https://packaging.python.org/en/latest/tutorials/packaging-projects/#generating-distribution-archives>`_ and `pypi-feature-request <https://github.com/scverse/cookiecutter-scverse/issues/88>`_.

Writing Documentation
----------------------

Please write documentation for new or changed features and use-cases. This project uses `sphinx <https://www.sphinx-doc.org/en/master/>`_ with the following features:
- `Numpy-style docstrings <https://numpydoc.readthedocs.io/en/latest/>`_
- `Sphinx autodoc typehints <https://github.com/agronholm/sphinx-autodoc-typehints>`_, to automatically reference annotated input and output types
- Docs use the `furo <https://pradyunsg.me/furo/quickstart/>`_ theme.

See the `scanpy developer docs <https://scanpy.readthedocs.io/en/latest/dev/documentation.html>`_ for more information on how to write documentation.

Upon every commit to the main branch, the documentation will be automatically built and published to `readthedocs <https://liana-py.readthedocs.io/en/latest/>`_.

Tutorials with Jupyter Notebooks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The documentation is set-up to render jupyter notebooks stored in the docs/notebooks.
Currently, only notebooks in .ipynb format are supported that will be included with both their input and output cells.
It is your responsibility to update and re-run the notebook whenever necessary.

Building the Docs Locally
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    cd docs
    make clean
    make html
    open _uild/html/index.html

Contributing to the Codebase
----------------------------

We welcome contributions to both the documentation and the codebase. If you have any questions, please don't hesitate to open an issue or reach out to us.

"""Fixtures shared across the test suite.

The test tree mirrors the layout of ``src/liana``, and the data every test
needs is built here rather than at module level, so that each test gets its
own, unmutated copy.
"""

import pathlib

import matplotlib
import pytest

# never try to open a window while testing
matplotlib.use("Agg")


@pytest.fixture(scope="session")
def data_dir():
    """Path to ``tests/data``, which holds the expected outputs of the methods."""
    return pathlib.Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def download_cache():
    """Directory the tests download their (large) external files to, once."""
    cache = pathlib.Path(__file__).parent / ".cache"
    cache.mkdir(exist_ok=True)

    return cache


@pytest.fixture(scope="session")
def metalinks_db(download_cache):
    """Path to MetaLinksDB, downloaded on first use."""
    import os

    from liana.resource.get_metalinks import _download_metalinksdb

    # NOTE: the db is always downloaded to the working directory
    cwd = os.getcwd()
    os.chdir(download_cache)
    try:
        return _download_metalinksdb(verbose=False)
    finally:
        os.chdir(cwd)


@pytest.fixture(scope="session")
def hcop_file(download_cache):
    """Path the human-mouse HCOP orthology table is cached at."""
    return str(download_cache / "human_mouse_hcop_fifteen_column.txt.gz")


@pytest.fixture
def pbmc68k():
    """Scanpy's reduced pbmc68k dataset."""
    from scanpy.datasets import pbmc68k_reduced

    return pbmc68k_reduced()


@pytest.fixture
def toy_adata():
    """`pbmc68k_reduced` with fake `sample` and `case` columns in `.obs`."""
    from liana.testing._sample_anndata import generate_toy_adata

    return generate_toy_adata()


@pytest.fixture
def toy_spatial():
    """`pbmc68k_reduced` with random coordinates and spatial connectivities."""
    from liana.testing._sample_anndata import generate_toy_spatial

    return generate_toy_spatial()


@pytest.fixture
def toy_mdata():
    """A two-modality MuData (`adata_x`, `adata_y`) built from `toy_spatial`."""
    from liana.testing._sample_anndata import generate_toy_mdata

    return generate_toy_mdata()


@pytest.fixture
def liana_res():
    """A toy ligand-receptor result table, as returned by the methods."""
    from liana.testing import sample_lrs

    return sample_lrs()


@pytest.fixture
def liana_res_by_sample():
    """A toy ligand-receptor result table with a `sample` column."""
    from liana.testing import sample_lrs

    return sample_lrs(by_sample=True)


@pytest.fixture(autouse=True)
def close_figures():
    """Close any figure a test left open, else matplotlib warns about too many."""
    yield

    from matplotlib import pyplot as plt

    plt.close("all")

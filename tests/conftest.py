"""Fixtures shared across the test suite.

The test tree mirrors the layout of ``src/liana``, and the data every test
needs is built here rather than at module level, so that each test gets its
own, unmutated copy.
"""

from __future__ import annotations

import pathlib
from collections.abc import Generator

import matplotlib
import pytest
from anndata import AnnData
from mudata import MuData
from pandas import DataFrame

# never try to open a window while testing
matplotlib.use("Agg")


@pytest.fixture(scope="session")
def data_dir() -> pathlib.Path:
    """Path to ``tests/data``, which holds the tests' inputs and expected outputs."""
    return pathlib.Path(__file__).parent / "data"


@pytest.fixture
def pbmc68k() -> AnnData:
    """Scanpy's reduced pbmc68k dataset."""
    from scanpy.datasets import pbmc68k_reduced

    return pbmc68k_reduced()


@pytest.fixture
def toy_adata() -> AnnData:
    """`pbmc68k_reduced` with fake `sample` and `case` columns in `.obs`."""
    from liana.datasets import generate_toy_adata

    return generate_toy_adata()


@pytest.fixture
def toy_spatial() -> AnnData:
    """`pbmc68k_reduced` with random coordinates and spatial connectivities."""
    from liana.datasets import generate_toy_spatial

    return generate_toy_spatial()


@pytest.fixture
def toy_mdata() -> MuData:
    """A two-modality MuData (`adata_x`, `adata_y`) built from `toy_spatial`."""
    from liana.datasets._sample_anndata import generate_toy_mdata

    return generate_toy_mdata()


@pytest.fixture
def liana_res() -> DataFrame:
    """A toy ligand-receptor result table, as returned by the methods."""
    from liana.datasets import sample_lrs

    return sample_lrs()


@pytest.fixture
def liana_res_by_sample() -> DataFrame:
    """A toy ligand-receptor result table with a `sample` column."""
    from liana.datasets import sample_lrs

    return sample_lrs(by_sample=True)


@pytest.fixture(autouse=True)
def close_figures() -> Generator[None]:
    """Close any figure a test left open, else matplotlib warns about too many."""
    yield

    from matplotlib import pyplot as plt

    plt.close("all")

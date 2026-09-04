"""Small helpers that keep the test suite type-checked.

`anndata` types `.X` as the full union of everything it can store (including `None` and backed arrays) and `.obs`/`.var` as `DataFrame | Dataset2D`, so the narrowing helpers from `liana._core._types` are re-exported here for the tests to use in place of the raw attributes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from anndata import AnnData
from fast_array_utils.conv import to_dense
from numpy.typing import NDArray
from pandas import DataFrame
from scipy.sparse import csr_matrix

from liana._core._types import (
    _to_matrix,
    get_coordinates,
    get_obs,
    get_raw_x,
    get_var,
    get_x,
)

__all__ = [
    "as_anndata",
    "as_frame",
    "get_obs",
    "get_obsp",
    "get_raw_x",
    "get_var",
    "get_coordinates",
    "get_csr",
    "get_layer",
    "get_layer_csr",
    "get_raw_csr",
    "get_x",
    "invalid",
    "not_none",
    "plot_data",
    "to_dense",
]


def not_none[T](value: T | None) -> T:
    """Narrow an optional result.

    Many of liana's entry points return `None` when `inplace=True` and a value otherwise; a test that passes `inplace=False` knows which it got.
    """
    assert value is not None
    return value


def as_anndata(value: object) -> AnnData:
    """Narrow a result the test knows is an :class:`~anndata.AnnData`.

    liana's entry points are typed for every shape they can return (`AnnData`, a frame, or `None` when `inplace=True`); a given call site knows which.
    """
    assert isinstance(value, AnnData)
    return value


def as_frame(value: object) -> DataFrame:
    """Narrow a result the test knows is a :class:`~pandas.DataFrame`."""
    assert isinstance(value, DataFrame)
    return value


def plot_data(plot: object) -> DataFrame:
    """The frame a plotnine plot was built from.

    `ggplot.data` is `Optional[DataLike]`, so it needs narrowing before a test can index it.
    """
    return as_frame(not_none(getattr(plot, "data", None)))


def invalid(value: object) -> Any:
    """Pass a deliberately invalid value to a typed parameter.

    Several public parameters are `Literal`s yet still validate at runtime, for callers that are not themselves type-checked.
    The tests that cover those guards route the bad value through here, so the intent is explicit and the escape is greppable rather than scattered as `# type: ignore`.
    """
    return value


def get_csr(adata: AnnData) -> csr_matrix:
    """``adata.X``, checked to be the CSR matrix liana's pipeline produces.

    Tests that read ``.data`` (the stored values) need the sparse form, not just "a matrix".
    """
    X = get_x(adata)
    assert isinstance(X, csr_matrix)
    return X


def get_raw_csr(adata: AnnData) -> csr_matrix:
    """``adata.raw.X``, checked to be a CSR matrix."""
    X = get_raw_x(adata)
    assert isinstance(X, csr_matrix)
    return X


def get_layer(adata: AnnData, key: str) -> NDArray[np.number]:
    """``adata.layers[key]`` as a dense array.

    `layers` values carry the same wide union as `.X`, so they need narrowing before numpy will accept them.
    """
    return to_dense(_to_matrix(adata.layers[key], what=f"adata.layers[{key!r}]"))


def get_layer_csr(adata: AnnData, key: str) -> csr_matrix:
    """``adata.layers[key]``, checked to be a CSR matrix.

    For the tests that compare ``.data``, i.e. the stored values rather than the dense contents.
    """
    layer = adata.layers[key]
    assert isinstance(layer, csr_matrix)
    return layer


def get_obsp(adata: AnnData, key: str) -> NDArray[np.number]:
    """``adata.obsp[key]`` as a dense array."""
    return to_dense(_to_matrix(adata.obsp[key], what=f"adata.obsp[{key!r}]"))

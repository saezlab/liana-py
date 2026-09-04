from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from liana._core._common import _get_liana_res
from liana._core._constants import DefaultValues as V
from liana._core._constants import Keys as K

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from anndata import AnnData


def _check_var(liana_res: pd.DataFrame, var_name: str, var: str | None) -> str:
    """Check that ``var`` names a column of ``liana_res`` and return it.

    Returning the name (rather than just validating) lets callers bind the result and carry a non-optional `str` from there on.
    """
    if var is None:
        raise ValueError(f"`{var_name}` must be provided!")
    if var not in liana_res.columns:
        raise ValueError(f"`{var}` ({var_name}) must be one of {liana_res.columns}")
    return var


def _prep_liana_res(
    adata: AnnData | None = None,
    liana_res: pd.DataFrame | None = None,
    source_labels: str | Sequence[str] | None = None,
    target_labels: str | Sequence[str] | None = None,
    ligand_complex: str | Sequence[str] | None = None,
    receptor_complex: str | Sequence[str] | None = None,
    uns_key: str = K.uns_key,
) -> pd.DataFrame:

    res = _get_liana_res(adata, liana_res, uns_key)

    # subset to only cell labels of interest
    res = _filter_labels(res, labels=source_labels, label_type="source")
    res = _filter_labels(res, labels=target_labels, label_type="target")

    res["interaction"] = res["ligand_complex"] + " -> " + res["receptor_complex"]

    if ligand_complex is not None:
        res = res[np.isin(res["ligand_complex"], ligand_complex)]
    if receptor_complex is not None:
        res = res[np.isin(res["receptor_complex"], receptor_complex)]

    return res


def _filter_labels(
    liana_res: pd.DataFrame,
    labels: str | Sequence[str] | None,
    label_type: str,
) -> pd.DataFrame:
    if labels is None:
        return liana_res

    wanted = [labels] if isinstance(labels, str) else list(labels)
    covered = np.isin(wanted, liana_res[label_type])
    if not covered.all():
        not_covered = np.array(wanted)[~covered]
        raise ValueError(f"{not_covered} not found in `liana_res['{label_type}']`!")
    return liana_res[np.isin(liana_res[label_type], wanted)]


def _aggregate_scores(
    res: pd.DataFrame,
    what: str,
    how: Literal["min", "max"],
    absolute: bool,
    entities: list[str],
) -> pd.DataFrame:
    res["score"] = np.absolute(res[what]) if absolute else res[what]
    return res.groupby(entities).agg(score=("score", how)).reset_index()


# single source of truth for the score inversion; see `process_scores(..., inverse_fn=)`
_invert_scores = V.inverse_fn


def _filter_by(
    liana_res: pd.DataFrame,
    filter_fn: Callable[[pd.Series], bool] | None,
) -> pd.DataFrame:
    if filter_fn is None:
        return liana_res

    msk = liana_res.apply(filter_fn, axis=1).astype(bool)
    relevant_interactions = np.unique(liana_res[msk].interaction)
    return liana_res[np.isin(liana_res["interaction"], relevant_interactions)]


def _get_top_n(
    liana_res: pd.DataFrame,
    top_n: int | None,
    orderby: str | None,
    orderby_ascending: bool | None,
    orderby_absolute: bool,
) -> pd.DataFrame:

    if top_n is not None:
        # get the top_n for each interaction
        if orderby is None:
            raise ValueError("Please specify the column to order the interactions.")
        if orderby_ascending is None:
            raise ValueError("Please specify if `orderby` is ascending or not.")
        how: Literal["min", "max"] = "min" if orderby_ascending else "max"

        top_lrs = _aggregate_scores(
            liana_res,
            what=orderby,
            how=how,
            absolute=orderby_absolute,
            entities=["interaction", "ligand_complex", "receptor_complex"],
        ).copy()
        top_interactions = top_lrs.sort_values("score", ascending=orderby_ascending).head(top_n).interaction

        # Filter liana_res to the interactions in top_lrs
        liana_res = liana_res[liana_res["interaction"].isin(top_interactions)]
        # set categories to the order of top_lrs
        liana_res["interaction"] = pd.Categorical(liana_res["interaction"], categories=top_interactions)

    return liana_res

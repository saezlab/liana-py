from __future__ import annotations

from collections.abc import Callable

import numpy as np
from anndata import AnnData
from pandas import DataFrame

from liana._constants import DefaultValues as V
from liana._constants import Keys as K
from liana._constants import PrimaryColumns as P
from liana._docs import d
from liana._common import _check_if_installed, _get_liana_res
from liana.method import process_scores


@d.dedent
def to_tensor_c2c(adata: AnnData = None,
                  sample_key: str = None,
                  score_key: str = None,
                  liana_res: DataFrame | None = None,
                  source_key: str = P.source,
                  target_key: str = P.target,
                  ligand_key: str = P.ligand_complex,
                  receptor_key: str = P.receptor_complex,
                  uns_key: str = K.uns_key,
                  non_expressed_fill: float | None = None,
                  inverse_fun: Callable = V.inverse_fun,
                  non_negative: bool = True,
                  return_dict: bool = False,
                  **kwargs
                  ):
    """
    Function to convert a LIANA result to a tensor for cell2cell analysis.

    Parameters
    ----------
    %(adata)s
    %(sample_key)s
    %(score_key)s
    liana_res
        A dataframe with the LIANA results. If None, it will be taken from `adata.uns[uns_key]`.
    %(source_key)s
    %(target_key)s
    %(ligand_key)s
    %(receptor_key)s
    %(uns_key)s
    non_expressed_fill
        Value to fill for non-expressed ligand-receptor pairs.
    %(inverse_fun)s
    non_negative
        Whether to make the tensor non-negative.
    return_dict
        Whether to return a dictionary of tensors.
    **kwargs
        keyword arguments to pass to Tensor-cell2cell's `cell2cell.tensor.external_scores.dataframes_to_tensor` function.

    Returns
    -------
    Returns a tensor of shape (n_samples, n_senders, n_receivers, n_interactions) or a dictionary of tensors if `return_dict` is True.

    Raises
    ------
    AssertionError
        If `uns_key` is not provided when given an `adata` instance.
    ValueError
        If neither `liana_res` or `adata` are provided, or if any of the provided keys (`sample_key`, `source_key`, `target_key`, `ligand_key` or `receptor_key`) are not found in `liana_res.uns[uns_key]` or if input data contains duplicates.

    Examples
    --------
    Expects a by-sample ligand-receptor result, as written by any method's
    `.by_sample`. A toy result stands in here:

    >>> import liana as li
    >>> adata = li.testing.generate_toy_adata()
    >>> adata.uns['liana_res'] = li.testing.sample_lrs(by_sample=True)
    >>> tensor = li.mu.to_tensor_c2c(adata,
    ...                              sample_key='sample',
    ...                              score_key='specificity_rank')

    The tensor can then be decomposed with Tensor-cell2cell.

    """
    # check if cell2cell is installed
    c2c = _check_if_installed("cell2cell")

    liana_res = _get_liana_res(adata, liana_res, uns_key)

    keys = np.array([sample_key, source_key, target_key, ligand_key, receptor_key])
    missing_keys = keys[[ key not in liana_res.columns for key in keys]]

    if any(missing_keys):
        raise ValueError(f'`{missing_keys}` not found in `adata.uns[{uns_key}]`! Please check your input.')

    # remove unneeded columns
    keys = [sample_key, source_key, target_key, ligand_key, receptor_key, score_key]
    keys = keys + ['lrs_to_keep'] if 'lrs_to_keep' in liana_res.columns else keys
    liana_res = liana_res[keys]

    # check for duplicates
    if liana_res[[sample_key, source_key, target_key, ligand_key, receptor_key]].duplicated().any():
        raise ValueError("Duplicate rows found in the input data")

    liana_res = process_scores(liana_res, score_key, inverse_fun)

    # set negative to 0
    if non_negative:
        liana_res[score_key] = liana_res[score_key].clip(lower=0)

    # set non-expressed to 0 (if lrs_to_keep column is present)
    if ('lrs_to_keep' in liana_res.columns) & (non_expressed_fill is not None):
        liana_res.loc[~liana_res['lrs_to_keep'], score_key] = non_expressed_fill

    # split into dictionary by sample
    liana_res = dict(tuple(liana_res.groupby(sample_key)))

    if return_dict:
        return liana_res

    tensor = c2c.tensor.dataframes_to_tensor(liana_res,
                                             sender_col=source_key,
                                             receiver_col=target_key,
                                             ligand_col=ligand_key,
                                             receptor_col=receptor_key,
                                             score_col=score_key,
                                             **kwargs)

    return tensor

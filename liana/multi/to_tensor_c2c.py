import numpy as np

from ._common import _process_scores
from liana._logging import _check_if_installed


def to_tensor_c2c(adata=None,
                  liana_res=None,
                  sample_key=None,
                  source_key='source',
                  target_key='target',
                  ligand_key='ligand_complex',
                  receptor_key='receptor_complex',
                  score_key=None,
                  uns_key = 'liana_res',
                  non_expressed_fill=None,
                  inverse_fun = lambda x: 1 - x,
                  non_negative = True,
                  return_dict=False,
                  **kwargs
                  ):
    """
    Function to convert a LIANA result to a tensor for cell2cell analysis.
    
    Parameters
    ----------
    
    adata : :class:`~anndata.AnnData`
        Annotated data matrix.
    liana_res : :class:`~pandas.DataFrame`
        LIANA result.
    sample_key : `str`, optional (default: None)
        Column name of the sample key in `liana_res`.
    source_key : `str`, optional (default: 'source')
        Column name of the sender/source cell types in `liana_res`.
    target_key : `str`, optional (default: 'target')
        Column name of the receiver/target cell types in `liana_res`.
    ligand_key : `str`, optional (default: 'ligand_complex')
        Column name of the ligand in `liana_res`.
    receptor_key : `str`, optional (default: 'receptor_complex')
        Column name of the receptor in `liana_res`.
    score_key : `str`, optional (default: None)
        Column name of the score in `liana_res`. If None, the score is inferred from the method.
    inverse_fun : `function`, optional (default: lambda x: 1 - x)
        Function to inverse the score. For example, if the score is in ascending order or probability,
        the inverse function should be 1 - probability. This is handled automatically for the scores in liana.
    non_expressed_fill : `float`, optional (default: None)
        Value to fill for non-expressed ligand-receptor pairs.
    non_negative : `bool`, optional (default: True)
        Whether to make the tensor non-negative.
    return_dict : `bool`, optional (default: False)
        Whether to return a dictionary of tensors.
    **kwargs : keyword arguments to pass to Tensor-cell2cell's `cell2cell.tensor.external_scores.dataframes_to_tensor` function.
        
    Returns
    -------
    Returns a tensor of shape (n_samples, n_senders, n_receivers, n_interactions) or a dictionary of tensors if `return_dict` is True.
        
    
    """
    
    # check if cell2cell is installed
    c2c = _check_if_installed("cell2cell")
    
    if (liana_res is None) & (adata is None):
        raise AttributeError('Ambiguous! One of `liana_res` or `adata` should be provided.')
    if adata is not None:
        assert uns_key in adata.uns_keys()
        liana_res = adata.uns[uns_key].copy()
    if liana_res is not None:
        liana_res = liana_res.copy()
    if (liana_res is None) & (adata is None):
        raise ValueError('`liana_res` or `adata` must be provided!')
    
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

    liana_res = _process_scores(liana_res, score_key, inverse_fun)

    # set negative to 0
    if non_negative:
        liana_res[score_key] = liana_res[score_key].clip(lower=0)

    # set non-expressed to 0 (if lrs_to_keep column is present)
    if ('lrs_to_keep' in liana_res.columns) & (non_expressed_fill is not None):
        liana_res.loc[~liana_res['lrs_to_keep'], score_key] = non_expressed_fill

    # split into dictionary by sample
    liana_res = {sample:df for sample, df in liana_res.groupby(sample_key)}
    
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

from ..method import get_method_scores

import numpy as np
from types import ModuleType


def _check_if_tensorc2c() -> ModuleType:

    try:
        import cell2cell as c2c

    except Exception:

        raise ImportError(
            'cell2cell is not installed. Please install it with: '
            'pip install cell2cell'
        )

    return c2c


def to_tensor_c2c(adata=None,
                  liana_res=None,
                  sample_key=None,
                  sender_col='source',
                  receiver_col='target',
                  ligand_col='ligand_complex',
                  receptor_col='receptor_complex',
                  score_col=None,
                  non_expressed_fill=None,
                  inverse_fun = lambda x: 1 - x,
                  non_negative = True,
                  return_dict=False,
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
    sender_col : `str`, optional (default: 'source')
        Column name of the sender/source cell types in `liana_res`.
    receiver_col : `str`, optional (default: 'target')
        Column name of the receiver/target cell types in `liana_res`.
    ligand_col : `str`, optional (default: 'ligand_complex')
        Column name of the ligand in `liana_res`.
    receptor_col : `str`, optional (default: 'receptor_complex')
        Column name of the receptor in `liana_res`.
    score_col : `str`, optional (default: None)
        Column name of the score in `liana_res`. If None, the score is inferred from the method.
    non_expressed_fill : `float`, optional (default: None)
        Value to fill for non-expressed ligand-receptor pairs.
    non_negative : `bool`, optional (default: True)
        Whether to make the tensor non-negative.
    return_dict : `bool`, optional (default: False)
        Whether to return a dictionary of tensors.
        
    Returns
    -------
    Returns a tensor of shape (n_samples, n_senders, n_receivers, n_interactions) or a dictionary of tensors if `return_dict` is True.
        
    
    """
    
    # check if cell2cell is installed
    c2c = _check_if_tensorc2c()
    
    if (liana_res is None) & (adata is None):
        raise AttributeError('Ambiguous! One of `liana_res` or `adata` should be provided.')
    if adata is not None:
        assert 'liana_res' in adata.uns_keys()
        liana_res = adata.uns['liana_res'].copy()
    if liana_res is not None:
        liana_res = liana_res.copy()
    if (liana_res is None) & (adata is None):
        raise ValueError('`liana_res` or `adata` must be provided!')
    
    if (sample_key is None) or (sample_key not in liana_res.columns):
        raise ValueError(f"Sample key `{sample_key}` not found in `liana_res`")
    
    if (score_col is None) or (score_col not in liana_res.columns):
        raise ValueError(f"Score column `{score_col}` not found in `liana_res`")
    
    # remove unneeded columns
    keys = [sample_key, sender_col, receiver_col, ligand_col, receptor_col, score_col]
    keys = keys + ['lrs_to_keep'] if 'lrs_to_keep' in liana_res.columns else keys
    liana_res = liana_res[keys]
    
    
    # check for duplicates
    if liana_res[[sample_key, sender_col, receiver_col, ligand_col, receptor_col]].duplicated().any():
        raise ValueError("Duplicate rows found in the input data")

    scores = get_method_scores()

    if not np.isin(score_col, list(scores.keys())).any():
        raise ValueError(f"Score column {score_col} not found method scores. ")

    # reverse if ascending order
    ascending_order = scores[score_col]
    if(ascending_order):
        liana_res[score_col] = inverse_fun(liana_res[score_col]) #

    # set negative to 0
    if non_negative:
        liana_res[score_col] = liana_res[score_col].clip(lower=0)

    # set non-expressed to 0 (if lrs_to_keep column is present)
    if ('lrs_to_keep' in liana_res.columns) & (non_expressed_fill is not None):
        liana_res.loc[~liana_res['lrs_to_keep'], score_col] = non_expressed_fill

    # split into dictionary by sample
    liana_res = {sample:df for sample, df in liana_res.groupby(sample_key)}
    
    if return_dict:
        return liana_res
    
    tensor = c2c.tensor.dataframes_to_tensor(liana_res,
                                             sender_col=sender_col,
                                             receiver_col=receiver_col,
                                             ligand_col=ligand_col,
                                             receptor_col=receptor_col,
                                             score_col=score_col)

    return tensor


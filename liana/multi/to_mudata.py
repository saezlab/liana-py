from __future__ import annotations

import numpy as np
import pandas as pd
import warnings as warnings

from anndata import AnnData
from mudata import MuData
from tqdm import tqdm

from ..method import process_scores
from liana._logging import _check_if_installed
from liana.method._pipe_utils import _check_groupby
from liana._docs import d
from liana._constants import DefaultValues as V, Keys as K, PrimaryColumns as P

@d.dedent
def adata_to_views(adata: AnnData,
                   groupby: str,
                   sample_key: str,
                   obs_keys: list = None,
                   view_sep: str = ':',
                   min_count: int = 10,
                   min_total_count: int = 15,
                   large_n: int = 10,
                   min_prop: float = 0.1,
                   keep_stats: bool = False,
                   verbose: bool = False,
                   **kwargs):
    """
    Converts an AnnData object to a MuData object with views that represent an aggregate for each entity in `adata.obs[groupby]`.

    Parameters
    ----------
    %(adata)s
    %(groupby)s
    %(sample_key)s
    obs_keys:
        Column names in `adata.obs` to merge with the MuData object
    view_sep:
        Separator to use when assigning `adata.var_names` to views
    min_count:
        Minimum number of counts per gene per sample to be included in the pseudobulk.
    min_total_count:
        Minimum number of counts per sample to be included in the pseudobulk.
    large_n:
        Number of samples per group that is considered to be "large".
    min_prop:
        Minimum proportion of samples that must have a count for a gene to be included in the pseudobulk.
    keep_stats:
        If True, keep the pseudobulk statistics in `mdata.uns['psbulk_stats']`. Default is False.
    %(verbose)s
    **kwargs
        Keyword arguments used to aggregate the values per cell into views. See `dc.filter_by_expr` for more details.

    Returns
    -------
    Returns a MuData object with views that represent an aggregate for each entity in `adata.obs[groupby]`.

    """

    # Check if MuData & decoupler are installed
    dc = _check_if_installed(package_name="decoupler")

    views = adata.obs[groupby].unique()
    views = tqdm(views, disable=not verbose)

    _check_groupby(adata=adata, groupby=groupby, verbose=verbose)

    padatas = {}
    if keep_stats:
        stats = []
    for view in (views):
        # filter AnnData to view
        temp = adata[adata.obs[groupby] == view].copy()
        # assign view to var_names
        temp.var_names = view + view_sep + temp.var_names

        padata = dc.get_pseudobulk(temp,
                                   sample_col=sample_key,
                                   groups_col=None,
                                   **kwargs
                                   )

        # only filter genes for views that pass QC
        if 0 in padata.shape:
            continue

        # edgeR filtering
        feature_mask = dc.filter_by_expr(padata,
                                         min_count=min_count,
                                         min_total_count=min_total_count,
                                         large_n=large_n,
                                         min_prop=min_prop,
                                         )
        padata = padata[:, feature_mask]

        # only append views that pass QC
        if 0 not in padata.shape:
            # keep psbulk stats
            if keep_stats:
                df = padata.obs.filter(items=['psbulk_n_cells', 'psbulk_counts'], axis=1)
                df.columns = [view + view_sep + col for col in df.columns]
                stats.append(df)

            del padata.obs
            padatas[view] = padata

    # Convert to MuData
    mdata = MuData(padatas)

    # process metadata
    _process_meta(adata=adata, mdata=mdata, sample_key=sample_key, obs_keys=obs_keys)

    # combine psbulk stats across views and add to mdata
    if keep_stats:
        mdata.uns['psbulk_stats'] = pd.concat(stats, axis=1)

    return mdata

@d.dedent
def lrs_to_views(adata: AnnData,
                 score_key: (str or None) = None,
                 inverse_fun: callable = V.inverse_fun,
                 obs_keys: (list or None) = None,
                 lr_prop: float = 0.5,
                 lr_fill: np.nan = np.nan,
                 lrs_per_view:int = 20,
                 lrs_per_sample:int = 10,
                 samples_per_view: int = 3,
                 min_variance:int = 0,
                 lr_sep: str = V.lr_sep,
                 cell_sep: str='&',
                 var_sep: str=':',
                 uns_key: str = K.uns_key,
                 sample_key: str = 'sample',
                 source_key: str = P.source,
                 target_key: str = P.target,
                 ligand_key: str = P.ligand_complex,
                 receptor_key: str = P.receptor_complex,
                 verbose: bool = V.verbose
                 ):
    """
    Converts a LIANA result to a MuData object with views that represent an aggregate for each entity in `adata.obs[groupby]`.

    Parameters
    ----------
    %(adata)s
    %(score_key)s
    %(inverse_fun)s
    obs_keys
        List of keys in `adata.obs` that should be included in the MuData object.
        These columns should correspond to the number of samples in `adata.obs[sample_key]`.
    lr_prop
        Reflects the minimum required proportion of samples for an interaction to be considered for building the views.
    lr_fill
        Value to fill in for interactions that are not present in a view. Default is `np.nan`.
    lrs_per_sample
        Reflects the minimum required number of interactions in a sample to be considered when building a specific view.
    lrs_per_view
        Reflects the minimum required number of interactions in a view to be considered for building the views.
    samples_per_view
        Reflects the minimum required samples to keep a view.
    min_variance
        Reflects the minimum required variance across samples for each interaction in each view.
        NaNs are ignored when computing the variance.
    %(lr_sep)s
    cell_sep
        Separator to use for the cell names in the views.
    var_sep
        Separator to use for the variable names in the views.
    %(uns_key)s
    %(sample_key)s
    %(source_key)s
    %(target_key)s
    %(ligand_key)s
    %(receptor_key)s
    %(verbose)s

    Returns
    -------
    Returns a MuData object with views that represent an aggregate for each entity in `adata.obs[groupby]`.

    """
    if (sample_key not in adata.obs.columns) or (sample_key not in adata.uns[uns_key].columns):
        raise ValueError(f'`{sample_key}` not found in `adata.obs` or `adata.uns[uns_key]`!' +
                         'Please ensure that the sample key is present in both objects.')

    if uns_key not in adata.uns_keys():
        raise ValueError(f'`{uns_key}` not found in `adata.uns`! Please run `li.mt.rank_aggregate.by_sample` first.')

    liana_res = adata.uns[uns_key].copy()

    if (score_key is None) or (score_key not in liana_res.columns):
        raise ValueError(f"Score column `{score_key}` not found in `liana_res`")

    if isinstance(obs_keys, list):
        if any([key not in adata.obs.keys() for key in obs_keys]):
            raise ValueError(f'`{obs_keys}` not found in `adata.obs`!')
    elif obs_keys is not None:
        raise ValueError('`obs_keys` must be a list or `None`!')

    keys = np.array([sample_key, source_key, target_key, ligand_key, receptor_key])
    missing_keys = keys[[ key not in liana_res.columns for key in keys]]

    if any(missing_keys):
        raise ValueError(f'`{missing_keys}` not found in `adata.uns[{uns_key}]`! Please check your input.')

    # concat columns (needed for MOFA)
    liana_res['interaction'] = liana_res[ligand_key] + lr_sep + liana_res[receptor_key]
    liana_res['ct_pair'] = liana_res[source_key] + cell_sep + liana_res[target_key]
    liana_res = liana_res[[sample_key, 'ct_pair', 'interaction', score_key]]

    # get scores & invert if necessary
    liana_res = process_scores(liana_res=liana_res,
                                score_key=score_key,
                                inverse_fun=inverse_fun)

    # count samples per interaction
    count_pairs = (liana_res.
                   drop(columns=score_key).
                   groupby(['interaction', 'ct_pair']).
                   count().
                   rename(columns={sample_key: 'count'}).
                   reset_index()
                   )

    sample_n = liana_res[sample_key].nunique()

    # Keep only lrs above a certain proportion of samples
    count_pairs = count_pairs[count_pairs['count'] >= sample_n * lr_prop]
    liana_res = liana_res.merge(count_pairs.drop(columns='count') , how='inner')

    # Keep only samples above a certain number of LRs
    count_lrs = (liana_res.
                 drop(columns=score_key).
                 groupby([sample_key, 'ct_pair']).
                 count().
                 rename(columns={'interaction': 'count'}).
                 reset_index()
                 )
    count_lrs = count_lrs[count_lrs['count'] >= lrs_per_sample]
    liana_res = liana_res.merge(count_lrs.drop(columns='count') , how='inner')

    # convert to anndata views
    views = liana_res['ct_pair'].unique()
    views = tqdm(views, disable=not verbose)

    lr_adatas = {}
    for view in views:
        lrs_per_ct = liana_res[liana_res['ct_pair']==view]
        lrs_wide = lrs_per_ct.pivot(index='interaction',
                                    columns=sample_key,
                                    values=score_key)

        lrs_wide.index = view + var_sep + lrs_wide.index
        lrs_wide = lrs_wide.replace(np.nan, lr_fill)

        if lrs_wide.shape[0] >= lrs_per_view: # check if enough LRs
            temp = _dataframe_to_anndata(lrs_wide)

            # keep only variables with variance > min_variance
            temp = temp[:, np.nanvar(temp.X, axis=0) > min_variance]

            if (temp.shape[0] >= samples_per_view): # check if enough samples
                lr_adatas[view] = temp

    # to mdata
    mdata = MuData(lr_adatas)

    # process metadata
    _process_meta(adata=adata, mdata=mdata, sample_key=sample_key, obs_keys=obs_keys)

    return mdata


def _dataframe_to_anndata(df):
    obs = pd.DataFrame(index=df.columns)
    var = pd.DataFrame(index=df.index)
    X = np.array(df.values).T

    return AnnData(X=X, obs=obs, var=var, dtype=np.float32)


def _remove_mod_var(mdata, markers, view_sep, var_column):
    for current_mod in mdata.mod.keys():
        # markers in markers dict for each modality except for current_mod
        negative_markers = [marker for mod in markers.keys() if mod != current_mod for marker in markers[mod]]

        if current_mod not in list(markers.keys()):
            warnings.warn('no markers in dict for view: {0}'.format(current_mod), Warning)
        else:
            #keep negative_markers not in markers[current_mod] and add view_sep
            negative_markers = [current_mod + view_sep + marker for marker in negative_markers if marker not in markers[current_mod]]

        if var_column is None:
            # remove negative_markers from current_mod
            mdata.mod[current_mod] = mdata.mod[current_mod][:, ~mdata.mod[current_mod].var_names.isin(negative_markers)]
        else:
            # set negative_markers to False in current_mod
            mdata.mod[current_mod].var.loc[mdata.mod[current_mod].var_names.isin(negative_markers), var_column] = False

    mdata.update()

@d.dedent
def filter_view_markers(mdata: MuData,
                        markers: dict,
                        view_sep: str = ':',
                        var_column: str = 'highly_variable',
                        inplace: bool =False
                        ):
    """
    Used for removing potential cell type marker genes found in the background of other views and thought to be contamination.
    In each view, sets highly variable genes to False if they are in the markers dict for another view, but not if they are in the markers for the same view.


    Parameters
    ----------
    %(mdata)s
    markers :class:`dict`
        Dictionary with markers for each view. Keys are the views and values are lists of markers. Can contain markers for views that are not in mdata.mod.keys().
    view_sep :class:`str`, optional
        Separator between view and gene names. Defaults to ':'.
    var_column :class:`str`, optional
        Column in mdata.mod['some_view'].var that contains the highly variable genes. Defaults to 'highly_variable'.
        If set to ``None``, instead of setting the hvg genes to False, the hvg genes will be removed from the view.
    %(inplace)s
    """
    # check if markers is a dict
    if not isinstance(markers, dict):
        raise TypeError('markers is not a dict')

    # check that all keys in markers are lists
    if not all(isinstance(markers[mod], list) for mod in markers.keys()):
        raise TypeError('not all values in markers are lists')

    # check that var_column is in var for all modalities
    if var_column is not None:
        if not all(var_column in mdata.mod[mod].var.columns for mod in mdata.mod.keys()):
            raise ValueError('{0} is not in the columns of .var for all modalities'.format(var_column))

    if inplace:
        _remove_mod_var(mdata, markers, view_sep, var_column)
    else:
        cdata = mdata.copy()
        _remove_mod_var(cdata, markers, view_sep, var_column)
        return cdata


def _process_meta(adata, mdata, sample_key, obs_keys):
    if obs_keys is not None:
        metadata = adata.obs[[sample_key, *obs_keys]].drop_duplicates()

        sample_n = adata.obs[sample_key].nunique()
        if metadata.shape[0] != sample_n:
            raise ValueError('`obs_keys` must be unique per sample in `adata.obs`')

        mdata.obs.index.name = None
        mdata.obs = (mdata.obs.
                     reset_index().
                     rename(columns={"index":sample_key}).
                     merge(metadata).
                     set_index(sample_key)
                     )

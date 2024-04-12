from __future__ import annotations

from sklearn.decomposition import NMF
import numpy as np
import pandas as pd
import plotnine as p9
from tqdm import tqdm

from anndata import AnnData
from liana.method._pipe_utils._pre import _choose_mtx_rep
from liana._logging import _logg, _check_if_installed
from liana._docs import d

@d.dedent
def nmf(adata: AnnData=None,
        df: pd.DataFrame=None,
        n_components: (int or None)=None,
        k_range: range = range(1, 11),
        use_raw: bool=False,
        layer: (str or None)=None,
        inplace: bool=True,
        verbose: bool=False,
        **kwargs):
    """
    Fits NMF to an AnnData object.

    Parameters
    ----------
    %(adata)s
    n_components : int, None
        Number of components to use. If None, the number of components is estimated using the elbow method.
    k_range : range
        Range of components to test. Default: range(1, 10).
    %(use_raw)s
    %(layer)s
    %(inplace)s
    **kwargs : dict
        Keyword arguments to pass to ``sklearn.decomposition.NMF``.

    Returns
    -------
    If inplace is True, it will add ``NMF_W`` and ``NMF_H`` to the ``adata.obsm`` and ``adata.varm``.
    If n_components is None, it will also add ``nfm_errors`` and ``nfm_rank`` to ``adata.uns``.

    If inplace is False, it will return ``W`` and ``H``, and if n_components is None, it will also return ``errors`` and ``n_components``.
    If n_components is None and inplace, ``errors`` and ``n_components`` will be assigned to ``adata.uns``.
    If ``df`` is provided, inplace is always False.

    """
    if adata is not None:
        if isinstance(adata, AnnData):
            X = _choose_mtx_rep(adata, layer=layer, use_raw=use_raw)
        else :
            raise ValueError('Provide an AnnData object.')
    elif df is not None:
        X = df.values
    else:
        raise ValueError('Provide either an AnnData object or a DataFrame.')


    if n_components is None:
        errors, n_components = estimate_elbow(X, k_range=k_range, verbose=verbose, **kwargs)
        _plot_elbow(errors, n_components)
    else:
        errors, n_components = None, n_components

    nmf = NMF(n_components=n_components, **kwargs)
    W = nmf.fit_transform(X)
    H = nmf.components_.T

    inplace = inplace and (adata is not None)
    if inplace:
        adata.obsm['NMF_W'] = W
        adata.varm['NMF_H'] = H
        adata.uns['nmf_errors'] = errors
        adata.uns['nmf_rank'] = n_components

    return None if inplace else (W, H, errors, n_components)


def estimate_elbow(X, k_range, verbose=False, **kwargs):
    kn = _check_if_installed('kneed')
    errors = []
    for k in tqdm(k_range, disable=not verbose):
        error = _calculate_error(X, k, **kwargs)
        errors.append(error)

    kneedle = kn.KneeLocator(x=k_range,
                             y=errors,
                             direction='decreasing',
                             curve='convex',
                             interp_method='interp1d',
                             S=1
                             )
    rank = kneedle.knee

    _logg(f'Estimated rank: {rank}', verbose=verbose)

    errors = pd.DataFrame(errors,
                          index=list(k_range),
                          columns=['error']). \
                              reset_index().rename(columns={'index': 'k'})

    return errors, rank


def _calculate_error(X, n_components, **kwargs):
    nmf = NMF(n_components=n_components, **kwargs)
    W = nmf.fit_transform(X)
    H = nmf.components_

    Xhat = np.dot(W, H)
    error = np.mean(np.abs(X - Xhat))

    return error


def _plot_elbow(errors, n_components, x='k', y='error'):
    p = (
        p9.ggplot(errors, p9.aes(x=x, y=y)) +
        p9.geom_line() +
        p9.geom_point() +
        p9.theme_bw() +
        p9.scale_x_continuous(breaks=errors[x].values) +
        p9.labs(x='Component number (k)', y='Reconstruction error') +
        p9.geom_vline(xintercept=n_components, linetype='dashed', color='red')
    )
    p.draw()

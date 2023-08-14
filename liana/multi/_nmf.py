from sklearn.decomposition import NMF
import numpy as np
import pandas as pd
import plotnine as p9
from tqdm import tqdm

from liana.method._pipe_utils._pre import _choose_mtx_rep

def nmf(adata, n_components=None, k_range=range(1, 10), use_raw=False, layer=None, inplace=True, verbose=False, **kwargs):
    """
    Fits NMF to an AnnData object.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object.
    n_components : int, None
        Number of components to use. If None, the number of components is estimated using the elbow method.
    k_range : range
        Range of components to test. Default: range(1, 10).
    use_raw : bool
        Whether to use ``.raw`` attribute of ``adata``.
    layer : str, None
        ``.layers`` key to use. If None, ``.X`` is used.
    inplace : bool
        If ``False``, return a copy. Otherwise, do operation inplace and return ``None``.
    **kwargs : dict
        Keyword arguments to pass to ``sklearn.decomposition.NMF``.
    
    Returns
    -------
    If inplace is True, it will add ``NMF_W`` and ``NMF_H`` to the ``adata.obsm`` and ``adata.varm`` AnnData objects.
    
    """
    
    if n_components is None:
        errors, n_components = estimate_elbow(adata.X, k_range=k_range, verbose=verbose, **kwargs)
        _plot_elbow(errors, n_components)
    
    nmf = NMF(n_components=n_components, **kwargs)
    X = _choose_mtx_rep(adata, layer=layer, use_raw=use_raw)
    W = nmf.fit_transform(X)
    H = nmf.components_.T
    
    if inplace:
        adata.obsm['NMF_W'] = W
        adata.varm['NMF_H'] = H
    return None if inplace else (W, H)


def check_if_kneed():
    try:
        import kneed as kn
    except Exception:
        raise ImportError('kneed is not installed. Please install it with: pip install kneed')
    return kn


def estimate_elbow(X, k_range, verbose=False, **kwargs):
    kn = check_if_kneed()
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
    
    if verbose:
        print(f'Estimated rank: {rank}')

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

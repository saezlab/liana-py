from sklearn.decomposition import NMF
import numpy as np
import pandas as pd
import plotnine as p9
from tqdm import tqdm


def nmf(adata, n_components=None, k_range=range(1, 10), inplace=True, verbose=False, **kwargs):
    
    if n_components is None:
        errors, n_components = estimate_elbow(adata.X, k_range=k_range, verbose=verbose, **kwargs)
        _plot_elbow(errors, n_components)
    
    nmf = NMF(n_components=n_components, **kwargs)
    W = nmf.fit_transform(adata.X)
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
    error = np.mean(np.sqrt((X - Xhat) ** 2))
    
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

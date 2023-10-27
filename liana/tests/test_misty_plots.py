import numpy as np

from liana.plotting import interactions, contributions, target_metrics
from liana.testing import generate_toy_spatial
from liana.testing import _sample_target_metrics, _sample_interactions

adata = generate_toy_spatial()
adata.uns['target_metrics'] = _sample_target_metrics()
adata.uns['interactions'] = _sample_interactions()
adata.view_names = ['intra', 'extra']


def test_target_contributions_plot():    
    contributions(misty=adata)    
    contributions(misty=adata, top_n=1, return_fig=False)


def test_target_metrics_plot():
    target_metrics(misty=adata, stat='gain_R2')
    target_metrics(misty=adata, stat='gain_R2', top_n=1, return_fig=False)
    target_metrics(misty=adata, stat='gain_R2', filterby='multi_R2', filter_lambda=lambda x: x > 0.5)


def test_misty_interactions_plot():
    interactions(misty=adata, top_n=3, view='extra', key=abs, ascending=False)

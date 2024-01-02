import liana.plotting as pl
from liana.testing import generate_toy_spatial
from liana.testing import _sample_target_metrics, _sample_interactions

adata = generate_toy_spatial()
adata.uns['target_metrics'] = _sample_target_metrics()
adata.uns['interactions'] = _sample_interactions()
adata.view_names = ['intra', 'extra']

# test aggregate
target_metrics = adata.uns['target_metrics'].copy()
target_metrics = target_metrics._append(target_metrics)
target_metrics['group'] = ['a'] * 3 + ['b'] * 3

interactions = adata.uns['interactions'].copy()
interactions = interactions._append(interactions)
interactions['group'] = ['a'] * 9 + ['b'] * 9

def test_target_contributions_plot():
    pl.contributions(misty=adata)
    pl.contributions(misty=adata, return_fig=False)


def test_target_metrics_plot():
    pl.target_metrics(misty=adata, stat='gain_R2')
    pl.target_metrics(misty=adata, stat='gain_R2', top_n=1, return_fig=False)
    pl.target_metrics(misty=adata, stat='gain_R2', filter_fun = lambda x: x['multi_R2'] > 0.5)


def test_interactions_plot():
    pl.interactions(misty=adata, top_n=3, view='extra', key=abs, ascending=False)
    plot_data = pl.interactions(interactions=interactions, view='extra', filter_fun=lambda x: x['group']=='b').data
    assert plot_data.shape[0] == 3


def test_target_metrics_aggregate():
    pl.target_metrics(target_metrics=target_metrics, stat='gain_R2', aggregate_fun='mean')

def test_contributions_aggregate():
    pl.contributions(target_metrics=target_metrics, view_names=['intra', 'extra'], aggregate_fun='median')

def test_interactions_aggregate():
    pl.interactions(interactions=interactions, view='intra', aggregate_fun='sum')

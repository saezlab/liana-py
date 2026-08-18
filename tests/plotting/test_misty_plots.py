import pandas as pd
import pytest

import liana.plotting as pl
from liana.testing import _sample_interactions, _sample_target_metrics


@pytest.fixture
def misty(toy_spatial):
    """A spatial AnnData carrying toy misty results."""
    toy_spatial.uns['target_metrics'] = _sample_target_metrics()
    toy_spatial.uns['interactions'] = _sample_interactions()
    toy_spatial.view_names = ['intra', 'extra']

    return toy_spatial


@pytest.fixture
def target_metrics():
    """Target metrics for two groups, e.g. two samples to aggregate over."""
    target_metrics = pd.concat([_sample_target_metrics()] * 2)
    target_metrics['group'] = ['a'] * 3 + ['b'] * 3

    return target_metrics


@pytest.fixture
def interactions():
    """Interactions for two groups, e.g. two samples to aggregate over."""
    interactions = pd.concat([_sample_interactions()] * 2)
    interactions['group'] = ['a'] * 9 + ['b'] * 9

    return interactions


def test_target_contributions_plot(misty):
    plot_data = pl.contributions(misty=misty).data

    # melted to one row per target x view, with no target or view dropped
    targets = misty.uns['target_metrics']['target']
    assert set(plot_data['target']) == set(targets)
    assert set(plot_data['view']) == {'intra', 'extra'}
    assert plot_data.shape[0] == len(targets) * 2

    # `return_fig=False` draws the plot instead of handing it back
    assert pl.contributions(misty=misty, return_fig=False) is None


def test_target_metrics_plot(misty):
    target_metrics = misty.uns['target_metrics']

    plot_data = pl.target_metrics(misty=misty, stat='gain_R2').data
    assert set(plot_data['target']) == set(target_metrics['target'])

    # top_n keeps the n best targets by the statistic asked for
    top = pl.target_metrics(misty=misty, stat='gain_R2', top_n=1).data
    best = target_metrics.loc[target_metrics['gain_R2'].idxmax(), 'target']
    assert set(top['target']) == {best}

    filtered = pl.target_metrics(misty=misty, stat='gain_R2',
                                 filter_fun=lambda x: x['multi_R2'] > 0.5).data
    expected = target_metrics[target_metrics['multi_R2'] > 0.5]['target']
    assert set(filtered['target']) == set(expected)

    assert pl.target_metrics(misty=misty, stat='gain_R2', return_fig=False) is None


def test_interactions_plot(misty, interactions):
    pl.interactions(misty=misty, top_n=3, view='extra', key=abs, ascending=False)
    plot_data = pl.interactions(interactions=interactions, view='extra',
                                filter_fun=lambda x: x['group']=='b').data
    assert plot_data.shape[0] == 3

    assert pl.interactions(misty=misty, view='extra', return_fig=False) is None


def test_target_metrics_aggregate(target_metrics):
    plot_data = pl.target_metrics(target_metrics=target_metrics, stat='gain_R2',
                                  aggregate_fun='mean').data

    # every group is kept - aggregation only decides the order the targets
    # are drawn in, best mean first
    assert plot_data.shape[0] == target_metrics.shape[0]
    expected = (target_metrics.groupby('target')['gain_R2'].mean()
                .sort_values(ascending=False).index.tolist())
    assert list(plot_data['target'].cat.categories) == expected


def test_contributions_aggregate(target_metrics):
    plot_data = pl.contributions(target_metrics=target_metrics,
                                 view_names=['intra', 'extra'],
                                 aggregate_fun='median').data

    n_targets = target_metrics['target'].nunique()
    assert plot_data.shape[0] == n_targets * 2
    expected = target_metrics.groupby('target')['intra'].median()
    actual = plot_data[plot_data['view'] == 'intra'].set_index('target')['contribution']
    pd.testing.assert_series_equal(actual.sort_index(), expected.sort_index(),
                                   check_names=False, check_dtype=False)


def test_interactions_aggregate(interactions):
    plot_data = pl.interactions(interactions=interactions, view='intra',
                                aggregate_fun='sum').data

    # only the requested view is drawn, with importances summed over the groups
    intra = interactions[interactions['view'] == 'intra']
    assert set(plot_data['target']) == set(intra['target'])
    expected = intra.groupby(['target', 'predictor'])['importances'].sum()
    actual = plot_data.set_index(['target', 'predictor'])['importances']
    pd.testing.assert_series_equal(actual.sort_index(), expected.sort_index(),
                                   check_names=False, check_dtype=False)


def test_misty_plots_raise_without_data():
    with pytest.raises(ValueError, match='Provide either a misty object or a target_metrics'):
        pl.target_metrics(stat='gain_R2')

    with pytest.raises(ValueError, match='Provide either a misty object or a target_metrics'):
        pl.contributions(view_names=['intra', 'extra'])

    with pytest.raises(ValueError, match='Provide either a misty object or interactions'):
        pl.interactions(view='intra')


def test_misty_plots_raise_on_missing_args(misty, target_metrics, interactions):
    with pytest.raises(ValueError, match='Provide a statistic to plot'):
        pl.target_metrics(misty=misty, stat=None)

    with pytest.raises(ValueError, match='Provide a list of view names to plot'):
        pl.contributions(target_metrics=target_metrics)

    with pytest.raises(ValueError, match='Provide a ``view`` to plot'):
        pl.interactions(interactions=interactions, view=None)


def test_contributions_filter(misty):
    plot_data = pl.contributions(misty=misty,
                                 filter_fun=lambda x: x['multi_R2'] > 0.5).data

    target_metrics = misty.uns['target_metrics']
    expected = target_metrics[target_metrics['multi_R2'] > 0.5]['target']
    assert set(plot_data['target']) == set(expected)


def test_contributions_drops_intra_when_absent(misty):
    # `intra` is dropped from the views when it is not among the metrics
    del misty.uns['target_metrics']['intra']
    plot_data = pl.contributions(misty=misty).data
    assert set(plot_data['view']) == {'extra'}


def test_contributions_filter_on_categorical_target(target_metrics):
    target_metrics['target'] = target_metrics['target'].astype('category')
    plot_data = pl.contributions(target_metrics=target_metrics,
                                 view_names=['intra', 'extra'],
                                 aggregate_fun='mean',
                                 filter_fun=lambda x: x['target'] != 'a').data
    assert 'a' not in set(plot_data['target'].cat.categories)

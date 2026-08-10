import numpy as np

from liana.plotting import dotplot, dotplot_by_sample, tileplot
from liana.testing import generate_toy_spatial, sample_lrs

liana_res = sample_lrs()

def test_dotplot_order():
    my_p = dotplot(liana_res=liana_res,
                   size='specificity_rank',
                   colour='magnitude',
                   top_n=20,
                   orderby='specificity_rank',
                   orderby_ascending=False,
                   target_labels=["A", "B", "C"]
                   )
    assert my_p is not None
    assert 'interaction' in my_p.data.columns
    np.testing.assert_equal(np.unique(my_p.data.interaction).shape, (20,))
    set(my_p.data.target)
    assert {'A', 'B', 'C'} == set(my_p.data.target)


def test_doplot_filter():
    my_p2 = dotplot(liana_res=liana_res,
                    size='specificity_rank',
                    colour='magnitude',
                    filter_fun=lambda x: x['specificity_rank'] > 0.95,
                    inverse_colour=True,
                    source_labels=["A"]
                    )
    assert my_p2 is not None
    # we force this, but not intended all interactions
    # to be only 0.95, but rather for an interaction to get
    # plotted, in at least one cell type pair it should be > 0.95
    assert all(my_p2.data['specificity_rank'] > 0.95) is True


def test_dotplot_bysample():
    liana_res = sample_lrs(by_sample=True)
    my_p3 = dotplot_by_sample(liana_res=liana_res,
                              size='specificity_rank',
                              colour='magnitude',
                              target_labels='E',
                              sample_key='sample')
    assert my_p3 is not None
    assert 'interaction' in my_p3.data.columns
    assert 'sample' in my_p3.data.columns
    assert 'B' not in my_p3.data['target']


def test_tileplot():
    my_p2 = tileplot(liana_res = liana_res,
                     # NOTE: fill & label need to exist for both
                     # ligand_ and receptor_ columns
                     fill='means',
                     label='pvals',
                     label_fun=lambda x: f'{x:.2f}',
                     top_n=10,
                     orderby='specificity_rank',
                     orderby_ascending=True
                     )
    assert my_p2 is not None
    assert isinstance(my_p2.data['pvals'].values[0], str)


def test_proximity_plot():
    from liana.plotting import connectivity

    adata = generate_toy_spatial()
    my_p4 = connectivity(adata=adata, idx=0)
    assert my_p4 is not None


def test_circle_plot():
    from scanpy.datasets import pbmc68k_reduced

    from liana.plotting import circle_plot
    adata = pbmc68k_reduced()
    unique_sources = np.unique(liana_res['source'])
    adata.obs['random'] = np.random.choice(unique_sources, size=adata.shape[0], replace=True)
    adata.uns['liana_res'] = liana_res

    circle_plot(adata, groupby='random', liana_res=liana_res,
                pivot_mode='mean', score_key='specificity_rank')
    circle_plot(adata, groupby='random', liana_res=liana_res, pivot_mode='counts',
                filter_fun=lambda x: x['specificity_rank'] < 0.95)

def test_annulus_plot():
    import pytest
    from liana.plotting import annulus_plot
    from liana.testing._sample_anndata import generate_toy_spatial

    adata = generate_toy_spatial()
    annulus_plot(
        adata,
        spatial_key="spatial",
        annulus_width=200,
        radius_step=200,
        n_rings=5,
        seed=42,
    )

    with pytest.raises(KeyError, match="not found in adata.obsm"):
        annulus_plot(adata, spatial_key="missing_key")


def test_lric_lineplot():
    import matplotlib
    import pytest

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    from liana.plotting import lric_lineplot

    radii = np.linspace(0, 500, 20)
    curves = {f"pair_{i}": np.random.rand(20) + 0.5 for i in range(5)}

    # small-multiples: returns figure, unused axes hidden
    fig = lric_lineplot(radii, curves, return_fig=True)
    assert isinstance(fig, Figure)
    axes = fig.axes
    visible = [ax for ax in axes if ax.get_visible()]
    assert len(visible) == 5

    # overlay mode
    fig2 = lric_lineplot(radii, curves, overlay=True, title="test", return_fig=True)
    assert isinstance(fig2, Figure)
    assert len(fig2.axes) == 1

    # per-curve radii as (r, g) tuples
    mixed = {"a": (radii * 0.5, np.ones(20)), "b": np.ones(20) * 1.2}
    fig3 = lric_lineplot(radii, mixed, return_fig=True)
    assert isinstance(fig3, Figure)

    # color variants
    lric_lineplot(radii, curves, colors="red", return_fig=True)
    lric_lineplot(radii, curves, colors=["red", "blue"], return_fig=True)
    lric_lineplot(radii, curves, colors={k: "green" for k in curves}, return_fig=True)

    # radii=None
    all_tuples = {"a": (radii * 0.5, np.ones(20)), "b": (radii * 0.8, np.ones(20) * 1.5)}
    fig_none = lric_lineplot(None, all_tuples, overlay=True, return_fig=True)
    assert isinstance(fig_none, Figure)

    # empty curves raises
    with pytest.raises(ValueError, match="at least one entry"):
        lric_lineplot(radii, {}, return_fig=True)


def test_feature_by_group():
    from liana.plotting import feature_by_group
    from liana.testing._sample_anndata import generate_toy_spatial
    adata = generate_toy_spatial()
    feature_by_group(
        adata=adata,
        groupby='bulk_labels',
        labels=['Dendritic', 'CD56+ NK'],
        feature='HES4',
        normalize=True,
        percentile_scaling=(5, 95),
        show_counts=True
    )

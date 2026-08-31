import plotnine as p9
import pytest

from liana.method.sp._LRIC import cross_pcf, lric
from liana.plotting import lric_divergence_plot, lric_lineplot
from liana.datasets import generate_toy_spatial
from liana.datasets._sample_resource import sample_resource

_KWARGS = {"max_radius": 100, "radius_step": 20, "verbose": False}


@pytest.fixture(scope="module")
def adata():
    ad = generate_toy_spatial()
    ad.obs["cell_type"] = ad.obs["bulk_labels"]
    res = sample_resource(ad, n_lrs=5, seed=42)
    cross_pcf(ad, groupby="cell_type", key_added="cross_pcf", **_KWARGS)
    lric(ad, resource=res, key_added="lric_ag", **_KWARGS)
    lric(ad, resource=res, groupby="cell_type", key_added="lric_ct", **_KWARGS)
    return ad


def test_cross_pcf_lineplot(adata):
    row = adata.uns["cross_pcf"].iloc[0]
    p = lric_lineplot(adata, "cross_pcf", interaction=row["interaction"], max_dist=60, return_fig=True)
    assert isinstance(p, p9.ggplot)
    # source/target select the same (symmetric) curve
    lric_lineplot(adata, "cross_pcf", source=row["source"], target=row["target"], return_fig=True)


def test_lric_agnostic_lineplot(adata):
    interaction = adata.uns["lric_ag"]["interaction"].iloc[0]
    p = lric_lineplot(adata, "lric_ag", interaction=interaction, return_fig=True)
    assert isinstance(p, p9.ggplot)
    # `liana_res=` is accepted in place of an AnnData
    lric_lineplot(liana_res=adata.uns["lric_ag"], interaction=interaction, return_fig=True)


def test_lric_pairwise_lineplot_decomposes(adata):
    row = adata.uns["lric_ct"].iloc[0]
    p = lric_lineplot(
        adata, "lric_ct", interaction=row["interaction"],
        source=row["source"], target=row["target"], return_fig=True,
    )
    assert isinstance(p, p9.ggplot)
    # full + architecture-only + expression-only
    assert set(p.data["curve"].unique()) == {"g (full)", "g_pcf", "g_expr"}


def test_max_dist_restricts_the_plotted_radii(adata):
    # `max_dist` means the same thing here as in `get_lric_auc` -- it is a window,
    # not just a marker
    row = adata.uns["cross_pcf"].iloc[0]
    sel = {"source": row["source"], "target": row["target"]}
    full = lric_lineplot(adata, "cross_pcf", **sel, return_fig=True)
    windowed = lric_lineplot(adata, "cross_pcf", **sel, max_dist=45, return_fig=True)
    assert (windowed.data["radius"] < 45).all()
    assert len(windowed.data) < len(full.data)

    with pytest.raises(ValueError, match="No radii below"):
        lric_lineplot(adata, "cross_pcf", **sel, max_dist=-1, return_fig=True)


def test_divergence_plot(adata):
    two = adata.uns["cross_pcf"]["interaction"].unique()[:2]
    p = lric_divergence_plot(
        adata, "cross_pcf",
        feature_a={"interaction": two[0]}, feature_b={"interaction": two[1]},
        min_bins=2, return_fig=True,
    )
    assert isinstance(p, p9.ggplot)
    # both curves are drawn
    assert p.data["curve"].nunique() == 2
    # `liana_res=` is accepted in place of an AnnData
    lric_divergence_plot(
        liana_res=adata.uns["cross_pcf"],
        feature_a={"interaction": two[0]}, feature_b={"interaction": two[1]},
        min_bins=2, return_fig=True,
    )


def test_divergence_plot_bad_selection_raises(adata):
    with pytest.raises(ValueError, match="No rows match"):
        lric_divergence_plot(
            adata, "cross_pcf",
            feature_a={"interaction": "nope^nope"}, feature_b={"interaction": "nope^nope"},
            return_fig=True,
        )


def test_bad_selection_raises(adata):
    with pytest.raises(ValueError, match="not found"):
        lric_lineplot(adata, "cross_pcf", interaction="nope", return_fig=True)
    # an under-specified selection is ambiguous
    with pytest.raises(ValueError, match="expected exactly one"):
        lric_lineplot(adata, "lric_ct", source=adata.uns["lric_ct"]["source"].iloc[0], return_fig=True)
    # `source` is not a column of an agnostic result
    with pytest.raises(ValueError, match="not a column"):
        lric_lineplot(adata, "lric_ag", source="CD19+ B", return_fig=True)

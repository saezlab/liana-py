import plotnine as p9
import pytest

from liana.method.sp._LRIC import cross_pcf, lric
from liana.plotting import lric_lineplot
from liana.testing import generate_toy_spatial
from liana.testing._sample_resource import sample_resource

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
    feature = next(iter(adata.uns["cross_pcf"]["results"]))
    p = lric_lineplot(adata, "cross_pcf", feature, max_dist=60, return_fig=True)
    assert isinstance(p, p9.ggplot)
    # symmetric: the reversed pair resolves to the same curve
    lric_lineplot(adata, "cross_pcf", feature[::-1], return_fig=True)


def test_lric_agnostic_lineplot(adata):
    feature = adata.uns["lric_ag"]["pair_names"][0]
    p = lric_lineplot(adata, "lric_ag", feature, return_fig=True)
    assert isinstance(p, p9.ggplot)


def test_lric_pairwise_lineplot_decomposes(adata):
    res = adata.uns["lric_ct"]
    pair = next(iter(res["results"]))
    feature = (pair, res["pair_names"][0])
    p = lric_lineplot(adata, "lric_ct", feature, return_fig=True)
    assert isinstance(p, p9.ggplot)
    # full + architecture-only + expression-only
    assert set(p.data["curve"].unique()) == {"g (full)", "g_pcf", "g_expr"}


def test_missing_feature_raises(adata):
    with pytest.raises(KeyError):
        lric_lineplot(adata, "cross_pcf", ("nope", "missing"), return_fig=True)

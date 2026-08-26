import pytest

from liana.method.sp._LRIC import cross_pcf, lric
from liana.testing import generate_toy_spatial
from liana.testing._sample_resource import sample_resource
from liana.utils import get_lric_auc

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


@pytest.mark.parametrize("uns_key", ["cross_pcf", "lric_ag", "lric_ct"])
def test_get_lric_auc(adata, uns_key):
    df = get_lric_auc(adata, uns_key, max_dist=60, min_bins=2)
    assert list(df.columns) == ["feature", "label", "score"]
    assert len(df) > 0
    # sorted most-enriched first
    assert df["score"].is_monotonic_decreasing


def test_min_bins_gates_out_everything(adata):
    # more bins required than exist in the window -> empty, but well-formed
    df = get_lric_auc(adata, "cross_pcf", max_dist=25, min_bins=99)
    assert list(df.columns) == ["feature", "label", "score"]
    assert df.empty

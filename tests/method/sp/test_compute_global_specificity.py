import pytest
from anndata import AnnData

from liana.method import compute_global_specificity


def test_compute_global_specificity(pbmc68k: AnnData) -> None:
    compute_global_specificity(
        adata=pbmc68k, groupby="bulk_labels", lr_sep=None, n_perms=1, uns_key="global_interactions"
    )

    assert "global_interactions" in pbmc68k.uns
    res = pbmc68k.uns["global_interactions"]
    assert list(res.columns) == ["index", "feature", "lr_mean", "pval"]
    assert res["pval"].between(0.0, 1.0).all()


def test_raises_if_invalid_groupby(pbmc68k: AnnData) -> None:
    with pytest.raises(KeyError):
        compute_global_specificity(adata=pbmc68k, groupby="notagroup", n_perms=1, lr_sep=None)

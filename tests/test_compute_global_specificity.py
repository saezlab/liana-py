import pytest

from liana.method import compute_global_specificity


def test_compute_global_specificity():
    from scanpy.datasets import pbmc68k_reduced
    adata = pbmc68k_reduced()

    compute_global_specificity(
        adata=adata,
        groupby="bulk_labels",
        lr_sep=None,
        n_perms=1,
        uns_key="global_interactions")

    assert "global_interactions" in adata.uns
    res = adata.uns["global_interactions"]
    assert hasattr(res, "shape")
    assert res["pval"].between(0.0, 1.0).all()


def test_raises_if_invalid_groupby():
    from scanpy.datasets import pbmc68k_reduced
    adata = pbmc68k_reduced()
    with pytest.raises(KeyError):
        compute_global_specificity(
            adata=adata,
            groupby="notagroup",
            n_perms=1,
            lr_sep=None
        )



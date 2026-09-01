import pytest
from anndata import AnnData
from pandas import DataFrame

from liana.datasets import _sample_dea
from liana.method import df_to_lr

groupby = "bulk_labels"


@pytest.fixture
def dea_df(toy_adata: AnnData) -> DataFrame:
    """Toy differential expression results, as e.g. decoupler would return them."""
    return _sample_dea(toy_adata, groupby)


def test_dea_to_lr(toy_adata: AnnData, dea_df: DataFrame) -> None:
    lr_res = df_to_lr(
        toy_adata,
        dea_df=dea_df,
        resource_name="consensus",
        expr_prop=0.1,
        groupby="bulk_labels",
        stat_keys=["stat", "pval", "padjusted"],
        use_raw=False,
        complex_col=None,
        verbose=True,
        min_cells=10,
        return_all_lrs=False,
    )
    assert lr_res.shape == (139, 22)
    # assert ligand_stat, ligand_pval, and ligand_padjusted are in lr_res.columns
    columns = lr_res.columns
    expected_columns = [
        "ligand",
        "ligand_stat",
        "ligand_pval",
        "ligand_padjusted",
        "ligand_expr",
        "receptor",
        "receptor_stat",
        "receptor_pval",
        "receptor_padjusted",
        "receptor_expr",
    ]
    for col in expected_columns:
        assert col in columns
    assert lr_res["interaction_padjusted"].mean() == 0.56700673783093


def test_dea_to_lr_params(toy_adata: AnnData, dea_df: DataFrame) -> None:
    lr_res = df_to_lr(
        toy_adata,
        dea_df=dea_df,
        expr_prop=0.1,
        min_cells=10,
        groupby="bulk_labels",
        stat_keys=["stat", "pval", "padjusted"],
        use_raw=False,
        complex_col="stat",
        verbose=True,
        return_all_lrs=True,
    )
    assert lr_res.shape == (3321, 23)

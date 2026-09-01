import numpy as np
from pandas import DataFrame
from tests._helpers import not_none, plot_data

from liana.plotting import dotplot, dotplot_by_sample


def test_dotplot_order(liana_res: DataFrame) -> None:
    my_p = not_none(
        dotplot(
            liana_res=liana_res,
            size="specificity_rank",
            colour="magnitude",
            top_n=20,
            orderby="specificity_rank",
            orderby_ascending=False,
            target_labels=["A", "B", "C"],
        )
    )
    assert "interaction" in plot_data(my_p).columns
    np.testing.assert_equal(np.unique(plot_data(my_p).interaction).shape, (20,))
    assert {"A", "B", "C"} == set(plot_data(my_p).target)


def test_doplot_filter(liana_res: DataFrame) -> None:
    my_p2 = not_none(
        dotplot(
            liana_res=liana_res,
            size="specificity_rank",
            colour="magnitude",
            filter_fn=lambda x: x["specificity_rank"] > 0.95,
            inverse_colour=True,
            source_labels=["A"],
        )
    )
    assert set(plot_data(my_p2)["source"]) == {"A"}
    # we force this, but not intended all interactions
    # to be only 0.95, but rather for an interaction to get
    # plotted, in at least one cell type pair it should be > 0.95
    assert all(plot_data(my_p2)["specificity_rank"] > 0.95) is True

    # `inverse_colour` replaces the colour column with its -log10
    np.testing.assert_allclose(
        plot_data(my_p2)["magnitude"],
        -np.log10(liana_res.loc[plot_data(my_p2).index, "magnitude"] + np.finfo(float).eps),
    )


def test_dotplot_bysample(liana_res_by_sample: DataFrame) -> None:
    my_p3 = not_none(
        dotplot_by_sample(
            liana_res=liana_res_by_sample,
            size="specificity_rank",
            colour="magnitude",
            target_labels="E",
            sample_key="sample",
        )
    )
    assert "interaction" in plot_data(my_p3).columns
    assert "sample" in plot_data(my_p3).columns
    # `target_labels` keeps only the labels asked for
    assert set(plot_data(my_p3)["target"]) == {"E"}

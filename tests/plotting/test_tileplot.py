from pandas import DataFrame
from tests._helpers import not_none, plot_data

from liana.plotting import tileplot


def test_tileplot(liana_res: DataFrame) -> None:
    my_p2 = not_none(
        tileplot(
            liana_res=liana_res,
            # NOTE: fill & label need to exist for both
            # ligand_ and receptor_ columns
            fill="means",
            label="pvals",
            label_fn=lambda x: f"{x:.2f}",
            top_n=10,
            orderby="specificity_rank",
            orderby_ascending=True,
        )
    )
    assert isinstance(plot_data(my_p2)["pvals"].to_numpy()[0], str)
    # `top_n` keeps the n best interactions by `orderby`, in that order
    assert plot_data(my_p2)["interaction"].notna().all()
    assert plot_data(my_p2)["interaction"].nunique() == 10
    interaction = liana_res["ligand_complex"] + " -> " + liana_res["receptor_complex"]
    best = liana_res.groupby(interaction)["specificity_rank"].min().sort_values().head(10).index.tolist()
    assert list(plot_data(my_p2)["interaction"].cat.categories) == best

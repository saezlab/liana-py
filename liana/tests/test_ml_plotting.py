import numpy as np
from ..testing._toy_ml import get_toy_ml
from ..plotting import dotplot, set_dotplot



adata = get_toy_ml()
CCC_res = adata.uns['CCC_res']

def test_gene_plot():
    gp = set_dotplot(adata=adata,
                   metabolite = 'A',
                   groupby='bulk_labels',
    )
    assert gp is not None

def test_check_dotplot_order():
    my_p = dotplot(liana_res=CCC_res,
                    size='specificity_rank',
                    colour='magnitude',
                    top_n=20,
                    orderby='specificity_rank',
                    orderby_ascending=False,
                    target_labels=["A", "B", "C"],
                    met = True
                    )
    assert my_p is not None
    assert 'interaction' in my_p.data.columns
    np.testing.assert_equal(np.unique(my_p.data.interaction).shape, (20,))
    set(my_p.data.target)
    assert {'A', 'B', 'C'} == set(my_p.data.target)


def test_check_dotplot_filter():
    my_p2 = dotplot(liana_res=CCC_res,
                    size='specificity_rank',
                    colour='magnitude',
                    filterby='specificity_rank',
                    filter_lambda=lambda x: x > 0.95,
                    inverse_colour=True,
                    source_labels=["A"], 
                    met = True
                    )
    assert my_p2 is not None
    # we force this, but not intended all interactions
    # to be only 0.95, but rather for an interaction to get
    # plotted, in at least one cell type pair it should be > 0.95
    assert all(my_p2.data['specificity_rank'] > 0.95) is True




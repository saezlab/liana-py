import numpy as np
from ..testing._toy_ml import get_toy_ml
from ..plotting import gene_plot


adata = get_toy_ml()

def test_gene_plot():
    gp = gene_plot(adata=adata,
                   metabolite = 'A',
                   groupby='bulk_labels',
    )
    assert gp is not None




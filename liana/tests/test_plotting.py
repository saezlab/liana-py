import numpy as np
from liana.plotting import dotplot, dotplot_by_sample, tileplot, circle
from liana.testing import sample_lrs
from liana.testing import generate_toy_spatial, generate_toy_adata
from pytest import raises

liana_res = sample_lrs()

def test_dotplot_order():
    my_p = dotplot(liana_res=liana_res,
                   size='specificity_rank',
                   colour='magnitude',
                   top_n=20,
                   orderby='specificity_rank',
                   orderby_ascending=False,
                   target_labels=["A", "B", "C"]
                   )
    assert my_p is not None
    assert 'interaction' in my_p.data.columns
    np.testing.assert_equal(np.unique(my_p.data.interaction).shape, (20,))
    set(my_p.data.target)
    assert {'A', 'B', 'C'} == set(my_p.data.target)


def test_doplot_filter():
    my_p2 = dotplot(liana_res=liana_res,
                    size='specificity_rank',
                    colour='magnitude',
                    filter_fun=lambda x: x['specificity_rank'] > 0.95,
                    inverse_colour=True,
                    source_labels=["A"]
                    )
    assert my_p2 is not None
    # we force this, but not intended all interactions
    # to be only 0.95, but rather for an interaction to get
    # plotted, in at least one cell type pair it should be > 0.95
    assert all(my_p2.data['specificity_rank'] > 0.95) is True


def test_dotplot_bysample():
    liana_res = sample_lrs(by_sample=True)
    my_p3 = dotplot_by_sample(liana_res=liana_res,
                              size='specificity_rank',
                              colour='magnitude',
                              target_labels='E',
                              sample_key='sample')
    assert my_p3 is not None
    assert 'interaction' in my_p3.data.columns
    assert 'sample' in my_p3.data.columns
    assert 'B' not in my_p3.data['target']


def test_tileplot():
    my_p2 = tileplot(liana_res = liana_res,
                     # NOTE: fill & label need to exist for both
                     # ligand_ and receptor_ columns
                     fill='means',
                     label='pvals',
                     label_fun=lambda x: f'{x:.2f}',
                     top_n=10,
                     orderby='specificity_rank',
                     orderby_ascending=True
                     )
    assert my_p2 is not None
    assert isinstance(my_p2.data['pvals'].values[0], str)


def test_proximity_plot():
    from liana.plotting import connectivity

    adata = generate_toy_spatial()
    my_p4 = connectivity(adata=adata, idx=0)
    assert my_p4 is not None


def test_circle_plot():
    liana_res = sample_lrs()
    adata = generate_toy_adata()
    adata.uns['liana_res'] = liana_res

    raises(ValueError,
           circle,
           adata=adata,
           uns_key='liana_res',
           groupby='bulk_labels',
           source_key='source',
           target_key='target',
           score_key='specificity_rank',
           source_labels='B',
           target_labels=['C'],
           pivot_mode='counts',
           mask_mode='or',
           figure_size=(5, 5),
           edge_alpha=0.5,
           edge_arrow_size=10
           )

    liana_res['source'] = adata.obs['bulk_labels'].sample(n=liana_res.shape[0], replace=True).values
    liana_res['target'] = adata.obs['bulk_labels'].sample(n=liana_res.shape[0], replace=True).values

    p = circle(adata=adata,
               uns_key='liana_res',
               groupby='bulk_labels',
               source_key='source',
               target_key='target',
               score_key='specificity_rank',
               source_labels=['CD19+ B', 'CD8+/CD45RA+ Naive Cytotoxic', 'CD8+ Cytotoxic T'],
               target_labels=['Dendritic', 'CD14+ Monocyte'],
               pivot_mode='counts',
               mask_mode='or',
               figure_size=(5, 5),
               edge_alpha=0.5,
               edge_arrow_size=10
               )
    p is not None

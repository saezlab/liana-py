import numpy as np

from liana.plotting import dotplot, dotplot_by_sample


def test_dotplot_order(liana_res):
    my_p = dotplot(liana_res=liana_res,
                   size='specificity_rank',
                   colour='magnitude',
                   top_n=20,
                   orderby='specificity_rank',
                   orderby_ascending=False,
                   target_labels=["A", "B", "C"]
                   )
    assert 'interaction' in my_p.data.columns
    np.testing.assert_equal(np.unique(my_p.data.interaction).shape, (20,))
    assert {'A', 'B', 'C'} == set(my_p.data.target)


def test_doplot_filter(liana_res):
    my_p2 = dotplot(liana_res=liana_res,
                    size='specificity_rank',
                    colour='magnitude',
                    filter_fun=lambda x: x['specificity_rank'] > 0.95,
                    inverse_colour=True,
                    source_labels=["A"]
                    )
    assert set(my_p2.data['source']) == {'A'}
    # we force this, but not intended all interactions
    # to be only 0.95, but rather for an interaction to get
    # plotted, in at least one cell type pair it should be > 0.95
    assert all(my_p2.data['specificity_rank'] > 0.95) is True

    # `inverse_colour` replaces the colour column with its -log10
    np.testing.assert_allclose(
        my_p2.data['magnitude'],
        -np.log10(liana_res.loc[my_p2.data.index, 'magnitude'] + np.finfo(float).eps),
    )


def test_dotplot_bysample(liana_res_by_sample):
    my_p3 = dotplot_by_sample(liana_res=liana_res_by_sample,
                              size='specificity_rank',
                              colour='magnitude',
                              target_labels='E',
                              sample_key='sample')
    assert 'interaction' in my_p3.data.columns
    assert 'sample' in my_p3.data.columns
    # `target_labels` keeps only the labels asked for
    assert set(my_p3.data['target']) == {'E'}

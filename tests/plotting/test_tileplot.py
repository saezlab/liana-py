from liana.plotting import tileplot


def test_tileplot(liana_res):
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

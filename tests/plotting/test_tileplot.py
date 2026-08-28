from liana.plotting import tileplot


def test_tileplot(liana_res):
    my_p2 = tileplot(liana_res = liana_res,
                     # NOTE: fill & label need to exist for both
                     # ligand_ and receptor_ columns
                     fill='means',
                     label='pvals',
                     label_fn=lambda x: f'{x:.2f}',
                     top_n=10,
                     orderby='specificity_rank',
                     orderby_ascending=True
                     )
    assert isinstance(my_p2.data['pvals'].values[0], str)
    # `top_n` keeps the n best interactions by `orderby`, in that order
    assert my_p2.data['interaction'].notna().all()
    assert my_p2.data['interaction'].nunique() == 10
    interaction = liana_res['ligand_complex'] + ' -> ' + liana_res['receptor_complex']
    best = (liana_res.groupby(interaction)['specificity_rank'].min()
            .sort_values().head(10).index.tolist())
    assert list(my_p2.data['interaction'].cat.categories) == best

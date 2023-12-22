from liana.testing import _sample_dea
from liana.multi import df_to_lr
from liana.testing._sample_anndata import generate_toy_adata


# Create a toy AnnData object
adata = generate_toy_adata()
groupby = 'bulk_labels'
dea_df = _sample_dea(adata, groupby)


def test_dea_to_lr():
    lr_res = df_to_lr(adata,
                       dea_df=dea_df,
                       resource_name='consensus',
                       expr_prop=0.1,
                       groupby='bulk_labels',
                       stat_keys=['stat', 'pval', 'padjusted'],
                       use_raw=False,
                       complex_col=None,
                       verbose=True,
                       min_cells=10,
                       return_all_lrs=False,
                       )
    assert lr_res.shape == (374, 22)
    # assert ligand_stat, ligand_pval, and ligand_padjusted are in lr_res.columns
    columns = lr_res.columns
    expected_columns = ['ligand', 'ligand_stat', 'ligand_pval', 'ligand_padjusted', 'ligand_expr',
                        'receptor', 'receptor_stat', 'receptor_pval', 'receptor_padjusted', 'receptor_expr']
    for col in expected_columns:
        assert col in columns
    assert lr_res['interaction_padjusted'].mean() == 0.5540001846991026

def test_dea_to_lr_params():
    lr_res = df_to_lr(adata,
                      dea_df=dea_df,
                      expr_prop=0.1,
                      min_cells=10,
                      groupby='bulk_labels',
                      stat_keys=['stat', 'pval', 'padjusted'],
                      use_raw=False,
                      complex_col='stat',
                      verbose=True,
                      return_all_lrs=True,
                      )
    assert lr_res.shape == (3321, 23)

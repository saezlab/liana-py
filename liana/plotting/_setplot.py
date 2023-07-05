from __future__ import annotations

import numpy as np
import pandas

from plotnine import ggplot, geom_point, aes, facet_wrap, \
    labs, theme_bw, theme, element_text 


def setplot(adata, metabolite, groupby, return_fig=True, use_raw=True):

    df = get_gene_dfs(id=metabolite, adata=adata, groupby=groupby, use_raw=use_raw)
    
    
    p = (ggplot(df, aes(x='bulk_labels', y='index', size='percentage', color='expression')) 
            + geom_point()
            + facet_wrap('~prod') 
            + theme_bw() 
            + theme(axis_text_x=element_text(angle=90, hjust=1)) 
            + labs(x='celltypes', 
                    y=f'Genes associated with {metabolite}'))


    if return_fig:
        return p

    p.draw()


def get_gene_dfs(id, adata, groupby, use_raw=True):

    prod_genes = adata.uns['mask'].index[adata.uns['mask'][id] == 1]
    deg_genes = adata.uns['mask'].index[adata.uns['mask'][id] == -1]

    if len(list(prod_genes) + list(deg_genes)) == 0:
        raise ValueError(f'No gene found for {id}!  Check if mask was passed correctly')

    if use_raw:
        prod_df = adata.raw.X[:,adata.var_names.isin(prod_genes)]
        deg_df = adata.raw.X[:,adata.var_names.isin(deg_genes)]
    else:
        prod_df = adata.X[:,adata.var_names.isin(prod_genes)]
        deg_df = adata.X[:,adata.var_names.isin(deg_genes)]

    p = adata.var_names[adata.var_names.isin(prod_genes)]
    deg = adata.var_names[adata.var_names.isin(deg_genes)]

    percentage = pandas.DataFrame(np.zeros((prod_df.shape[1], len(adata.obs[groupby].unique()))), columns=adata.obs[groupby].unique(), index=p)
    expression = pandas.DataFrame(np.zeros((prod_df.shape[1], len(adata.obs[groupby].unique()))), columns=adata.obs[groupby].unique(), index=p)

    for i in range(prod_df.shape[1]):
        for j in adata.obs[groupby].unique():
            d = prod_df[adata.obs[groupby] == j,i]
            percentage[j][i] = np.sum(d > 0)/d.shape[0]
            expression[j][i] = d.mean()

    prod_percentage = percentage
    prod_expression = expression

    percentage = pandas.DataFrame(np.zeros((deg_df.shape[1], len(adata.obs[groupby].unique()))), columns=adata.obs[groupby].unique(), index=deg)
    expression = pandas.DataFrame(np.zeros((deg_df.shape[1], len(adata.obs[groupby].unique()))), columns=adata.obs[groupby].unique(), index=deg)

    for i in range(deg_df.shape[1]):
        for j in adata.obs[groupby].unique():
            d = deg_df[adata.obs[groupby] == j,i]
            percentage[j][i] = np.sum(d > 0)/d.shape[0]
            expression[j][i] = d.mean()

    deg_percentage = percentage
    deg_expression = expression

    # melt the dataframe to make a dotplot and add prod as a column
    prod_percentage = prod_percentage.reset_index().melt(id_vars='index', var_name='bulk_labels', value_name='percentage')

    # make the same with prod_expression and merge both dataframes
    prod_expression = prod_expression.reset_index().melt(id_vars='index', var_name='bulk_labels', value_name='expression')

    prod = pandas.merge(prod_percentage, prod_expression, on=['index', 'bulk_labels'])

    # do the same for deg
    deg_percentage = deg_percentage.reset_index().melt(id_vars='index', var_name='bulk_labels', value_name='percentage')
    deg_expression = deg_expression.reset_index().melt(id_vars='index', var_name='bulk_labels', value_name='expression')
    deg = pandas.merge(deg_percentage, deg_expression, on=['index', 'bulk_labels'])

    # add a column to both dataframes that says if the gene is produced or degraded
    prod['prod'] = 'produced'
    deg['prod'] = 'degraded'

    # concatenate both dataframes
    df = pandas.concat([prod, deg])

    return df
        
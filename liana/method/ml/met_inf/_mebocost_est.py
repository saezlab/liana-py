# from liana.method.ml._ml_Method import MetabMethod, MetabMethodMeta

# #import liana.method.ml.scores as _mebocost
# import pandas as pd
# import numpy as np

# def _mebocost_estimation(me_res, adata, verbose) -> pd.DataFrame: 

#     method = 'mean'
#     met_gene = me_res
#     mIdList = met_gene['HMDB_ID'].unique().tolist()

#     with_exp_gene_m = []
#     met_from_gene = pd.DataFrame()
#     for mId in mIdList:
#         gene_pos = met_gene[(met_gene['HMDB_ID'] == mId) & (met_gene['direction'] == 'product')]['gene_name'].tolist()
#         gene_pos = set(gene_pos) & set(adata.var_names.tolist())
#         gene_neg = met_gene[(met_gene['HMDB_ID'] == mId) & (met_gene['direction'] == 'substrate')]['gene_name'].tolist()
#         gene_neg = set(gene_neg) & set(adata.var_names.tolist())

#         if len(gene_pos) == 0:
#             continue

#         with_exp_gene_m.append(mId)
#         pos_g_index = np.where(adata.var_names.isin(gene_pos))
#         pos_exp = pd.DataFrame(adata.T[pos_g_index].X.toarray(), 
#                             index = adata.var_names[pos_g_index].tolist(),
#                             columns = adata.obs_names.tolist())

#         if not gene_neg:
#             m_from_enzyme = pos_exp.agg(method)
#         else:
#             neg_g_index = np.where(adata.var_names.isin(gene_neg))
#             neg_exp = pd.DataFrame(adata.T[neg_g_index].X.toarray(), 
#                             index = adata.var_names[neg_g_index].tolist(),
#                             columns = adata.obs_names.tolist())
#             pos = pos_exp.agg(method)
#             neg = neg_exp.agg(method)
#             m_from_enzyme = pos - neg
#         met_from_gene = pd.concat([met_from_gene, m_from_enzyme], axis = 1)

#     met_from_gene.columns = with_exp_gene_m
#     met_from_gene = met_from_gene.T

#     return met_from_gene

#     # Initialize mebocost Meta
# _mebocost_est = MetabMethodMeta(est_method_name="MEBOCOST_EST",
#                                 score_method_name="MEBOCOST",
#                                 fun_est=_mebocost_estimation,
#                                 complex_cols= list,
#                                 add_cols=None, ## attention
#                                 fun=None,
#                                 magnitude=None, ## attention
#                                 magnitude_ascending=None,  ## attention
#                                 specificity=None, ## attention
#                                 specificity_ascending=True,  ## attention
#                                 permute=False,
#                                 est_reference='Zheng, R., Zhang, Y., Tsuji, T., Zhang, L., Tseng, Y.-H.& Chen,'
#                                  'K., 2022,“MEBOCOST: Metabolic Cell-Cell Communication Modeling '
#                                  'by Single Cell Transcriptome,” BioRxiv.',
#                                 score_reference=None
#                     )

# # Initialize callable Method instance
# mebocost_est = MetabMethod(_ESTIMATION=_mebocost_est, output = 'metabolites', _SCORE=None)

    
# # # write function that calculates the mebocost score from transcriptome and metabolome data
# # def _mebocost_estimation(me_res, adata, verbose) -> pd.DataFrame: 
# #     """
# #     Estimate metabolite abundances by weighting the abundances of the enzymes that produce/degrade them

# #     Parameters
# #     ----------
# #     me_res
# #         DataFrame with the degrading and producing enzymes for each metabolite
# #     adata   
# #         annData object with the transcriptome data  
        
# #     Returns
# #     -------
# #     tuple(MR_interaction, confidence_score)
    
# #     """

# #     method = 'mean'
# #     met_gene = me_res
# #     mIdList = met_gene['HMDB_ID'].unique().tolist()
# #     ## estimating for each met
# #     met_from_gene = pd.DataFrame()
# #     with_exp_gene_m = []
# #     for mId in mIdList:
# #         ## genes for the reaction of producing the metabolite
# #         gene_pos = met_gene[(met_gene['HMDB_ID'] == mId) & (met_gene['direction'] == 'product')]['gene_name'].tolist()
# #         gene_pos = list(set(gene_pos) &
# #                        set(adata.var_names.tolist()))
# #         ## genes for the reaction of taking the metabolite as substrate
# #         gene_neg = met_gene[(met_gene['HMDB_ID'] == mId) & (met_gene['direction'] == 'substrate')]['gene_name'].tolist()
# #         gene_neg = list(set(gene_neg) & set(adata.var_names.tolist())) if gene_neg else []
# #         ## estimate by aggerating gene_pos and gene_neg
# #         ## only estimate when there are genes of positive reactons
# #         if len(gene_pos) != 0:
# #             with_exp_gene_m.append(mId)
# #             ## gene pos matrix
# #             pos_g_index = np.where(adata.var_names.isin(gene_pos))
# #             pos_exp = pd.DataFrame(adata.T[pos_g_index].X.toarray(), 
# #                                 index = adata.var_names[pos_g_index].tolist(),
# #                                 columns = adata.obs_names.tolist())

# #             ## if neg genes, do subraction from pos genes
# #             if not gene_neg:
# #                 if method == 'mean':
# #                     m_from_enzyme = pos_exp.mean()
# #                 elif method == 'median':
# #                     m_from_enzyme = pos_exp.median()
# #                 elif method == 'max':
# #                     m_from_enzyme = pos_exp.max()
# #                 elif method == 'gmean':
# #                     m_from_enzyme = pos_exp.apply(lambda col: gmean(col))
# #                 else:
# #                     continue
# #             else:
# #                 ## gene neg matrix
# #                 neg_g_index = np.where(adata.var_names.isin(gene_neg))
# #                 neg_exp = pd.DataFrame(adata.T[neg_g_index].X.toarray(), 
# #                                 index = adata.var_names[neg_g_index].tolist(),
# #                                 columns = adata.obs_names.tolist())
# #                 if method == 'mean':
# #                     pos = pos_exp.mean()
# #                     neg = neg_exp.mean()
# #                 elif method == 'median':
# #                     pos = pos_exp.median()
# #                     neg = neg_exp.median()
# #                 elif method == 'max':
# #                     pos = pos_exp.max()
# #                     neg = neg_exp.max()
# #                 elif method == 'gmean':
# #                     pos = pos_exp.apply(lambda col: gmean(col))
# #                     neg = neg_exp.apply(lambda col: gmean(col), axis = 1)
# #                 else:
# #                     raise ValueError('method should be one of [mean, gmean, median, max]')
# #                 m_from_enzyme = pos - neg
# #             met_from_gene = pd.concat([met_from_gene,
# #                                              m_from_enzyme], axis = 1)
# #     met_from_gene.columns = with_exp_gene_m
# #     ## tranpose
# #     met_from_gene = met_from_gene.T ## rows = metabolite (HMDB ID), columns = cells


# #     return met_from_gene


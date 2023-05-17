import numpy as np
import pandas as pd
import logging
from scipy.sparse import isspmatrix_csr, csr_matrix

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

from anndata import AnnData
from mudata import MuData


class MistyData(MuData):
    # TODO: change to SpatialData when Squidpy is also updated
    def __init__(self, data, obs, spatial_key, **kwargs):
        """
        Construct a MistyData object from a dictionary of views (anndatas).
        
        Parameters
        ----------
        data : `dict`
            Dictionary of views (anndatas). Requires an intra-view called "intra".
        obs : `pd.DataFrame`
            DataFrame of observations
        spatial_key : `str`
            Key in the .obsm attribute of each view that contains the spatial coordinates
        **kwargs : `dict`
            Keyword arguments passed to the MuData Super class
        """
        super().__init__(data, **kwargs)
        self.view_names = list(self.mod.keys())
        self.obs = obs
        self.spatial_key = spatial_key
        self._check_views()
    
    def _check_views(self):
        assert isinstance(self, MuData), "views must be a MuData object"
        assert "intra" in self.view_names, "views must contain an intra view"
        
        for view in self.view_names:
            if not isspmatrix_csr(self.mod[view].X):
                 logging.warning(f"view {view} is not a csr_matrix. Converting to csr_matrix")
                 self.mod[view].X = csr_matrix(self.mod[view].X)
            if view=="intra":
                continue
            if f"{self.spatial_key}_connectivities" not in self.mod[view].obsp.keys():
                raise ValueError(f"view {view} does not contain `{self.spatial_key}_connectivities` key in .obsp")

            
                
    
    def _get_conn(self, view_name):
        return self.mod[view_name].obsp[f"{self.spatial_key}_connectivities"]

    
    def __call__(self, 
                 n_estimators = 100,
                 bypass_intra = False,
                 keep_same_predictor = False, # NOTE: -> predict_self
                 k_cv = 10,
                 alphas = [0.1, 1, 10],
                 group_intra_by = None, # -> intra_groupby
                 group_env_by = None, # extra_groupby
                 n_jobs = -1,
                 seed = 1337,
                 inplace=True
                 ):
        """
        A Multi-view Learning for dissecting Spatial Transcriptomics data (MISTY) model.
        
        Parameters
        ----------
        n_estimators : `int`, optional (default: 100)
            Number of trees in the random forest models used to model single views
        bypass_intra : `bool`, optional (default: False)
            Whether to bypass modeling the intraview features importances via LOFO
        group_intra_by : `str`, optional (default: None)
            Column in the .obs attribute used to group cells in the intra-view
            If None, all cells are considered as one group
        group_env_by : `str`, optional (default: None)
            Column in the .obs attribute used to group cells in the extra-view(s)
            If None, all cells are considered as one group.
        alphas : `list`, optional (default: [0.1, 1, 10])
            List of alpha values used to choose from, that control the strength of the ridge regression,
            used for the multi-view part of the model
        k_cv : `int`, optional (default: 10)
            Number of folds for cross-validation used in the multi-view model
        n_jobs : `int`, optional (default: -1)
            Number of cores used to construct random forest models
        seed : `int`, optional (default: 1337)
            Specify random seed for reproducibility
        inplace : `bool`, optional (default: True)
            Whether to write the results to the .uns attribute of the object or return 
            two DataFrames, one for target metrics and one for importances.
        
        Returns
        -------
        If inplace is True, the results are written to the `.uns` attribute of the object.
        Else, two DataFrames are returned, one for target metrics and one for importances.

        """
        
        # TODO: function that checks if the groupby is in the obs
        # and does this for both extra & intra
        intra_groups = np.unique(self.obs[group_intra_by]) if group_intra_by else [None]
        extra_groups = np.unique(self.obs[group_env_by]) if group_env_by else [None]
        
        view_str = list(self.view_names)
        
        if bypass_intra:
            view_str.remove('intra')
        intra = self.mod['intra']
        
        # init list to store the results for each intra group and env group as dataframe;
        targets_list, importances_list = [], []
        intra_features = intra.var_names.to_list()
        
        # loop over each target and build one RF model for each view
        for target in intra_features:
            
            for intra_group in intra_groups:
                intra_obs_msk = intra.obs[group_intra_by] == \
                        intra_group if intra_group else np.ones(intra.shape[0], dtype=bool)
                
                # to array
                y = intra[intra_obs_msk, target].X.toarray().reshape(-1)
                # intra is always non-self, while other views can be self
                predictors_nonself, insert_index = _check_target_in_predictors(target, intra_features)

                # TODO: rename to target_importances
                importance_dict = {}
                
                # model the intraview
                if not bypass_intra:
                    obp_intra, importance_dict["intra"] = _single_view_model(y,
                                                                             intra,
                                                                             intra_obs_msk,
                                                                             predictors_nonself, 
                                                                             n_estimators,
                                                                             n_jobs,
                                                                             seed
                                                                             )
                    if insert_index is not None and keep_same_predictor: 
                        # add self-interactions as nan
                        importance_dict["intra"][target] = np.nan

                # loop over the group_views_by
                for extra_group in extra_groups:
                    # store the oob predictions for each view to construct predictor matrix for meta model
                    oob_list = []

                    if not bypass_intra:
                        oob_list.append(obp_intra)

                    # model the juxta and paraview (if applicable)
                    for view_name in [v for v in view_str if v != "intra"]:
                        extra = self.mod[view_name]
                        extra_obs_msk = self.obs[group_env_by] == extra_group if extra_group else np.ones(extra.shape[0], dtype=bool)
                        
                        extra_features = extra.var_names.to_list()
                        _predictors, _ =  _check_target_in_predictors(target, extra_features) if not keep_same_predictor else (extra_features, None)
                        
                        # NOTE: indexing here is expensive, but we do it to avoid memory issues
                        connectivity = self._get_conn(view_name)
                        view = _mask_connectivity(extra, connectivity, extra_obs_msk, _predictors)
                        
                        oob_predictions, importance_dict[view_name] = \
                            _single_view_model(y,
                                               view,
                                               intra_obs_msk,
                                               _predictors,
                                               n_estimators,
                                               n_jobs,
                                               seed
                                               )
                        oob_list.append(oob_predictions)

                    # train the meta model with k-fold CV 
                    intra_r2, multi_r2, coefs = _multi_model(y,
                                                             np.column_stack(oob_list),
                                                             intra_group,
                                                             bypass_intra,
                                                             view_str,
                                                             k_cv,
                                                             alphas, 
                                                             seed
                                                             )
                    
                    # write the results to a dataframe
                    targets_df = _format_targets(target,
                                                 intra_group,
                                                 extra_group,
                                                 view_str,
                                                 intra_r2,
                                                 multi_r2,
                                                 coefs
                                                 )
                    targets_list.append(targets_df)
                    
                    importances_df = _format_importances(target=target, 
                                                         intra_group=intra_group, 
                                                         extra_group=extra_group,
                                                         importance_dict=importance_dict
                                                         )
                    importances_list.append(importances_df)


        # create result dataframes
        target_metrics, importances = _concat_dataframes(targets_list,
                                                         importances_list,
                                                         view_str)
        
        if inplace:
            self.uns['target_metrics'] = target_metrics
            self.uns['importances'] = importances
        else:
            return target_metrics, importances


def _format_targets(target, intra_group, env_group, view_str, intra_r2, multi_r2, coefs):
    # TODO: Remove dot from column names
    target_df = pd.DataFrame({"target": target,
                              "intra_group": intra_group,
                              "env_group": env_group, 
                              "intra.R2": intra_r2,
                              "multi.R2": multi_r2},
                             index=[0]
                             )
    target_df["gain.R2"] = target_df["multi.R2"] - target_df["intra.R2"]
    target_df[view_str] = coefs
    
    return target_df


def _format_importances(target, intra_group, extra_group, importance_dict):
    
    importances_df = pd.DataFrame(importance_dict).reset_index().rename(columns={'index': 'predictor'})
    importances_df[['target', 'intra_group', 'extra_group']] = target, intra_group, extra_group
        
    return importances_df


def _concat_dataframes(targets_list, importances_list, view_str):
    target_metrics = pd.concat(targets_list, axis=0, ignore_index=True)
    
    target_metrics.loc[:, view_str] = target_metrics.loc[:, view_str].clip(lower=0)
    target_metrics.loc[:, view_str] = target_metrics.loc[:, view_str].div(target_metrics.loc[:, view_str].sum(axis=1), axis=0)
    
    importances = pd.concat(importances_list, axis=0, ignore_index=True)
    importances = pd.melt(importances,
                          id_vars=["target", "predictor", "intra_group", "extra_group"], 
                          value_vars=view_str, var_name="view", value_name="value")
    
    return target_metrics, importances



def _single_view_model(y, view, intra_obs_msk, predictors, n_estimators=100, n_jobs=-1, seed=1337):
    # TODO: remove this it's always sparse from select_mtx
    X = view[intra_obs_msk, predictors].X
        
    rf_model = RandomForestRegressor(n_estimators=n_estimators, 
                                     oob_score=True,
                                     n_jobs=n_jobs, 
                                     random_state=seed).fit(y=y, X=X)
    
    named_importances = dict(zip(predictors, rf_model.feature_importances_))
    
    return rf_model.oob_prediction_, named_importances


def _multi_model(y, oob_predictions, intra_group, bypass_intra, view_str, k_cv, alphas, seed):
    if oob_predictions.shape[0] < k_cv:
        logging.warn(f"Number of samples in {intra_group} is less than k_cv. "
                     "{intra_group} values set to NaN")
        return np.nan, np.nan, np.repeat(np.nan, len(view_str))
        
    kf = KFold(n_splits=k_cv, shuffle=True, random_state=seed)
    R2_vec_intra, R2_vec_multi = np.zeros(k_cv), np.zeros(k_cv)
    coef_mtx = np.zeros((k_cv, len(view_str)))
    
    for cv_idx, (train_index, test_index) in enumerate(kf.split(oob_predictions)):
        ridge_multi_model = RidgeCV(alphas=alphas).fit(X=oob_predictions[train_index], y=y[train_index])
        R2_vec_multi[cv_idx] = ridge_multi_model.score(X=oob_predictions[test_index], y=y[test_index])
        coef_mtx[cv_idx, :] = ridge_multi_model.coef_

        if not bypass_intra: 
            # NOTE: first column of obp is always intra only prediction if bypass_intra is False
            obp_train = oob_predictions[train_index, 0].reshape(-1, 1)
            obp_test = oob_predictions[test_index, 0].reshape(-1, 1)
            
            ridge_intra_model = RidgeCV(alphas=alphas).fit(X=obp_train, y=y[train_index])
            R2_vec_intra[cv_idx] = ridge_intra_model.score(X=obp_test, y=y[test_index])

    intra_r2 = R2_vec_intra.mean() if not bypass_intra else 0
    return intra_r2, R2_vec_multi.mean(), coef_mtx.mean(axis=0)


def _mask_connectivity(adata, connectivity, env_obs_msk, predictors):
    
    weights = connectivity.copy()
    weights[:, ~env_obs_msk] = 0
    X = weights @ adata[:, predictors].X
    view = AnnData(X=X, obs=adata.obs, var=pd.DataFrame(index=predictors))
    
    return view

# TODO: rename to _get_nonself
def _check_target_in_predictors(target, predictors):
    if target in predictors:
        insert_idx = np.where(np.array(predictors) == target)[0][0]
        predictors_subset = predictors.copy()
        predictors_subset.pop(insert_idx)
    else:
        predictors_subset = predictors
        insert_idx = None
    return predictors_subset, insert_idx

import numpy as np
import pandas as pd
from liana._logging import _logg
from tqdm import tqdm

from scipy.sparse import isspmatrix_csr, csr_matrix

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
import statsmodels.api as sm

from mudata import MuData

class MistyData(MuData):
    """MistyData Class used to construct multi-view objects"""
    def __init__(self, data, obs=None, spatial_key='spatial', **kwargs):
        """
        Construct a MistyData object from a dictionary of views (anndatas).
        
        Parameters
        ----------
        data : `dict`
            Dictionary of views (anndatas) or an mdata object.
            Requires an intra-view called "intra".
        obs : `pd.DataFrame`
            DataFrame of observations. If None, the obs of the intra-view is used.
        spatial_key : `str`
            Key in the .obsm attribute of each view that contains the spatial coordinates.
            Default is 'spatial'.
        **kwargs : `dict`
            Keyword arguments passed to the MuData Super class
        """
        if isinstance(data, MuData):
            data = data.mod
        
        super().__init__(data, **kwargs)
        self.view_names = list(self.mod.keys())
        self.spatial_key = spatial_key
        self._check_views()
        self.obs = obs if obs is not None else self.mod['intra'].obs

    def _check_views(self):
        assert isinstance(self, MuData), "views must be a MuData object"
        assert "intra" in self.view_names, "views must contain an intra view"
        
        for view in self.view_names:
            if not isspmatrix_csr(self.mod[view].X):
                _logg(f"view {view} is not a csr_matrix. Converting to csr_matrix", verbose=True, level='warn')
                self.mod[view].X = csr_matrix(self.mod[view].X)
            if view=="intra":
                continue
            if f"{self.spatial_key}_connectivities" not in self.mod[view].obsp.keys():
                raise ValueError(f"view {view} does not contain `{self.spatial_key}_connectivities` key in .obsp")
    
    def _get_conn(self, view_name):
        return self.mod[view_name].obsp[f"{self.spatial_key}_connectivities"]
    
    def __call__(self,
                 model='rf',
                 bypass_intra = False,
                 predict_self = False,
                 k_cv = 10,
                 alphas = [0.1, 1, 10],
                 maskby = None,
                 n_jobs = -1,
                 seed = 1337,
                 inplace=True,
                 verbose=False,
                 **kwargs
                 ):
        """
        A Multi-view Learning for dissecting Spatial Transcriptomics data (MISTy) model.
        
        Parameters
        ----------
        n_estimators : `int`, optional (default: 100)
            Number of trees in the random forest models used to model single views
        model : `str`, optional (default: 'rf')
            Model used to model the single views. Default is 'rf'.
            Can be either 'rf' (random forest) or 'linear' (linear regression).
        predict_self : `bool`, optional (default: False)
            Whether to predict self-interactions. These are determined purely by the feature names.
        bypass_intra : `bool`, optional (default: False)
            Whether to bypass modeling the intraview via LOFO.
            In other words, whether to bypass modelling each target by leave-one-feature-out within the same spots.
        maskby : `str`, optional (default: None)
            Column in the .obs attribute used to group or mask observations in the intra-view
            If None, all cells are considered as one group.
        k_cv : `int`, optional (default: 10)
            Number of folds for cross-validation used in the multi-view model, 
            and single-view models if model is 'linear'.
        alphas : `list`, optional (default: [0.1, 1, 10])
            List of alpha values used to choose from, that control the strength of the ridge regression,
            used for the multi-view part of the model. Only used if there are more than 2 views being modeled (including intra).
        n_jobs : `int`, optional (default: -1)
            Number of cores used to construct random forest models
        seed : `int`, optional (default: 1337)
            Specify random seed for reproducibility
        inplace : `bool`, optional (default: True)
            Whether to write the results to the .uns attribute of the object or return 
            two DataFrames, one for target metrics and one for importances.
        verbose : `bool`, optional (default: False)
            Whether to show a progress bar.
        **kwargs : `dict`
            Keyword arguments passed to the Regressors. Note that n_jobs & random_state are already set.
        
        Returns
        -------
        If inplace is True, the results are written to the `.uns` attribute of the object.
        Otherwise two DataFrames are returned, one for target metrics and one for importances.

        """
        view_str = list(self.view_names)
        obs_masks = _create_obs_masks(self.mod['intra'], maskby)
        
        if bypass_intra:
            view_str.remove('intra')
        intra = self.mod['intra']
        
        targets_list, importances_list = [], []
        intra_features = intra.var_names.to_list()
        progress_bar = tqdm(intra_features, disable=not verbose)
        
        for target in (progress_bar):
            for intra_group in obs_masks.keys():
                msk = obs_masks[intra_group]
                importance_dict = {}
                if verbose:
                    d = f"Now learning: {target}" + \
                        (f" masked by {intra_group}" if intra_group is not None else "")
                    progress_bar.set_description(d)
                    
                predictors_nonself, insert_index = _get_nonself(target, intra_features)
                y = intra[msk, target].X.toarray().reshape(-1)
                X = intra[msk, predictors_nonself].X.toarray()
                
                if not bypass_intra:
                    predictions_intra, importance_dict["intra"] = \
                        _single_view_model(y,
                                           X,
                                           predictors_nonself,
                                           model=model,
                                           k_cv=k_cv,
                                           seed=seed,
                                           n_jobs=n_jobs,
                                           **kwargs
                                           )
                    if insert_index is not None and predict_self: 
                        # add self-interactions as nan
                        importance_dict["intra"][target] = np.nan

                # store the predictions for each view to construct predictor matrix for meta model
                predictions_list = []

                if not bypass_intra:
                    predictions_list.append(predictions_intra)

                # model the juxta and paraview (if applicable)
                for view_name in [v for v in view_str if v != "intra"]:
                    extra = self.mod[view_name]
                    
                    extra_features = extra.var_names.to_list()
                    _predictors, _ =  _get_nonself(target, extra_features) if not predict_self else (extra_features, None)
                    
                    # NOTE: we multiply before masking
                    weights = self._get_conn(view_name)
                    X = weights @ extra[:, _predictors].X.toarray()
                    X = X[msk, :]
                    
                    predictions_extra, importance_dict[view_name] = \
                        _single_view_model(y,
                                           X,
                                           _predictors, 
                                           model=model,
                                           k_cv=k_cv,
                                           seed=seed,
                                           n_jobs=n_jobs,
                                           **kwargs
                                           )
                    predictions_list.append(predictions_extra)

                target_metrics = _multi_model(y,
                                              np.column_stack(predictions_list),
                                              intra_group,
                                              bypass_intra,
                                              view_str,
                                              target,
                                              k_cv,
                                              alphas, 
                                              seed
                                              )
                targets_list.append(target_metrics)
                
                importances_df = _format_importances(target=target,
                                                     intra_group=intra_group, 
                                                     importance_dict=importance_dict
                                                     )
                importances_list.append(importances_df)

        target_metrics, importances = _concat_dataframes(targets_list,
                                                         importances_list,
                                                         view_str)
        
        if inplace:
            self.uns['target_metrics'] = target_metrics
            self.uns['interactions'] = importances
        else:
            return target_metrics, importances


def _create_dict(**kwargs):
    return {k: v for k, v in kwargs.items() if v is not None}

def _format_targets(target, intra_group, view_str, intra_r2, multi_r2, coefs):
    d = _create_dict(target=target,
                     intra_group=intra_group,
                     intra_R2=intra_r2,
                     multi_R2=multi_r2,
                     gain_R2=multi_r2 - intra_r2,
                     )
    
    target_df = pd.DataFrame(d, index=[0])
    target_df[view_str] = coefs
    
    return target_df


def _format_importances(target, intra_group, importance_dict):
    
    importances_df = pd.DataFrame(importance_dict).reset_index().rename(columns={'index': 'predictor'})
    importances_df[['target', 'intra_group']] = target, intra_group
        
    return importances_df


def _concat_dataframes(targets_list, importances_list, view_str):
    target_metrics = pd.concat(targets_list, axis=0, ignore_index=True)
    importances = pd.concat(importances_list, axis=0, ignore_index=True)
    importances = pd.melt(importances,
                          id_vars=["target", "predictor", "intra_group"],
                          value_vars=view_str, 
                          var_name="view",
                          value_name="importances"
                          )
    
    # drop intra and extra group columns if they are all None
    importances = importances.dropna(axis=1, how='all')
    importances = importances.dropna(axis=0)
    
    return target_metrics, importances


def _single_view_model(y, X, predictors, model, k_cv, seed, n_jobs, **kwargs):
    if model=='rf':
        model = RandomForestRegressor(oob_score=True,
                                      n_jobs=n_jobs,
                                      random_state=seed,
                                      **kwargs,
                                      )
        model = model.fit(y=y, X=X)
        predictions = model.oob_prediction_
        importances = model.feature_importances_
        
    elif model=='linear':
        model = LinearRegression(n_jobs=1, **kwargs)
        predictions = cross_val_predict(model,
                                        X, y,
                                        cv=KFold(n_splits=k_cv,
                                                 random_state = seed,
                                                 shuffle=True),
                                        n_jobs=n_jobs
                                        )
        importances = sm.OLS(y, X).fit().tvalues
        
    else:
        raise ValueError(f"model {model} is not supported")
    
    named_importances = dict(zip(predictors, importances))
    
    return predictions, named_importances


def _multi_model(y, predictions, intra_group, bypass_intra, view_str, target, k_cv, alphas, seed):
    n_views = len(view_str)
    
    if (predictions.shape[0] < k_cv) or (y.var() == 0.0):
        if predictions.shape[0] < k_cv:
            error_message = (f"Number of samples is less than k_cv, {target} metrics set to NaN")
        else:
            error_message = (f"Variance of '{target}' is 0.0, metrics set to NaN")
        
        _logg(error_message, verbose=True, level='warn')
        return _format_targets(target,
                            intra_group,
                            view_str,
                            np.nan,
                            np.nan,
                            np.repeat(np.nan, n_views)
                            )
        
    kf = KFold(n_splits=k_cv, shuffle=True, random_state=seed)
    R2_vec_intra, R2_vec_multi = np.zeros(k_cv), np.zeros(k_cv)
    coef_mtx = np.zeros((k_cv, n_views))
    
    model = RidgeCV(alphas=alphas) if n_views > 2 else LinearRegression()
    
    for cv_idx, (train_index, test_index) in enumerate(kf.split(predictions)):
        multi_model = model.fit(X=predictions[train_index], y=y[train_index])
        R2_vec_multi[cv_idx] = multi_model.score(X=predictions[test_index], y=y[test_index])
        coef_mtx[cv_idx, :] = multi_model.coef_

        if not bypass_intra:
            pred_train = predictions[train_index, 0].reshape(-1, 1)
            pred_test = predictions[test_index, 0].reshape(-1, 1)
            
            intra_model = model.fit(X=pred_train, y=y[train_index])
            R2_vec_intra[cv_idx] = intra_model.score(X=pred_test, y=y[test_index])

    # format R2s
    intra_r2 = R2_vec_intra.mean().clip(min=0) if not bypass_intra else 0
    multi_r2 = R2_vec_multi.mean().clip(min=0)
    
    # format coefficients
    coefs = coef_mtx.mean(axis=0).clip(min=0)
    coefs = coefs / coefs.sum()
    
    # format metrics to a dataframe
    target_metrics = _format_targets(target,
                                     intra_group,
                                     view_str,
                                     intra_r2,
                                     multi_r2,
                                     coefs
                                     )
        
    return target_metrics

def _get_nonself(target, predictors):
    if target in predictors:
        insert_idx = np.where(np.array(predictors) == target)[0][0]
        predictors_subset = predictors.copy()
        predictors_subset.pop(insert_idx)
    else:
        predictors_subset = predictors
        insert_idx = None
    return predictors_subset, insert_idx

def _create_obs_masks(intra, maskby):
    obs_masks = {}
    # if maskby is a column of only boleans take it as is    
    if maskby is None:
        obs_masks[None] = np.ones(intra.shape[0], dtype=bool)
    elif intra.obs[maskby].dtype == bool:
        obs_masks[None] = intra.obs[maskby]
    # else if maskby is column of strings convert to categorical
    elif intra.obs[maskby].dtype == 'category':
        for intra_group in intra.obs[maskby].cat.categories:
            obs_masks[intra_group] = intra.obs[maskby] == intra_group
    else:
        raise ValueError(f"maskby column {maskby} must be a column of booleans or categorical")
        
        
    return obs_masks
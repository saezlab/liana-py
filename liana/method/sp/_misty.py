
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import identity, issparse
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import anndata as ad
import pandas as pd
import squidpy as sq


def _gauss_weight(distance_mtx, l):
    return np.exp(-distance_mtx**2 / l**2)


def _exponential_weight(distance_mtx, l):
    return np.exp(-distance_mtx / l)


def _get_distance_weights(adata, bandwidth, kernel="gaussian", add_self=True, spatial_key="spatial", zoi=0):
    kdtree = cKDTree(adata.obsm[spatial_key])
    if kernel == "gaussian":
        max_dist = 3*bandwidth
    elif kernel == "exponential":
        max_dist = 5*bandwidth
    sdm = kdtree.sparse_distance_matrix(kdtree, max_distance=max_dist, output_type="coo_matrix")
    sdm = sdm.tocsr()
    sdm.data[sdm.data < zoi] = np.inf
    if kernel == "gaussian":
        sdm.data = _gauss_weight(sdm.data, bandwidth)
    elif kernel == "exponential":
        sdm.data = _exponential_weight(sdm.data, bandwidth)
    if not add_self:
        sdm -= identity(n=sdm.shape[0], format="csr", dtype=sdm.dtype)
    return sdm


def _get_neighbors(adata, juxta_cutoff=np.inf, add_self=True, spatial_key="spatial"):
    neighbors, dists = sq.gr.spatial_neighbors(adata, coord_type="generic", 
                                               copy=True, delaunay=True, spatial_key=spatial_key)
    neighbors[dists > juxta_cutoff] = 0
    if add_self:
        neighbors += identity(neighbors.shape[0]) # add self for visium (not for the other assay though)
    return neighbors


def _check_features(adata, features, type_str):
    if features is not None:
        missing_features = set(features).difference(set(adata.var_names))
        if len(missing_features) > 0:
            logging.warning(f"Missing {type_str} features: {missing_features}.")
        features = list(set(features).difference(missing_features))
    else:
        features = adata.var_names.tolist()
    return features


def _get_paraview_groups(adata, predictors, bandwidth, group_env_by, kernel="gaussian", add_self=True, spatial_key="spatial", zoi=0):
    distance_weights = _get_distance_weights(adata=adata, bandwidth=bandwidth, kernel=kernel, add_self=add_self, spatial_key=spatial_key, zoi=zoi)
    paraviews = {}
    if group_env_by: 
        groups = np.unique(adata.obs[group_env_by])
        for group in groups:
            # set the weights for para cells that are not in the group to 0 (so we essentially make columns of the distance weight matrix 0,
            # but since the distance weight matrix is sparse, we need to do this in a slightly more complicated way
            # TODO: make this faster/better
            distance_weights_csr = distance_weights.copy()
            distance_weights_csr.data[np.isin(distance_weights_csr.indices, np.where(adata.obs[group_env_by]!=group)[0])] = 0
            paraviews[group] = ad.AnnData(X=distance_weights_csr@adata[:, predictors].X, obs=adata.obs, var=pd.DataFrame(index=predictors))
    else:
        paraviews["all"] = ad.AnnData(X=distance_weights@adata[:, predictors].X, obs=adata.obs, var=pd.DataFrame(index=predictors))
    return paraviews


def _get_juxtaview_groups(adata, predictors, group_env_by, juxta_cutoff=np.inf, add_self=True, spatial_key="spatial"):
    neighbors = _get_neighbors(adata, juxta_cutoff=juxta_cutoff, add_self=add_self, spatial_key=spatial_key)
    juxtaviews = {}
    if group_env_by:
        groups = np.unique(adata.obs[group_env_by])
        for group in groups:
            # see comment in get_paraview_groups
            neighbors_csr = neighbors.copy()
            neighbors_csr.data[np.isin(neighbors_csr.indices, np.where(adata.obs[group_env_by]!=group)[0])] = 0
            juxtaviews[group] = ad.AnnData(X=neighbors_csr@adata[:, predictors].X, obs=adata.obs, var=pd.DataFrame(index=predictors))
    else:
        juxtaviews["all"] = ad.AnnData(X=neighbors@adata[:, predictors].X, obs=adata.obs, var=pd.DataFrame(index=predictors))
    return juxtaviews


def _compose_views_groups(xdata, predictors, bypass_intra, add_juxta, add_para, group_env_by, juxta_cutoff, bandwidth, kernel, zoi, add_self, spatial_key):
    views = {}
    if not bypass_intra:
        views["intra"] = xdata
    if add_juxta:
        views["juxta"] = _get_juxtaview_groups(xdata, predictors, group_env_by=group_env_by, juxta_cutoff=juxta_cutoff, 
                                               add_self=add_self, spatial_key=spatial_key)
    if add_para:
        views["para"] = _get_paraview_groups(xdata, predictors, bandwidth=bandwidth, group_env_by=group_env_by, kernel=kernel, 
                                             add_self=add_self, spatial_key=spatial_key, zoi=zoi)
    return views


def _check_anndata_objects_groups(xdata, ydata, spatial_key, group_intra_by, group_env_by):
    # check that the adata_source and adata_target have the same number of spots/cells
    assert xdata.X.shape[0] == ydata.X.shape[0]

    # check that the adata_source and adata_target have the same spatial coordinates
    assert np.all(xdata.obsm[spatial_key] == ydata.obsm[spatial_key])
    
    # check the group_target_by and group_views_by variables
    if group_intra_by is not None:
        assert group_intra_by in ydata.obs.columns
    if group_env_by is not None:
        assert group_env_by in xdata.obs.columns


def _check_target_in_predictors(target, predictors):
    if target in predictors:
        insert_idx = np.where(np.array(predictors) == target)[0][0]
        predictors_subset = predictors.copy()
        predictors_subset.pop(insert_idx)
    else:
        predictors_subset = predictors
        insert_idx = None
    return predictors_subset, insert_idx


def _single_view_model(y, view, target_group_bool, predictors, n_estimators=100, n_jobs=-1, seed=1337):
    if issparse(view[target_group_bool, predictors].X):
        X = np.asarray(view[target_group_bool, predictors].X.todense())
    else:
        X = view[target_group_bool, predictors].X
    rf_model = RandomForestRegressor(n_estimators=n_estimators, oob_score=True, n_jobs=n_jobs, random_state=seed).fit(X=X, y=y)
    return rf_model.oob_prediction_, rf_model.feature_importances_


def _multi_model(y, oob_predictions, intra_group, bypass_intra, view_str, k_cv, alpha, seed):
    if oob_predictions.shape[0] < k_cv:
        logging.warn(f"Number of samples in {intra_group} is less than k_cv, using {oob_predictions.shape[0]} instead of {k_cv}")
        few_sample_flag = True
        k_cv = oob_predictions.shape[0]
    else:
        few_sample_flag = False
    kf = KFold(n_splits=k_cv, shuffle=True, random_state=seed)
    R2_vec_intra, R2_vec_multi = np.zeros(k_cv), np.zeros(k_cv)
    coef_mtx = np.zeros((k_cv, len(view_str)))
    for cv_idx, (train_index, test_index) in enumerate(kf.split(oob_predictions)):
        ridge_multi_model = Ridge(alpha=alpha).fit(X=oob_predictions[train_index], y=y[train_index])
        if few_sample_flag:
            R2_vec_multi[cv_idx] = ridge_multi_model.score(X=oob_predictions[test_index], y=y[test_index])
        else:
            R2_vec_multi[cv_idx] = ridge_multi_model.score(X=oob_predictions[test_index], y=y[test_index]).mean()
        coef_mtx[cv_idx, :] = ridge_multi_model.coef_

        if not bypass_intra: # first column of oob_predictions is always intra only prediction of bypass_intra is False
            ridge_intra_model = Ridge(alpha=alpha).fit(X=oob_predictions[train_index, 0].reshape(-1, 1), y=y[train_index])
            if few_sample_flag:
                R2_vec_intra[cv_idx] = ridge_intra_model.score(X=oob_predictions[test_index, 0].reshape(-1, 1), y=y[test_index])
            else:
                R2_vec_intra[cv_idx] = ridge_intra_model.score(X=oob_predictions[test_index, 0].reshape(-1, 1), y=y[test_index]).mean()

    intra_r2 = R2_vec_intra.mean() if not bypass_intra else 0
    return intra_r2, R2_vec_multi.mean(), coef_mtx.mean(axis=0)


def _make_dataframes(target, predictors, intra_group, env_group, view_str, intra_r2, multi_r2, coefs, importance_dict):
    performance_df = pd.DataFrame({"target": target, "intra_group": intra_group, "env_group": env_group, 
                                   "intra.R2": intra_r2, "multi.R2": multi_r2}, index=[0])
    coef_df = pd.DataFrame([coefs], columns=view_str, index=[0])
    coef_df = pd.DataFrame({"target": target, "intra_group": intra_group, "env_group": env_group}, index=[0]).join(coef_df)
    importance_df = pd.DataFrame({"target": np.repeat([target], len(predictors)), "predictor": predictors, 
                                    "intra_group": np.repeat([intra_group], len(predictors)),
                                    "env_group": np.repeat([env_group], len(predictors))})
    for view_name, importance_score in importance_dict.items():
        importance_df[view_name] = importance_score
    return performance_df, coef_df, importance_df


def _concat_dataframes(performances_list, contributions_list, importances_list, view_str):
    performances = pd.concat(performances_list, axis=0, ignore_index=True)
    performances["gain.R2"] = performances["multi.R2"] - performances["intra.R2"]
    contributions = pd.concat(contributions_list, axis=0, ignore_index=True)
    contributions.loc[:, view_str] = contributions.loc[:, view_str].clip(lower=0)
    contributions.loc[:, view_str] = contributions.loc[:, view_str].div(contributions.loc[:, view_str].sum(axis=1), axis=0)
    importances = pd.concat(importances_list, axis=0, ignore_index=True)
    importances = pd.melt(importances, id_vars=["target", "predictor", "intra_group", "env_group"], 
                          value_vars=view_str, var_name="view", value_name="value")
    return performances, contributions, importances


def misty(mdata, 
          x_mod,
          y_mod = None,
          targets = None,
          predictors = None,
          bandwidth = None, 
          juxta_cutoff = np.inf, 
          keep_same_predictor = False,  # TODO: maybe rename this variable
          zoi = 0, 
          kernel = "gaussian", 
          add_self = False, 
          spatial_key = "spatial", 
          add_juxta = True, 
          add_para = True, 
          bypass_intra = False, 
          group_intra_by = None, 
          group_env_by = None, 
          alpha = 1, 
          k_cv = 10, 
          n_estimators = 100,
          n_jobs = -1,
          seed = 1337,
          inplace = True,
          overwrite = False):
    """
    Misty: a multi-view integration method for spatial transcriptomics data.

    Parameters
    ----------
    mdata : `anndata`
        MuData object containing two modalities of interest
    x_mod : `str`
        Name of the modality to be used as predictors
    y_mod : `str`, optional (default: None)
        Name of the modality to be used as targets. If None, y_mod = x_mod
    targets : `list`, optional (default: None)
        List of targets to be used, must be features of y_mod 
        If None all feature of y_mod will be used.
    predictors : `list`, optional (default: None)
        List of predictors to be used, must be features of x_mod
        If None all feature of x_mod will be used.
    bandwidth : `float`, optional (default: None)
        Bandwidth for the Gaussian kernel.
    juxta_cutoff : `float`, optional (default: np.inf)
        Maximal distance to be considered a direct neighbor in the juxview.
    zoi : `float`, optional (default: 0)
        Radius that is ignored when computing the paraview.
        To be use if juxtaview and paraview should have no overlap
    kernel : `str`, optional (default: "gaussian")
        Possible values: "gaussian", "exponential"
    add_self : `bool`, optional (default: True)
        Whether to add self when constructing the juxtaview and paraview
        Should be set to True if using spots with several cells, e.g. 10X Visium
    spatial_key : `str`, optional (default: "spatial")
        Key in the .obsm attribute of the AnnData objects where the spatial coordinates are stored
    add_juxta : `bool`, optional (default: True)
        Whether to add the juxtaview 
    add_para : `bool`, optional (default: True)
        Whether to add the paraview
    bypass_intra : `bool`, optional (default: False)
        Whether to bypass modeling the intraview
    group_intra_by : `str`, optional (default: None)
        Column in the .obs attribute of mdata[x_mod] which is used to construct different models per group
        If None, all cells are considered to be in the same intra-group
    group_env_by : `str`, optional (default: None)
        Column in the .obs attribute of mdata[y_mod] which is used to construct juxta- and paraview per group
        If None, all cells are considered to be in the same environment
    alpha : `float`, optional (default: 1)
        Regularization parameter for the ridge regression (multi-view model)
    k_cv : `int`, optional (default: 10)
        Number of folds for cross-validation used in the multi-view model
    n_estimators : `int`, optional (default: 100)
        Number of trees in the random forest models used to model single views
    n_jobs : `int`, optional (default: -1)
        Number of cores used to construct random forest models
    seed : `int`, optional (default: 1337)
        Specify random seed for reproducibility
    inplace : `bool`, optional (default: True)
        Whether to add the results as dictionary to the mdata.uns["misty_results"] attribute or return the dictionary.
        The dictionary contains the following keys: "performances", "contributions", "importances"
    overwrite: `bool`, optional (default: False)
        Whether to overwrite existing results in mdata.uns["misty_results"] if inplace=True
    """
    
    # validate inputs 
    if not overwrite and ("misty_results" in mdata.uns.keys()) and inplace:
        raise ValueError("mdata already contains misty results. Set overwrite=True to overwrite.")
    if x_mod not in mdata.mod.keys():
        raise ValueError(f"Predictor modality {x_mod} not found in mdata.")
    if y_mod is not None and y_mod not in mdata.mod.keys():
        raise ValueError(f"Target modality {y_mod} not found in mdata.")
    if add_para and bandwidth is None:
        raise ValueError("bandwith must be specified if add_para=True")
    
    xdata = mdata[x_mod]
    ydata = mdata[y_mod] if y_mod else xdata

    _check_anndata_objects_groups(xdata, 
                                  ydata, 
                                  spatial_key=spatial_key, 
                                  group_intra_by=group_intra_by, 
                                  group_env_by=group_env_by)
    
    predictors = _check_features(xdata, predictors, type_str="predictors")
    targets = _check_features(ydata, targets, type_str="targets")

    intra_groups = np.unique(ydata.obs[group_intra_by]) if group_intra_by else [None]
    env_groups = np.unique(xdata.obs[group_env_by]) if group_env_by else [None]

    views = _compose_views_groups(xdata, 
                                  predictors,
                                  bypass_intra, 
                                  add_juxta, 
                                  add_para, 
                                  group_env_by, 
                                  juxta_cutoff, 
                                  bandwidth, 
                                  kernel, 
                                  zoi,
                                  add_self, 
                                  spatial_key)
    view_str = list(views.keys())

    # init list to store the results for each intra group and env group as dataframe;
    # in last step all dataframe are concatenated 
    performances_list, contributions_list, importances_list = [], [], []

    for intra_group in intra_groups:

        intra_group_bool = ydata.obs[group_intra_by] == intra_group if intra_group else np.ones(ydata.shape[0], dtype=bool)

        # loop over each target and build one RF model for each view
        for target in targets:

            if issparse(ydata.X):
                y = np.asarray(ydata[intra_group_bool, target].X.todense()).reshape(-1)
            else:
                y = ydata[intra_group_bool, target].X.reshape(-1)

            # remove target feature from predictors if it is in predictors
            predictor_subset, insert_index = _check_target_in_predictors(target, predictors)

            importance_dict = {}
            
            # model the intraview
            if not bypass_intra:
                oob_predictions_intra, importance_dict["intra"] = _single_view_model(y, 
                                                                                     views["intra"], 
                                                                                     intra_group_bool, 
                                                                                     predictor_subset, 
                                                                                     n_estimators, 
                                                                                     n_jobs, 
                                                                                     seed)
                if insert_index is not None and keep_same_predictor:
                    importance_dict["intra"] = np.insert(importance_dict["intra"], insert_index, np.nan)

            # loop over the group_views_by
            for env_group in env_groups:
                
                # store the oob predictions for each view to construct predictor matrix for meta model
                oob_list = []

                if not bypass_intra:
                    oob_list.append(oob_predictions_intra)

                # model the juxta and paraview (if applicable)
                for view_name in [v for v in view_str if v != "intra"]:
                    view = views[view_name][env_group] if env_group else views[view_name]["all"]
                    oob_predictions, importance_dict[view_name] = _single_view_model(y, 
                                                                                     view, 
                                                                                     intra_group_bool, 
                                                                                     predictors if keep_same_predictor else predictor_subset, 
                                                                                     n_estimators,
                                                                                     n_jobs, 
                                                                                     seed)
                    oob_list.append(oob_predictions)

                # train the meta model with k-fold CV 
                intra_r2, multi_r2, coefs = _multi_model(y, 
                                                         np.column_stack(oob_list),
                                                         intra_group, 
                                                         bypass_intra, 
                                                         view_str, 
                                                         k_cv, 
                                                         alpha, 
                                                         seed)

                # store the results
                performance_df, coef_df, importance_df = _make_dataframes(target, 
                                                                          predictors if keep_same_predictor else predictor_subset, 
                                                                          intra_group, 
                                                                          env_group, 
                                                                          view_str,
                                                                          intra_r2, 
                                                                          multi_r2, 
                                                                          coefs,
                                                                          importance_dict)
                performances_list.append(performance_df)
                contributions_list.append(coef_df)
                importances_list.append(importance_df)

    # create result dataframes
    performances, contributions, importances = _concat_dataframes(performances_list, 
                                                                  contributions_list, 
                                                                  importances_list, 
                                                                  view_str)
    if inplace:
        mdata.uns["misty_results"] = {"performances": performances, "contributions": contributions, "importances": importances}
    else:
        return {"performances": performances, "contributions": contributions, "importances": importances}


### start: temporary functions for testing ###
from IPython.core.display_functions import display
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import seaborn as sns


def plot_distance_weights(adata, bandwidth, cells, kernel="gaussian", add_self=True, spatial_key="spatial", zoi=0, **kwargs):
    distance_weights = _get_distance_weights(adata=adata, bandwidth=bandwidth, kernel=kernel, add_self=add_self, spatial_key=spatial_key, zoi=zoi)
    for cell_idx in cells:
        ax = plt.subplot(aspect="equal")
        im = ax.scatter(x=adata.obsm["spatial"][:, 0], y=adata.obsm["spatial"][:, 1], 
                        c=np.asarray(distance_weights[cell_idx, :].todense()).reshape(-1), **kwargs)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax.set_title(cell_idx)
        plt.show()


def plot_neighbors(adata, cells, juxta_cutoff=np.inf, add_self=True, spatial_key="spatial", **kwargs):
    neighbors = _get_neighbors(adata, juxta_cutoff=juxta_cutoff, add_self=add_self, spatial_key=spatial_key)
    for cell_idx in cells:
        ax = plt.subplot(aspect="equal")
        im = ax.scatter(x=adata.obsm["spatial"][:, 0], y=adata.obsm["spatial"][:, 1], 
                        c=np.asarray(neighbors[cell_idx, :].todense()).reshape(-1), **kwargs)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax.set_title(cell_idx)
        plt.show()


def plot_performance(performance_df, stat="gain.R2"):
    sns.stripplot(performance_df, x="target", y=stat,
              order = performance_df.sort_values(stat, ascending=False).target.tolist(),
              color = 'black', jitter=False)
    plt.xticks(rotation=90)
    plt.show()


def plot_contribution(contribution_df):
    views = contribution_df.drop(["target", "intra_group", "env_group"], axis=1).columns.tolist()
    contribution_df.plot.bar(stacked=True, x="target", y=views)
    plt.xticks(rotation=90)
    plt.show()


def plot_importance(importance_df, view, cutoff=0, intra_group=None, env_group=None):
    numeric_columns = importance_df.select_dtypes(include = ['int64', 'float64']).columns.tolist()
    importance_df[numeric_columns] += 1e-8
    mask = (importance_df["value"] > cutoff) 
    if intra_group:
        mask = mask & (importance_df["intra_group"] == intra_group)
    if env_group:
        mask = mask & (importance_df["env_group"] == env_group)
    df = importance_df.loc[mask]
    df = df[df["view"] == view].drop(["view", "intra_group", "env_group"], axis=1)
    display(df.pivot_table(index="target", columns="predictor", values="value").style.background_gradient(cmap='viridis'))
### end: temporary functions for testing ###

from __future__ import annotations

import numpy as np
import pandas as pd
from mudata import MuData
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import KFold
from tqdm import tqdm

from liana._constants import DefaultValues as V
from liana._constants import Keys as K
from liana._docs import d
from liana._logging import _logg
from liana.method.sp._misty._single_view_models import SingleViewModel


@d.dedent
class MistyData(MuData):
    """
    MistyData Class used to construct multi-view objects.

    Construct a MistyData object from a dictionary of views (anndatas).

    Parameters
    ----------
    data
        Dictionary of views (`AnnData`s) or a `MuData` object. Note that only the `data.X` attribute is used.
        An intra-view called "intra" is required.
    obs
        DataFrame of observations. If None, the obs of the intra-view is used.
    %(spatial_key)s
    enforce_obs
        If True, the number of observations in each extra-view must match the intra-view.
        Then the connectivities are stored in the .obsp attribute, while the weighted matrix is stored in .layers['weighted'].
        If False, the connectivities are stored in the .obsm attribute, while the weighted matrix is transposed and stored in .varm['weighted'].
    **kwargs
        Keyword arguments passed to the MuData Super class

    Attributes
    ----------
    view_names
        List of names of the different views
    spatial_key
        Key in `data.obsm` containing the spatial coordinates.
    enforce_obs
        See parameter with the same name.
    obs
        See parameter with the same name.

    """

    def __init__(self,
                 data: dict | MuData,
                 obs: pd.DataFrame | None = None,
                 spatial_key: str = K.spatial_key,
                 enforce_obs: bool = True,
                 **kwargs
                 ):
        source = data if isinstance(data, MuData) else None
        if source is not None:
            data = source.mod

        super().__init__(data, **kwargs)

        # preserve container-level attributes that MuData drops when rebuilt from .mod
        if source is not None:
            for attr in ("uns", "obsm", "varm", "obsp", "varp"):
                setattr(self, attr, getattr(source, attr))

        self.view_names = list(self.mod.keys())
        self.spatial_key = spatial_key
        self.enforce_obs = enforce_obs
        self._check_views()
        self.obs = obs if obs is not None else self.mod['intra'].obs

    def _check_views(self):
        assert isinstance(self, MuData), "views must be a MuData object"
        assert "intra" in self.view_names, "views must contain an intra view"

        for view in self.view_names:
            if view=="intra":
                continue
            if self.enforce_obs:
                if f"{self.spatial_key}_connectivities" not in self.mod[view].obsp.keys():
                    raise ValueError(f"view {view} does not contain `{self.spatial_key}_connectivities` key in .obsp")
                if self.mod[view].shape[0] != self.mod['intra'].shape[0]:
                    raise ValueError(f"view {view} has {self.mod[view].shape[0]} observations, " + \
                                    f"but the intra-view has {self.mod['intra'].shape[0]} observations")
            else:
                if f"{self.spatial_key}_connectivities" not in self.mod[view].obsm.keys():
                    raise ValueError(f"view {view} does not contain `{self.spatial_key}_connectivities` key in .obsm")

            self._set_weighted_matrix(view)

    def _set_weighted_matrix(self, view_name):
        if self.enforce_obs:
            weights = self.mod[view_name].obsp[f"{self.spatial_key}_connectivities"]
            self.mod[view_name].layers['weighted'] = weights @ self.mod[view_name].X
        else:
            weights = self.mod[view_name].obsm[f"{self.spatial_key}_connectivities"].T
            self.mod[view_name].varm['weighted'] = (weights @ self.mod[view_name].X).T

    def get_weighted_matrix(self,
                            view_name: str,
                            predictors: list[str] = None
                            ) -> pd.Index | np.ndarray:
        """
        Returns the weighted matrix for a given set of predictors in a view.

        Parameters
        ----------
        view_name
            Name of the view of interest.
        predictors
            List of predictors from which to retrieve the weights.

        Returns
        -------
        Weighted matrix of the requested view and predictors. If no predictors are provided, returns the variable names.

        """
        if predictors is None:
            predictors = self.mod[view_name].var_names

        if self.enforce_obs:
            return self.mod[view_name][:, predictors].layers['weighted']
        else:
            return self.mod[view_name][:, predictors].varm['weighted'].T

    @d.dedent
    def __call__(self,
                 model: SingleViewModel,
                 bypass_intra: bool = False,
                 predict_self: bool = False,
                 maskby: str = None,
                 k_cv: int = 10,
                 alphas: np.array | list[float] = np.array([0.1, 1, 10]),
                 seed: int = V.seed,
                 inplace: bool = V.inplace,
                 verbose: bool = V.verbose,
                 **kwargs
                 ) -> None | tuple[pd.DataFrame, pd.DataFrame]:
        """
        A Multi-view Learning for dissecting Spatial Transcriptomics data (MISTy) model.

        Parameters
        ----------
        model
            Single-view model of class SingleViewModel. Default options are RandomForestModel, LinearModel, and RobustLinearModel
            available via ``liana.method.sp._misty._single_view_models``.
        bypass_intra
            Whether to bypass modeling the intraview via leave-one-feature-out (LOFO).
            In other words, whether to bypass modelling each target by LOFO within the same spots.
        predict_self
            Whether to predict self-interactions. These are determined purely by the feature names.
        maskby
            Column in the .obs attribute used to group or mask observations in the intra-view
            If None, all cells are considered as one group.
        k_cv
            Number of folds for cross-validation used in the multi-view model,
            and single-view models if model is 'linear'.
        alphas
            List of alpha values used to choose from, that control the strength of the ridge regression,
            used for the multi-view part of the model. Only used if there are more than 2 views being modeled (including intra).
        %(seed)s
        %(inplace)s
        %(verbose)s
        **kwargs
            Keyword arguments passed to the Regressors. Note that random_state is already set via ``seed``.

        Returns
        -------
        If inplace is True, the results are written to the `.uns` attribute of the object.
        Otherwise two DataFrames are returned, one for target metrics and one for importances.

        """
        model = model(seed, **kwargs)  # type: ignore[operator]
        view_str = list(self.view_names)
        # Dict of ligand-receptor interactions by receptor, used to filter
        # predictors in the extra views.
        ligands_by_receptor = self.uns.get(
            "_misty_ligands_by_receptor"
        )

        is_celltype_misty = self.uns.get("_misty_by_cell_type", False)

        if is_celltype_misty:
            if maskby is None:
                maskby = "_misty_receiver"
            elif maskby != "_misty_receiver":
                raise ValueError(
                    "lrMistyDataByCellType requires "
                    "maskby='_misty_receiver'."
                )

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
                importance_dict: dict = {}
                if verbose:
                    d = f"Now learning: {target}" + \
                        (f" masked by {intra_group}" if intra_group is not None else "")
                    progress_bar.set_description(d)

                predictors_nonself, insert_index = _get_nonself(target, intra_features)
                y = intra[msk, target].X.toarray().flatten()
                X = intra[msk, predictors_nonself].X.toarray()

                if not bypass_intra:
                    model.fit(y=y,
                              X=X,
                              predictors=predictors_nonself,
                              k_cv=k_cv,
                              )
                    predictions_intra, importance_dict["intra"] = \
                        model.predictions, model.importances

                    if insert_index is not None and predict_self:
                        # add self-interactions as nan
                        importance_dict["intra"][target] = np.nan

                # store the predictions for each view to construct predictor matrix for meta model
                predictions_list: list = []
                target_view_str: list[str] = []

                if not bypass_intra:
                    predictions_list.append(predictions_intra)
                    target_view_str.append("intra")

                # model the juxta and paraview (if applicable)
                for view_name in [v for v in view_str if v != "intra"]:
                    extra = self.mod[view_name]

                    extra_features = extra.var_names.to_list()
                    if ligands_by_receptor is None:
                        _predictors = (
                            extra_features
                            if predict_self
                            else _get_nonself(target, extra_features)[0]
                        )
                    else:
                        allowed_ligands = set(
                            ligands_by_receptor.get(target, [])
                        )

                        candidate_predictors = [
                            ligand
                            for ligand in extra_features
                            if ligand in allowed_ligands
                        ]

                        if predict_self:
                            _predictors = candidate_predictors
                        else:
                            _predictors, _ = _get_nonself(
                                target,
                                candidate_predictors,
                            )

                    # A sender view may not contain any ligands paired with
                    # this target. It must not be fitted or included in the
                    # target-specific meta-model in that case.
                    if not _predictors:
                        continue

                    X = self.get_weighted_matrix(view_name, _predictors).toarray()
                    X = X[msk, :]
                    model.fit(y=y,
                              X=X,
                              predictors=_predictors,
                              k_cv=k_cv,
                              )
                    predictions_extra, importance_dict[view_name] = \
                        model.predictions, model.importances

                    predictions_list.append(predictions_extra)
                    target_view_str.append(view_name)

                if predictions_list:
                    target_metrics = _multi_model(
                        y,
                        np.column_stack(predictions_list),
                        intra_group,
                        bypass_intra,
                        target_view_str,
                        target,
                        k_cv,
                        alphas,
                        seed,
                    )
                else:
                    target_metrics = _format_targets(
                        target,
                        intra_group,
                        view_str,
                        np.nan,
                        np.nan,
                        np.repeat(np.nan, len(view_str)),
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

        if is_celltype_misty:
            receiver_celltype = self.uns["_misty_receiver_celltype"]
            target_metrics["receiver_celltype"] = receiver_celltype
            importances["receiver_celltype"] = receiver_celltype
            if "view" in importances.columns:
                is_extra = importances["view"].str.startswith("extra_")
                importances["sender_celltype"] = (
                    importances["view"]
                    .str.replace("^extra_", "", regex=True)
                    .where(is_extra)
                )
            else:
                importances["sender_celltype"] = np.nan

        if inplace:
            self.uns[K.target_metrics] = target_metrics
            self.uns[K.interactions] = importances
            return None
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
    for view_name in view_str:
        if view_name not in importances.columns:
            importances[view_name] = np.nan
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


def _multi_model(y, predictions, intra_group, bypass_intra, view_str, target, k_cv, alphas, seed):
    n_views = len(view_str)

    if (predictions.shape[0] < k_cv) or (y.var() == 0.0):
        if predictions.shape[0] < k_cv:
            warning_message = (f"Number of samples is less than k_cv, {target} metrics set to NaN")
        else:
            warning_message = (f"Variance of '{target}' is 0.0, metrics set to NaN")

        _logg(warning_message, verbose=True, level='warn')
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

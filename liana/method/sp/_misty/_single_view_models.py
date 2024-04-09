from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import numpy as np
from sklearn.model_selection import KFold, cross_val_predict

class SingleViewModel:
    """
    Base class for single view models. Subclasses should implement the fit method.
    """
    def __init__(self, seed, **kwargs):
        self.seed = seed
        self.kwargs = kwargs  # Store kwargs to be used in fit method
        self.model = None
        self.predictions = None
        self.importances = None

    def fit(self, y, X, predictors, k_cv=None):
        """
        Fit the model to the data and store the predictions and importances.

        Parameters
        ----------
        y : np.ndarray
            Target variable
        X : np.ndarray
            Feature matrix
        predictors : list
            List of feature names
        k_cv : int
            Number of cross-validation folds. If None, no cross-validation is performed.
        """
        raise NotImplementedError("This method should be implemented by subclasses")

    def _k_fold_predict(self, y, X, k_cv, fit_method):
        predictions = np.zeros_like(y)
        kf = KFold(n_splits=k_cv, random_state=self.seed, shuffle=True)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train = y[train_index]
            model = fit_method(y_train, X_train)
            y_pred = model.predict(X_test)
            predictions[test_index] = y_pred.flatten()
        return predictions

class RandomForestModel(SingleViewModel):
    """
    Random forest model (from sklearn) using out-of-bag predictions for feature importances.
    """

    def fit(self, y, X, predictors, k_cv=None):
        self.model = RandomForestRegressor(oob_score=True, random_state=self.seed, **self.kwargs)
        self.model.fit(X, y)
        self.predictions = self.model.oob_prediction_
        self.importances = dict(zip(predictors, self.model.feature_importances_))

class LinearModel(SingleViewModel):
    """
    Linear regression model using statsmodels.OLS for feature importances, and cross_val_predict with sklearn LinearRegression for predictions.
    **kwargs are passed to statsmodels.OLS.
    """
    def fit(self, y, X, predictors, k_cv):
        # pop n_jobs if it exists
        n_jobs = self.kwargs.pop('n_jobs', -1)
        model = LinearRegression(n_jobs=1)
        self.predictions = cross_val_predict(model,
                                             X, y,
                                             cv=KFold(n_splits=k_cv,
                                                      random_state = self.seed,
                                                      shuffle=True),
                                             n_jobs=n_jobs
                                             )
        X = sm.add_constant(X)
        model_full = sm.OLS(y, X, **self.kwargs).fit()
        self.importances = dict(zip(predictors, model_full.tvalues[1:]))

    def _fit_ols(self, y, X):
        return LinearRegression(**self.kwargs).fit(y=y, X=X)

class RobustLinearModel(SingleViewModel):
    """
    Robust linear regression model using statsmodels.RLM.
    """
    def fit(self, y, X, predictors, k_cv):
        X = sm.add_constant(X)
        self.predictions = self._k_fold_predict(y, X, k_cv, self._fit_robust)
        model_full = sm.RLM(y, X, **self.kwargs).fit()
        self.importances = dict(zip(predictors, model_full.tvalues[1:]))

    def _fit_robust(self, y, X):
        return sm.RLM(y, X, **self.kwargs).fit()

from collections.abc import Callable

import numpy as np
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict


class SingleViewModel:
    """
    Base class for single view models. Subclasses should implement the fit method.

    Parameters
    ----------
    seed
        Pseudo-random number generator (PRNG) state seed.
    kwargs
        Other arguments used in a specific method. See the specific documentation in the corresponding child class.

    Attributes
    ----------
    seed : int
        The assigned initial state for the PRNG.
    kwargs : dict[str, Any]
        Keyword arguments passed to the fit function in a child class.
    model : str
        The model name.
    predictions : np.ndarray
        Contains the resulting predictions in array form.
    importances : dict[str, np.ndarray]
        Contains the importance scores of the different predictors.

    """

    def __init__(self, seed: int, **kwargs):
        self.seed = seed
        self.kwargs = kwargs  # Store kwargs to be used in fit method
        self.model = None
        self.predictions = None
        self.importances = None

    def fit(self,
            y: np.ndarray,
            X: np.ndarray,
            predictors: list[str],
            k_cv: int = None
            ):
        """
        Fit the model to the data and store the predictions and importances.

        Parameters
        ----------
        y
            Target variable
        X
            Feature matrix
        predictors
            List of feature names
        k_cv
            Number of cross-validation folds. If None, no cross-validation is performed.

        Raises
        ------
        NotImplementedError
            Base class method, children classes replace it with their own method.
        """
        raise NotImplementedError("This method should be implemented by subclasses")

    def _k_fold_predict(self,
                        y: np.ndarray,
                        X: np.ndarray,
                        k_cv: int,
                        fit_method: Callable
                        ) -> np.ndarray:
        """
        Computes K-Fold cross-validation (CV)

        Parameters
        ----------
        y
            Target variable
        X
            Feature matrix
        k_cv
            Number of CV steps
        fit_method
            Model function to compute estimates

        Returns
        -------
        Matrix with the prediction results for each round of CV

        """
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
    """Random forest model (from sklearn) using out-of-bag predictions for feature importances. Inherits from `SingleViewModel`"""

    def fit(self,
            y: np.ndarray,
            X: np.ndarray,
            predictors: list[str],
            k_cv: int = None
            ):
        """
        Fits a Random Forest (RF) model.

        Parameters
        ----------
        y
            Target variable
        X
            Feature matrix
        predictors
            List of feature names
        k_cv
            Not used

        """
        self.model = RandomForestRegressor(oob_score=True, random_state=self.seed, **self.kwargs)
        self.model.fit(X, y)  # type: ignore[union-attr, attr-defined]
        self.predictions = self.model.oob_prediction_  # type: ignore[union-attr, attr-defined]
        self.importances = dict(zip(predictors, self.model.feature_importances_, strict=False))  # type: ignore[union-attr, attr-defined, assignment]


class LinearModel(SingleViewModel):
    """Linear regression model using `statsmodels.OLS` for feature importances, and `cross_val_predict` with `sklearn.LinearRegression` for predictions. Inherits from `SingleViewModel`"""

    def fit(self,
            y: np.ndarray,
            X: np.ndarray,
            predictors: list[str],
            k_cv: int = None
            ):
        """
        Fits a Linear Model (LM) model.

        Parameters
        ----------
        y
            Target variable
        X
            Feature matrix
        predictors
            List of feature names
        k_cv
            Number of cross-validation folds. If None, no cross-validation is performed.

        """
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
        self.importances = dict(zip(predictors, model_full.tvalues[1:], strict=False))  # type: ignore[assignment]

    def _fit_ols(self, y, X):
        return LinearRegression(**self.kwargs).fit(y=y, X=X)


class RobustLinearModel(SingleViewModel):
    """Robust linear regression model using `statsmodels.RLM`. Inherits from `SingleViewModel`"""

    def fit(self,
            y: np.ndarray,
            X: np.ndarray,
            predictors: list[str],
            k_cv: int = None
            ):
        """
        Fits a robust linear model.

        Parameters
        ----------
        y
            Target variable
        X
            Feature matrix
        predictors
            List of feature names
        k_cv
            Number of cross-validation folds. If None, no cross-validation is performed.

        """
        X = sm.add_constant(X)
        self.predictions = self._k_fold_predict(y, X, k_cv, self._fit_robust)
        model_full = sm.RLM(y, X, **self.kwargs).fit()
        self.importances = dict(zip(predictors, model_full.tvalues[1:], strict=False))  # type: ignore[assignment]

    def _fit_robust(self, y, X):
        return sm.RLM(y, X, **self.kwargs).fit()

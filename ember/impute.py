from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.base import BaseEstimator, TransformerMixin

class GeneralImputer(BaseEstimator, TransformerMixin):
    """Useful transformer that uses one of sklearn imputers based on provided parameters

    """

    def __init__(self, kind, strategy='mean'):
        """Transformer initializer

        Args:
            kind (str): kind of imputer. Either 'Simple', 'Iterative', 'KNN' or 'Indicate'
            strategy (str, optional): If needed for chosen imputer. Defaults to 'mean'.
        """
        self.kind = kind
        self.strategy = strategy
        self.imputer = None

        if kind == 'Simple':
            self.imputer = SimpleImputer(strategy=strategy)
        elif kind == 'Iterative':
            self.imputer = IterativeImputer()
        elif kind == 'KNN':
            self.imputer = KNNImputer()
        elif kind == 'Indicate':
            self.imputer = SimpleImputer(strategy='constant', fill_value=0, add_indicator=True)


    def fit(self, X, y = None):
        """Fit the imputer on X

        Args:
            X : {array-like, sparse matrix}, shape (n_samples, n_features)
                    Input data, where ``n_samples`` is the number of samples and
                    ``n_features`` is the number of features.

        Returns:
            self : object
        """

        self.imputer.fit(X, y)
        return self

    def transform(self, X, y=None):
        """
            Impute all missing values in X.

           
            Args:
                X : array-like of shape (n_samples, n_features)
                    The input data to complete.

            Returns:
                X : array-like of shape (n_samples, n_output_features)
                    The imputed dataset. `n_output_features` is the number of features
                    that is not always missing during `fit`.
        """
        return self.imputer.transform(X)
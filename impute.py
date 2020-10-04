from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.base import BaseEstimator, TransformerMixin

class GeneralImputer(BaseEstimator, TransformerMixin):

    def __init__(self, kind, strategy='mean'):
        self.kind = kind
        self.strategy = strategy
        self.imputer = None

        if kind == 'Simple':
            self.imputer = SimpleImputer(strategy=strategy)
        elif kind == 'Iterative':
            self.imputer = IterativeImputer()
        elif kind == 'KNNImputer':
            self.imputer = KNNImputer()
        else:
            self.imputer = SimpleImputer(strategy='constant', fill_value=0, add_indicator=True)


    def fit(self, X, y = None):

        self.imputer.fit(X)
        return self

    def transform(self, X, y=None):

        return self.imputer.transform(X)
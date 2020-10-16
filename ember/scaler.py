from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


class GeneralScaler(BaseEstimator, TransformerMixin):
    """Class wrapping up few scalers for easier use 
    """

    def __init__(self, kind):
        """General Scaler Initializer

        Args:
            kind (str): string representing scaler type. Can be one of following ('SS' - StandardScaler, 'MMS' -  MinMaxScaler)

        Raises:
            Exception: Raised if kind not supported.
        """

        if kind == 'SS':
            self.scaler = StandardScaler()
        elif kind == 'MMS':
            self.scaler = MinMaxScaler()
        else:
            raise Exception("No such kind of scaler supported! \n Supported kinds: SS MMS")

    def fit(self, X, y=None):
        """Fit scaler

        Args:
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            The data used to fit scaler

        y
            Ignored
        """

        self.scaler.fit(X)
        return self

    def transform(self, X, y=None):
        """Transform data by fitted scaler

        Args:
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            The data to be transformed

        y
            Ignored
        """

        return self.scaler.transform(X)
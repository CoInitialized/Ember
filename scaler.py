from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


class GeneralScaler(BaseEstimator, TransformerMixin):
    """

        TODO: ADD MORE SCALERS

    """

    def __init__(self, kind):

        if kind == 'SS':
            self.scaler = StandardScaler()
        elif kind == 'MMS':
            self.scaler = MinMaxScaler()
        else:
            raise Exception("No such kind of scaler supported! \n Supported kinds: SS MMS")

    def fit(self, X, y=None):

        self.scaler.fit(X)
        return self

    def transform(self, X, y=None):

        return self.scaler.transform(X)
from sklearn.base import BaseEstimator, TransformerMixin

class Fraction_Selector(BaseEstimator, TransformerMixin):

    @staticmethod
    def select_by_fraction_missing(X, fraction: float, drop : bool = False, ignored_columns = None):

        dataframe = X

        if ignored_columns is None:
            ignored_columns = []

        if fraction > 1 or fraction < 0:
            raise Exception("fraction value has to be between 0 and 1!")

        df_len = len(dataframe)
        columns = list(dataframe.columns)
        to_choose = []
        for column in columns:
            if column not in ignored_columns:
                percent_missing = dataframe[column].isnull().sum() / df_len
                if percent_missing > fraction:
                    to_choose.append(column)

        if drop:
            return dataframe.drop(columns = to_choose), to_choose
        else:
            return dataframe.loc[:, to_choose], to_choose

    def __init__(self, fraction: float, drop : bool = False, ignored_columns : list = []):
        self.fraction = fraction
        self.drop = drop
        self.ignored_columns = ignored_columns

    def fit(self, X, y = None):
        self.frame, self.chosen_columns = self.select_by_fraction_missing(self.fraction, self.drop, self.ignored_columns)
        return self

    def trasform(self, X, y = None):
        return self.frame


class DtypeSelector(BaseEstimator, TransformerMixin):

    def __init__(self, dtype):
        self.dtype = dtype

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.select_dtypes(include=self.dtype)
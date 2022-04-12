from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd



class Fraction_Selector(BaseEstimator, TransformerMixin):
    """ Useful transformer selecting and returning only columns having less than specified percentage of missing values

    """

    @staticmethod
    def select_by_fraction_missing(X, fraction: float, inplace : bool = False, ignored_columns = None):
        print(type(X))
        if not isinstance(X, np.ndarray) and not isinstance(X, pd.DataFrame):
            raise Exception("Not valid type of X")

        if isinstance(X, np.ndarray):
            dataframe = pd.DataFrame(X)
        else:
            dataframe = X

        if ignored_columns is None:
            ignored_columns = []

        if fraction > 1 or fraction < 0:
            raise Exception("fraction value has to be between 0 and 1!")

        df_len = len(dataframe)
        columns = list(dataframe.columns)
        to_choose = []
        for column in columns:
            print(column)
            if column not in ignored_columns:
                percent_missing = dataframe[column].isnull().sum() / df_len
                print(f"Missing {percent_missing}")
                if percent_missing < fraction:
                    print("Selecting")
                    to_choose.append(column)
                else:
                    print("Ignoring")
            else:
                print("ignoring")

        if inplace:
            return dataframe.drop(columns = to_choose), to_choose
        else:
            return dataframe.loc[:, to_choose], to_choose

    def __init__(self, fraction: float, inplace : bool = False, ignored_columns : list = []):
        """Transformer initializer

        Args:
            fraction (float): Percentage of missing values to be considered as threshold to drop column
            inplace (bool, optional): Whether to modify provided array or return modified copy. Defaults to False.
            ignored_columns (list, optional): If data is provided as dataframe this argument can be used to specify which columns to ignore checking. Defaults to [].
        """
        self.fraction = fraction
        self.inplace = inplace
        self.ignored_columns = ignored_columns

    def fit(self, X, y = None):
        """Fits transformer, columns to be droped are saved into memory

        Args:
            X (numpy.ndarray or pandas.DataFrame): The data to be transformed
            y:
                Ignored


        """
        self.frame, self.chosen_columns = self.select_by_fraction_missing(X, self.fraction, self.inplace, self.ignored_columns)
        return self

    def transform(self, X, y = None):
        """Return data with only < fraction missing values in column.

        Args:
            X (numpy.ndarray or pandas.DataFrame): The data to be transformed
            y:
                Ignored



        Returns:
            (numpy.ndarray or pandas.DataFrame): Transformed data
        """
        return X.loc[:, self.chosen_columns]


class DtypeSelector(BaseEstimator, TransformerMixin):
    """Useful transformer selecting and returning only columns of specified datatype

    """

    def __init__(self, dtype):
        """Transformer initializer

        Args:
            dtype (type): datatype to be selected
        """ 
        self.dtype = dtype

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """Selects and returns columns of selected datatype
       
        Args:
        X : {array-like}, shape [n_samples, n_features]
            The data to be transformed

        y
            Ignored

        Returns:
            numpy.ndarray: array of selected data
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        return X.select_dtypes(include=self.dtype)


class NameFixer:
    """Fixed column names of dataframe that can cause problem in later processing by other libraries
    """
    def __init__(self):
        pass

    @staticmethod
    def fix_forbidden_name(x):
        """Performs regex substitution on string deleting forbidden symbols

        Args:
            x (str): string to be fixed

        Returns:
            str: fixed string
        """
        error_regex = re.compile(r"([<>\[\]])")
        result =  re.sub(error_regex, '', x)
        if result == x:
            return None
        else:
            return result

    @classmethod
    def fix(cls, frame):
        """Fix column names for whole dataframe

        Args:
            frame (pandas.DataFrame): dataframe to be transformed

        Raises:
            Exception: frame is not a pandas DataFrame

        Returns:
            pandas.DataFrame: fixed dataframe
        """
        
        fixed_columns = {}

        if isinstance(frame, pd.DataFrame):

            for i, column in enumerate(frame.columns):
                fixed = cls.fix_forbidden_name(column)
                if fixed:
                    fixed_columns[column] = fixed
        else:
            raise Exception("frame argument should be pandas DataFrame")

        return frame.rename(columns = fixed_columns)

            

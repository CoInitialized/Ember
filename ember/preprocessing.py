from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
import numpy as np
import pandas as pd
from category_encoders import TargetEncoder, LeaveOneOutEncoder, WOEEncoder

class Preprocessor:
    """Class providing way to build multibranch pipelines
    """


    def __init__(self):
        self.branches = {}  
        self.pipes = {}
        self.union = None
        self.pipeline = None


    def add_branch(self, name):
        """Add new branch to pipeline with selected name

        Args:
            name (str): name of new branch
        """
        self.branches[name] = []

    def add_transformer_to_branch(self, name, transformer):
        """Add new transformer to branch of chosen name. If branch doesn't exist, create it and put transformer in it.

        Args:
            name (str): name of selected branch
            transformer (Any class deriving from (BaseEstimator, TransformerMixin)): chosen estimator
        """
        if name in list(self.branches.keys()):
            self.branches[name].append(transformer)
        else:
            self.branches[name] = [transformer]

    def merge(self):
        """Merge branches and return ready-to-fitpipeline

        Returns:
            FeatureUnion: final pipeline
        """
        for branch in self.branches.keys():
            self.pipes[branch] = make_pipeline(*self.branches[branch])
        self.union = FeatureUnion([(pipe_name, self.pipes[pipe_name]) for pipe_name in self.pipes.keys()])

        return self.union


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
        
        Returns:
            Transformed data
        """

        return self.scaler.transform(X)

class MultiColumnTransformer(BaseEstimator, TransformerMixin):

    """
        Transformer generalization for many columns. Used to fit the same transformer for each individual column
    """

    def __init__(self, transformer, match_col_names = False, **kwargs):
        """Transformer initializer

        Args:
            transformer (class): Transformed to be used
            match_col_names (bool, optional): If fitting on dataframe this argument ensures that also column names match, not only indices. Defaults to False.
        """
        self.transformer = transformer
        self.kwargs = kwargs
        self.match_col_names = match_col_names
        self.transformers = []
        self.columns = []

    def __getitem__(self, index):
        return self.transformers[index]

    def fit(self, X, y = None):
        """Fit transformer on given data

        Args:
            X (pandas.Dataframe or numpy.array, shape [n_samples, n_features]): The data used to fit transformer. 
            y:
                Ignored

        Raises:
            Exception: Unsupported data structure

        Returns:
            [type]: [description]
        """
  
        if isinstance(X, pd.DataFrame):
            self.columns =  list(X.columns)
        elif isinstance(X, np.ndarray):
            if len(X.shape) == 2:
                self.columns = list(range(X.shape[1]))
                X = pd.DataFrame(X)
            else:
                raise Exception("Unsupported data structure. Provided structure should be two-dimentional. In case if your data is single column reshape it to (-1,1)")

        else:
            raise Exception("Unsupported data structure. Use pandas dataframe or 2-dim numpy array instead")

        self.transformers = [self.transformer(**self.kwargs) for i in range(len(self.columns))]


        for i in range(len(self.columns)):
            self.transformers[i].fit(X.iloc[:, i].astype(str))


        return self
    
    def transform(self, X, y = None):
        """Transform each column with corresponding transformer

        Args:
            X (pandas.Dataframe or numpy.array, shape [n_samples, n_features]): The data to be transformed matching fitted structure. 
            y:
                Ignored

        Raises:
            Exception: Unsupported data structure
            Exception: Enexpected shape of return from one of transformers

        Returns:
            Transformed data
        """

        if isinstance(X, pd.DataFrame):
            if self.match_col_names and self.columns != list(X.columns):
                raise Exception("Invalid column names structure")
            X = X.to_numpy().reshape(-1,len(self.columns))
        elif isinstance(X, np.ndarray):
            if len(X.shape) == 2:
                if len(self.columns) != X.shape[1]:
                    raise Exception("Number of columns don't match with fit")
            else:
                raise Exception("Unsupported data structure. Provided structure should be two-dimentional. In case if your data is single column reshape it to (-1,1)")

        else:
            raise Exception("Unsupported data structure")

        final = None

        for i in range(len(self.columns)):
            result = self.transformers[i].transform(X[:, i])
            if not isinstance(result, np.ndarray):
                raise Exception("One of transformers returned {} instead of {}".format(type(result), np.ndarray))
            if len(result.shape) > 2:
                raise Exception("One of transformers returned over 2-dim ndarray ")
            if len(result.shape) == 1:
                result = result.reshape(-1,1)
            if not isinstance(final, np.ndarray):
                final = result
            else:
                final = np.concatenate((final, result), axis = 1)    


        return final




class GeneralEncoder(BaseEstimator, TransformerMixin):
    """Class wrapping up few encoders for easier use 
    """



    def __init__(self, kind, **kwargs):
        """Encoder initializer

        Args:
            kind (str): Either 'OHE' (One hot encoder), 'TE' (Target encoder),'LOOE' (Leave one out encoder),'WOE' (Weigth of evidence encoder) or 'LE' (Label encoder)

        Raises:
            Exception: Encoder type not supported
        """
        self.kind = kind
        if kind not in ['OHE','TE','LOOE','WOE', 'LE']:
            raise Exception("Encoder type not supported, choose one of ('OHE','TE','LOOE','WOE', 'LE')")
        else:
            if kind == 'OHE':
                self.encoder = OneHotEncoder(**kwargs)
            elif kind == 'TE':
                self.encoder = TargetEncoder(**kwargs)
            elif kind == 'LOOE':
                self.encoder = LeaveOneOutEncoder(**kwargs)
            elif kind == 'WOE':
                self.encoder = WOEEncoder(**kwargs)
            elif kind == 'LE':
                self.encoder = MultiColumnTransformer(LabelEncoder)

    def fit(self, X, y = None):
        """Fit encoder with data X

        Args:
            X : {array-like, sparse matrix} of shape (n_samples, n_features)
            y :
                Ignored

        Returns:
            self : object
        """
        if isinstance(self.encoder, OneHotEncoder):
            self.encoder.fit(X)
        else:
            self.encoder.fit(X,y)

        return self

    def transform(self, X, y = None):
        """Transforms data

        Args:
            X : {array-like, sparse matrix} of shape (n_samples, n_features)
            y :
                Ignored

        Returns:
            Transformed data
        """
        return self.encoder.transform(X)




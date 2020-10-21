from preprocessing import Preprocessor, GeneralEncoder, GeneralScaler
import pandas as pd
import numpy as np
from impute import GeneralImputer
from optimize import GridSelector, BayesSelector
from sklearn.pipeline import make_pipeline
from selector import DtypeSelector


class Learner:
    """Automatic learning model performing all tasks including preprocessing and model selection
    """
    def __init__(self, objective, frame = None, target = None, X = None, y = None):
        """Model initializer

        Args:
            objective (str): either 'classification' or 'regression'
            frame (pandas.DataFrame, optional): Dataframe with data. If not provided X has to be provided. Defaults to None.
            target (pandas.Series, optional): Series with labels. If not provided y has to be provided. Defaults to None.
            X (numpy.ndarray, optional), shape [n_samples, n_features]: Array with features. If not provided gframe has to be provided. Defaults to None.
            y (numpy.ndarray, optional), shape [n_samples,]: Array with labels. If not provided target has to be provided. Defaults to None.

        Raises:
            Exception: Unsupported objective    
            Exception: No valid pair of arguments provided
        """
        if objective == 'regression' or objective == 'classification':
            self.objective = objective
        else:
            raise Exception("Unsupported objective, choose classification or regression instead")

        if ((X is not None or y is not None) and (frame is not None or target is not None)):
            raise Exception("You can choose either frame and target column or X and y")


        if X is not None and y is not None:
            self.X = pd.DataFrame(X)
            self.y = pd.Series(y)

        elif frame is not None and target is not None:
            self.X = frame.drop(columns = [target])
            self.y = frame['target']
        else:
            raise Exception("Not enough data provided, provide frame and target or X and y")


        self.model = None

        #### PREPARING PIPELINES ####

        ## target pipeline ##

        target_preprocessor = Preprocessor()
        target_preprocessor.add_branch('target')


        if self.y.dtype == np.object:

            target_preprocessor.add_transformer_to_branch('target', GeneralImputer('simple', 'most_frequent'))
            target_preprocessor.add_transformer_to_branch('target', GeneralEncoder('LE'))
        else:
            if self.objective == 'classification':
                target_preprocessor.add_transformer_to_branch('target', GeneralImputer('simple', 'most_frequent'))
            elif self.objective == 'regression':
                target_preprocessor.add_transformer_to_branch('target', GeneralImputer('simple', 'mean'))
            else:
                pass
                
        ## features pipeline ##

        feature_preprocessor = Preprocessor()

        is_number = np.vectorize(lambda x: np.issubdtype(x, np.number))
        is_object = np.vectorize(lambda x: np.issubdtype(x, np.object))

        if is_object(X.dtypes).any():
            feature_preprocessor.add_branch("categorical")
            feature_preprocessor.add_transformer_to_branch("categorical", DtypeSelector(np.object))
            feature_preprocessor.add_transformer_to_branch("categorical", GeneralImputer('Simple', strategy='most_frequent'))
            feature_preprocessor.add_transformer_to_branch("categorical", GeneralEncoder(kind = 'TE'))


        if is_number(X.dtypes).any():
            feature_preprocessor.add_branch('numerical')
            feature_preprocessor.add_transformer_to_branch("numerical", DtypeSelector(np.number))
            feature_preprocessor.add_transformer_to_branch("numerical", GeneralImputer('Simple'))
            feature_preprocessor.add_transformer_to_branch("numerical", GeneralScaler('SS'))

        
        self.feature_preprocessor = feature_preprocessor.merge()
        self.target_preprocessor = target_preprocessor.merge()

    def fit(self, speed = 'fast', optimizer = 'grid', X_test = None, y_test = None, cv = 5):
        model = None
        if optimizer == 'bayes':
            if speed == 'fast':
                model = BayesSelector(self.objective, 10, X_test, y_test, cv)
            elif speed == 'medium':
                model = BayesSelector(self.objective, 35, X_test, y_test, cv)
            elif speed == 'slow':
                model = BayesSelector(self.objective, 100, X_test, y_test, cv)
            elif isinstance(speed, int):
                model = BayesSelector(self.objective, speed, X_test, y_test, cv)
            else:
                raise Exception("Speed specifier not supported")
        elif optimizer == 'grid':
            if speed == 'fast':
                model = GridSelector(self.objective, 2, folds = cv)
            elif speed == 'medium':
                model = GridSelector(self.objective, 4, folds = cv)
            elif speed == 'slow':
                model = GridSelector(self.objective, 6, folds = cv)
            elif isinstance(speed, int):
                model = GridSelector(self.objective, speed, folds = cv)
            else:
                raise Exception("Speed specifier not supported")
        else:
            raise Exception("Optimizer not supported")
        
        y = np.array(self.y).reshape(-1,1)
        y = self.target_preprocessor.fit_transform(y).ravel()

        self.model = make_pipeline(self.feature_preprocessor, model)

        self.model.fit(self.X,y)

        return self

    def predict(self, X):

        return self.model.predict(X)






    
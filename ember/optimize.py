import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from hyperopt import hp, tpe, Trials, STATUS_OK, fmin
from .search_space import grid_params, get_bayes_params
from sklearn.metrics import accuracy_score, r2_score
from functools import partial
import copy
from sklearn.model_selection import KFold
import catboost

def fix_hyperparams(params):
    """Helper function to be removed and solved with better search spaces definitions
    """
    inable_table = ['n_estimators','max_depth','num_leaves']
    for x in inable_table:
        if x in params.keys():
            params[x] = int(params[x])

    outable = ['model_type', 'model','name']
    for x in outable:
        if x in params.keys():
            del params[x] 

    return params


class Selector:
    """Base class for optimizer
    """

    def __init__(self, objective):
        """Optimizer initializer

        Args:
            objective (str): Either 'classification' or 'regression'
        """
        self.objective = objective
        self.best_model = None

    def predict(self, X_test):
        """Perform prediction on data X_test

        Args:
            X_test : {array-like, sparse matrix} of shape (n_samples, n_features)

        Returns:
            [type]: [description]
        """
        return self.best_model.predict(X_test)

    def score(self, X, y):
        """[summary]

        Args:
            X : {array-like, sparse matrix} of shape (n_samples, n_features)
            y ([type]): [description]

        Raises:
            Exception: [description]

        Returns:
            [type]: [description]
        """
        if self.best_model:
            return self.best_model.score(X,y)
        else:
            raise Exception("No fitted model available")


class GridSelector(Selector):
    def __init__(self,objective, steps = 6, folds = 5, scoring = 'auto', n_jobs = -1):

        super().__init__(objective)
        self.figures = {}
        self.params = grid_params
        self.steps = steps
        self.folds = folds

        self.n_jobs = n_jobs

        self.models = {}
        self.best_model = None
        if objective == 'classification':
            self.scoring = 'accuracy' if scoring == 'auto' else scoring
            self.models = {
                "XGB": XGBClassifier,
                "LGBM": LGBMClassifier,
                "CAT": CatBoostClassifier
            }
           
        elif objective == 'regression':
            self.scoring = 'r2' if scoring == 'auto' else scoring
            self.models = {
                "XGB": XGBRegressor,
                "LGBM": LGBMRegressor,
                "CAT": CatBoostRegressor

            }
        else: 
            raise Exception("Unknown objective, choose classification or regression")


    def make_plot(self, scores, params, output_file = None, show = False):
        fig, axes = plt.subplots()
        args = list(params.keys())
        values = list(params.values())
        axes.set_ylabel('score')
        if len(args) == 1:
            arg_name = args[0]
            arg_val = values[0]       
            scores = np.array(scores) 
            axes.plot(arg_val, scores)
            axes.set_xlabel(arg_name)
               
        elif len(args) == 2:
            arg1_name, arg2_name = args[0], args[1]
            arg1_val, arg2_val = values[0], values[1]
            scores = np.array(scores).reshape(len(arg1_val), len(arg2_val))
            for i, value in enumerate(arg1_val):
                axes.plot(arg2_val, scores[i], label = f'{arg1_name}: {value}')

            axes.set_xlabel(arg2_name)

            axes.legend()

        else:
            raise Exception('Ni mo supportu na 3d graphy')

        if output_file:
            fig.savefig(output_file)
        if show:
            fig;
            plt.show()

        return fig, axes

    def fit(self,X, y):
        """[summary]

        Args:
            X : {array-like, sparse matrix} of shape (n_samples, n_features)
            y ([type]): [description]

        Returns:
            [type]: [description]
        """

        best_score = None
        gcv = 1

        for key in self.params.keys():
            learned_params = {}
            if key == 'CAT':
                learned_params['logging_level'] = 'Silent'
            print(f"Searching model for {key}")
            for i in range(min([self.steps,len(self.params[key])])):
                print(f'step {i}')
                del gcv
                gcv = GridSearchCV(estimator = self.models[key](**learned_params), param_grid = self.params[key][i], scoring=self.scoring, cv=self.folds, verbose = 0, n_jobs =self.n_jobs, refit= False if i < (self.steps - 1) else True)
                print('fitowanie')
                gcv.fit(X,y)
                print('updatowanie')
                learned_params.update(gcv.best_params_)
                print('plotowanie')
                fig, axes = self.make_plot(scores = gcv.cv_results_['mean_test_score'], params = self.params[key][i],show=False)
                name = "_".join([x for x in self.params[key][i].keys()])
                print(name)
                self.figures[name] = fig
            try:
                #może nie być najlepszego modelu XD
                self.models[key] = gcv.best_estimator_
                if best_score:
                    score = gcv.best_estimator_.score(X,y)
                    if score > best_score:
                        best_score = score
                        self.best_model = gcv.best_estimator_
                else:
                    best_score = gcv.best_estimator_.score(X,y)
                    self.best_model = gcv.best_estimator_ 
            except:
                pass

            del learned_params
            
        return self

class BayesSelector(Selector):
    def __init__(self, objective, max_evals, X_test = None, y_test = None, cv = None, **kwargs):
        """[summary]

        Args:
            objective (str): Either 'classification' or 'regression'
            max_evals ([type]): Number of optimization steps
            X_test : {array-like, sparse matrix} of shape (n_samples, n_features). Optional. Defaults to None.
            y_test : array-like of shape (n_samples,), Target values 
            cv (int, optional): Number of cross validation folds. Defaults to None.

        Raises:
            Exception: [description]
            Exception: [description]
        """
        super().__init__(objective)


        self.objective = objective  
        self.max_evals = max_evals
        self.kwargs = kwargs
        self.X_test = X_test
        self.y_test = y_test
        self.cv = cv

        if (X_test is not None) ^ (y_test is not None):
            raise Exception("You have to provide both X_test and y_test not only one of them!")
        

        if objective == 'classification':
            self.scoring = accuracy_score
            self.loss = lambda y_true, y_pred: 1 - self.scoring(y_true, y_pred)
            self.models = {
                "XGB": XGBClassifier,
                "LGBM": LGBMClassifier,
                "CAT": CatBoostClassifier
            }
           
        elif objective == 'regression':
            self.scoring = r2_score
            self.loss = lambda y_true, y_pred, : (-1 * self.scoring(y_true, y_pred))
            self.models = {
                "XGB": XGBRegressor,
                "LGBM": LGBMRegressor,
                "CAT": CatBoostRegressor

            }
        else: 
            raise Exception("Unknown objective, choose classification or regression")


    def objective_function(self, space):
        """Function to be optimized
        """


        model = self.models[space['name']]
        name = space['name']
        space = fix_hyperparams(space)

        ### TEST IF IT ACTUALLY WORKS
        if name == 'CAT':
            _model = model(logging_level = 'Silent', **space)
        else:
            _model = model(**space)

        loss = None

        if self.cv:
            losses = []
            kf = KFold(n_splits=self.cv)
            for train_index, test_index in kf.split(self.X_train):
                X_train, X_test = self.X_train[train_index], self.X_train[test_index]
                y_train, y_test = self.y_train[train_index], self.y_train[test_index]
                model = copy.deepcopy(_model)   
                model.fit(X_train, y_train)
                losses.append(self.loss(y_test, model.predict(X_test)))
                del model
            loss = sum(losses) / len(losses)
            _model.fit(self.X_train, self.y_train)
            del losses
        
        elif self.X_test is not None and self.y_test is not None:
            _model.fit(self.X_train, self.y_train)
            loss = self.loss(self.y_test, _model.predict(self.X_test))
        
        else:
            _model.fit(self.X_train, self.y_train)
            loss = self.loss(self.y_train, _model.predict(self.X_train))
        

        train_score = self.scoring(self.y_train, _model.predict(self.X_train))


        return {'status': STATUS_OK, 'loss': loss,
                'train score': train_score
            }

    

    def org_results(self,trials, hyperparams):

        fit_idx = -1
        for idx, fit  in enumerate(trials):
            hyp = fit['misc']['vals']
            xgb_hyp = {key:[val] for key, val in hyperparams.items()}
            if hyp == xgb_hyp:
                fit_idx = idx
                break
                
        train_time = str(trials[-1]['refresh_time'] - trials[0]['book_time'])
        train_score = round(trials[fit_idx]['result']['train score'], 3)

        results = {
            'parameter search time': train_time,
            'training score': train_score,
            'parameters': hyperparams
        }
        return results

    def fit(self, X, y):
        """Fit the and optimize model according to the given training data.

        Args:
            X : {array-like, sparse matrix} of shape (n_samples, n_features)
            y : array-like of shape (n_samples,), Target values 

        Returns:
            self: object
        """

        self.X_train = X
        self.y_train = y
        trials = Trials()

        hyperparams = fmin(fn = self.objective_function, 
                            max_evals = self.max_evals, 
                            trials = trials,
                            algo = tpe.suggest,
                            space = get_bayes_params(**self.kwargs)
                            )

        results = self.org_results(trials.trials, hyperparams)

        name = list(hyperparams.keys())[0].split('_')[-1].upper()
        for key in list(hyperparams.keys()):
            hyperparams['_'.join(key.split('_')[:-1])] = hyperparams.pop(key)

        hyperparams = fix_hyperparams(hyperparams)

        self.best_model = self.models[name](**hyperparams) 
        self.best_model.fit(X, y)   
        return self

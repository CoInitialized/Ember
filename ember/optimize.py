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
from sklearn.model_selection import KFold, StratifiedKFold
import catboost
from skopt.plots import plot_convergence
from skopt.callbacks import DeltaYStopper
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize
from functools import partial
from .search_space import get_baesian_space
import tqdm 

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
    """
        Base class for optimizer
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
            array-like of shape = [n_samples] or shape = [n_samples, n_classes: The predicted values
        """
        return self.best_model.predict(X_test)

    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels in case of classification
         or the coefficient of determination R^2 of the prediction in case of regression.
        Args:
            X : {array-like, sparse matrix} of shape (n_samples, n_features)
            y array-like of shape (n_samples,) or (n_samples, n_outputs): True values for X.

        Raises:
            Exception: Exception raised when no best model has been created, check if you have used fit function 

        Returns:
            float: score of the model
        """
        if self.best_model:
            return self.best_model.score(X,y)
        else:
            raise Exception("No fitted model available")


class GridSelector(Selector):
    def __init__(self,objective, steps = 6, folds = 5, scoring = 'auto', n_jobs = -1):
        """ Class performing CV search in order to find best model

        Args:
            objective string: either regression or classification, will decide which version of alghoritm to use
            steps (int, optional): Determines how many parameters will be searched for a given model. Defaults to 6.
            folds (int, optional): Determines how many folds to use during the process of cross-validation. Defaults to 5.
            scoring (str, optional): Determines which kind of scoring to use. Defaults to 'auto'.
            n_jobs (int, optional): Determines how many threads will be used during search. Defaults to -1.

        Raises:
            Exception: Exception thrown if unknown objective will be used
        """
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

    def fit(self,X, y, plot = False):
        """ Build a model from the training set (X, y).

        Args:
            X : {array-like, sparse matrix} of shape (n_samples, n_features)
            y array-like of shape = [n_samples]: The target values (class labels in classification, real numbers in regression).

        Returns:
            object: Returns self.
        """

        best_score = None
        gcv = 1

        for key in self.params.keys():
            learned_params = {}
            if key == 'CAT':
                learned_params['logging_level'] = 'Silent'
            elif key == 'XGB':
                learned_params['verbosity'] = 0
                # pass
            print(f"Searching model for {key}")
            for i in range(min([self.steps,len(self.params[key])])):
                print(f'step {i}')
                del gcv
                refit_bool = False if i < (min([self.steps,len(self.params[key])-1]) - 1) else True
                np.random.seed(200)
                gcv = GridSearchCV(estimator = self.models[key](**learned_params), param_grid = self.params[key][i], scoring=self.scoring, cv=self.folds, verbose = 0, n_jobs =self.n_jobs, refit= refit_bool)
                print('fitowanie')
                gcv.fit(X,y)
                print('updatowanie')
                learned_params.update(gcv.best_params_)
                print('plotowanie')
                if plot:
                    fig, axes = self.make_plot(scores = gcv.cv_results_['mean_test_score'], params = self.params[key][i],show=False)
                    self.figures[name] = fig
                
                name = "_".join([x for x in self.params[key][i].keys()])
                print(name)
            

            best_estimator_for_key = None
            best_score_for_key = 0

            try:
                best_estimator_for_key = gcv.best_estimator_
                best_score_for_key = gcv.best_score_
                if best_score:
                    if best_score_for_key > best_score:
                        best_score = best_score_for_key
                        self.best_model = gcv.best_estimator_
                else:
                    best_score = best_score_for_key
                    self.best_model = gcv.best_estimator_ 
            except:
                pass

            del learned_params
            
            np.random.seed(200)
            if self.objective == 'classification':
                kf = StratifiedKFold(n_splits=self.folds)
                split = kf.split(X, y)
            else:
                kf = KFold(n_splits=self.folds)
                split = kf.split(X)

            
            scores = []
            for train_index, test_index in split:
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                if key == 'CAT':
                    base_model = self.models[key](logging_level = 'Silent')
                elif key == 'XGB':
                    base_model = self.models[key](verbosity = 0)
                else:
                    base_model = self.models[key]()
                base_model.fit(X_train,y_train)
                score = base_model.score(X_test, y_test)
                scores.append(score)
                del base_model
            if np.mean(scores) > best_score_for_key:
                best_score_for_key = np.mean(scores)
                if key == 'CAT':
                    model = self.models[key](logging_level = 'Silent')
                elif key == 'XGB':
                    model = self.models[key](verbosity = 0)
                else:
                    model = self.models[key]()
                model.fit(X,y)
                self.best_model = model
                best_estimator_for_key = model
            if best_score_for_key > best_score:
                best_score = best_score_for_key


            self.models[key] = copy.deepcopy(best_estimator_for_key)
            del best_estimator_for_key

        return self

class BayesSelector(Selector):
    def __init__(self, objective, max_evals, X_test = None, y_test = None, cv = None, **kwargs):
        """Class performing bayesian search based on hyperopt in order to find best model

        Args:
            objective (str): Either 'classification' or 'regression'
            max_evals ([type]): Number of optimization steps
            X_test : {array-like, sparse matrix} of shape (n_samples, n_features). Optional. Defaults to None.
            y_test : array-like of shape (n_samples,), Target values 
            cv (int, optional): Number of cross validation folds. Defaults to None.

        Raises:
            Exception: Exception thrown when only one of X_test and y_test is provided
            Exception: Exception thrown if unknown objective will be used
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
            self.loss = lambda y_true, y_pred, : 1 + (-1 * self.scoring(y_true, y_pred))
            self.models = {
                "XGB": XGBRegressor,
                "LGBM": LGBMRegressor,
                "CAT": CatBoostRegressor

            }
        else: 
            raise Exception("Unknown objective, choose classification or regression")


    def objective_function(self, space, silent = True):
        """Function to be optimized
        """


        model = self.models[space['name']]
        name = space['name']
        space = fix_hyperparams(space)

        ### TEST IF IT ACTUALLY WORKS
        if name == 'CAT' and silent:
            _model = model(logging_level = 'Silent', **space)
        elif name == 'XGB' and silent:
            _model = model(verbosity = 0, **space)
        else:
            _model = model(**space)

        loss = None

        if self.cv:
            losses = []
            np.random.seed(200)

            if self.objective == 'classification':
                kf = StratifiedKFold(n_splits=self.cv)
                split = kf.split(self.X_train, self.y_train)
            else:
                kf = KFold(n_splits=self.cv)
                split = kf.split(self.X_train)
                
            for train_index, test_index in split:
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

        self.best_score = results['training score']

        name = list(hyperparams.keys())[0].split('_')[-1].upper()
        for key in list(hyperparams.keys()):
            hyperparams['_'.join(key.split('_')[:-1])] = hyperparams.pop(key)

        hyperparams = fix_hyperparams(hyperparams)
        self.best_model = self.models[name](**hyperparams)
        np.random.seed(200)
        if self.cv:
            if self.objective == 'classification':
                kf = StratifiedKFold(n_splits=self.cv)
                split = kf.split(X, y)
            else:
                kf = KFold(n_splits=self.cv)
                split = kf.split(X)
            self.best_score = None
            for model in ([self.best_model] + [model() for model in  list(self.models.values())]):
                scores = []
                for train_index, test_index in split:
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    base_model = copy.deepcopy(model)
                    base_model.fit(X_train, y_train)
                    score = base_model.score(X_test, y_test)
                    scores.append(score)
                    del base_model
                if self.best_score == None or np.mean(scores) > self.best_score:
                    self.best_score = np.mean(scores)
                    self.best_model = copy.deepcopy(model)

        elif self.X_test is not None and self.y_test is not None:
            for model in ([self.best_model] + [model() for model in  list(self.models.values())]):
                base_model = copy.deepcopy(model)
                base_model.fit(self.X_train, self.y_train)
                score = self.scoring(self.y_test, base_model.predict(self.X_test))
                if self.best_score == None or score > self.best_score:
                    self.best_score = score
                    self.best_model = copy.deepcopy(model)
        else:
            pass

        self.best_model.fit(X, y)
        return self

class BaesianSklearnSelector(Selector):
    def __init__(self, objective, max_evals, X_test = None, y_test = None, cv = None, **kwargs):
        """Class performing bayesian search based on skopt in order to find best model

        Args:
            objective (str): Either 'classification' or 'regression'
            max_evals ([type]): Number of optimization steps
            X_test : {array-like, sparse matrix} of shape (n_samples, n_features). Optional. Defaults to None.
            y_test : array-like of shape (n_samples,), Target values 
            cv (int, optional): Number of cross validation folds. Defaults to None.

        Raises:
            Exception: Exception thrown when only one of X_test and y_test is provided
            Exception: Exception thrown if unknown objective will be used
        """
        
        super().__init__(objective)
        self.objective = objective  
        self.max_evals = max_evals
        self.model = None
        self.kwargs = kwargs
        self.X_test = X_test
        self.y_test = y_test
        self.cv = cv
        self.space = None
        self.best_loss = None
        self.best_score = None
        if (self.X_test is not None) ^ (self.y_test is not None):
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
            self.loss = lambda y_true, y_pred, : 1 + (-1 * self.scoring(y_true, y_pred))
            self.models = {
                "XGB": XGBRegressor,
                "LGBM": LGBMRegressor,
                "CAT": CatBoostRegressor

            }
        else: 
            raise Exception("Unknown objective, choose classification or regression")


    def objectivefunc(self,model_key,names,listofparams, silent = True):
        """Function to be optimized
        """
        ### TEST IF IT ACTUALLY WORKS
        params = {}
        for name,param in zip(names,listofparams):
            params[name] = param
        if model_key == 'CAT' and silent:
            params['logging_level'] = 'Silent'
        elif model_key == 'XGB' and silent:
            params['verbosity'] = 0
        _model = self.model(**params)

        loss = None

        if self.cv:
            losses = []
            np.random.seed(200)

            if self.objective == 'classification':
                kf = StratifiedKFold(n_splits=self.cv)
                split = kf.split(self.X_train, self.y_train)
            else:
                kf = KFold(n_splits=self.cv)
                split = kf.split(self.X_train)
                
            for train_index, test_index in tqdm.tqdm(split):
                X_train, X_test = self.X_train[train_index], self.X_train[test_index]
                y_train, y_test = self.y_train[train_index], self.y_train[test_index]
                model = copy.deepcopy(_model)   
                model.fit(X_train, y_train)
                losses.append(self.loss(y_test, model.predict(X_test)))
                del model
            loss = sum(losses) / len(losses)
            _model.fit(self.X_train, self.y_train)
            score = (1-loss)
            del losses
        
        elif self.X_test is not None and self.y_test is not None:
            _model.fit(self.X_train, self.y_train)
            loss = self.loss(self.y_test, _model.predict(self.X_test))
            score = _model.score(self.X_test, self.y_test)
        
        else:
            _model.fit(self.X_train, self.y_train)
            loss = self.loss(self.y_train, _model.predict(self.X_train))
            score = _model.score(self.X_test, self.y_test)
        
        if self.best_loss == None or loss < self.best_loss:
            self.best_model = copy.deepcopy(_model)
            self.best_loss = loss
            self.best_score = score
        del _model
        return loss

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
        results = []
        for key in list(self.models.keys()):
            print(key)
            self.model = self.models[key]
            names = [x.name for x in get_baesian_space()[key]]
            res_gp = gp_minimize(
                                    partial(self.objectivefunc,key,names),
                                    get_baesian_space()[key],
                                    n_calls=self.max_evals,
                                    random_state=0,
                                    callback=DeltaYStopper(0.001)
                                )       
            results.append((key,res_gp))
        fig, ax = plt.subplots()
        plot= plot_convergence(*results,ax=ax);
        np.random.seed(200)
        self.best_score = None
        if self.cv:
            if self.objective == 'classification':
                kf = StratifiedKFold(n_splits=self.cv)
                split = kf.split(X, y)
            else:
                kf = KFold(n_splits=self.cv)
                split = kf.split(X)
            self.best_score = None
            for model in ([self.best_model] + [model() for model in  list(self.models.values())]):
                scores = []
                for train_index, test_index in split:
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    base_model = copy.deepcopy(model)
                    base_model.fit(X_train, y_train)
                    score = base_model.score(X_test, y_test)
                    scores.append(score)
                    del base_model
                if self.best_score == None or np.mean(scores) > self.best_score:
                    self.best_score = np.mean(scores)
                    self.best_model = copy.deepcopy(model)

        elif self.X_test is not None and self.y_test is not None:
            for model in ([self.best_model] + [model() for model in  list(self.models.values())]):
                base_model = copy.deepcopy(model)
                base_model.fit(self.X_train, self.y_train)
                score = self.scoring(self.y_test, base_model.predict(self.X_test))
                if self.best_score == None or score > self.best_score:
                    self.best_score = score
                    self.best_model = copy.deepcopy(model)
        else:
            pass
        self.best_model.fit(X, y)
        return fig,results,self.best_model



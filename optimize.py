import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from hyperopt import hp, tpe, Trials, STATUS_OK, fmin
from search_space import grid_params, bayes_params
from sklearn.metrics import accuracy_score, r2_score
from functools import partial

class Selector:

    def __init__(self, objective):
        self.objective = objective
        self.best_model = None

    def predict(self, X_test):
        self.best_model.predict(X_test)

    def score(self, X, y):
        if self.best_model:
            return self.best_model.score(X,y)
        else:
            raise Exception("No fitted model available")


class GridSelector(Selector):
    def __init__(self,objective):
        super().__init__(objective)
        self.figures = {}
        self.params = grid_params


        self.models = {}
        self.best_model = None
        if objective == 'classification':
            self.scoring = 'accuracy'
            self.models = {
                "XGB": XGBClassifier,
                "LGBM": LGBMClassifier,
                "CAT": CatBoostClassifier
            }
           
        elif objective == 'regression':
            self.scoring = 'r2'
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

    def fit(self,X, y, steps = 6, folds = 2, scoring = 'auto', n_jobs = -1, cv = 2):

        best_score = None
        gcv = 1
        if scoring == 'auto':
            scoring = self.scoring
        for key in self.params.keys():
            learned_params = {}
            print(f"Searching model for {key}")
            for i in range(min([steps,len(self.params[key])])):
                print(f'step {i}')
                del gcv
                gcv = GridSearchCV(estimator = self.models[key](**learned_params), param_grid = self.params[key][i], scoring=scoring, cv=folds, verbose = 0, n_jobs = n_jobs, refit= False if i < (steps - 1) else True)
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
    def __init__(self, objective, max_evals):
        super().__init__(objective)
        self.space_xgb = bayes_params['XGB']


        self.space_lgb = bayes_params['LGBM']

        self.space_cat = bayes_params['CAT']

        self.objective = objective
        
        self.max_evals = max_evals

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


    def objective_function(self, space, model):
        inable_table = ['n_estimators','max_depth','num_leaves']
        for x in inable_table:
            if x in space.keys():
                space[x] = int(space[x])

        if isinstance(model, CatBoostClassifier) or isinstance(model, CatBoostRegressor):
            _model = model(logging_level = 'Silent', **space)
        else:
            _model = model(**space)

        _model.fit(self.X_train, self.y_train)
        train_score = self.scoring(self.y_train, _model.predict(self.X_train))
        
        return {'status': STATUS_OK, 'loss': self.loss(self.y_train, _model.predict(self.X_train)),
                'train score': train_score
            }

    

    def org_results(self,trials, hyperparams, model_name):
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
            'model': model_name,
            'parameter search time': train_time,
            'training score': train_score,
            'parameters': hyperparams
        }
        return results

    def fit(self, X, y):

        self.X_train = X
        self.y_train = y
        
        self.all_frameworks = {'objective':self.objective,'frameworks':[
        {
            'name':'XGB',
            'fn':  partial(self.objective_function, model = self.models['XGB']),
            'space': self.space_xgb
        },
        {
            'name':'LGBM',
            'fn': partial(self.objective_function, model = self.models['LGBM']),
            'space': self.space_lgb
        },
        {
            'name':'CAT',
            'fn': partial(self.objective_function, model = self.models['CAT']),
            'space': self.space_cat
        }
        ]}

        results_cumulative = {}
        for model in self.all_frameworks['frameworks']:
            print("Training {}".format(model['name']))
            trials = Trials()
            hyperparams = fmin(fn = model['fn'], 
                            max_evals = self.max_evals, 
                            trials = trials,
                            algo = tpe.suggest,
                            space = model['space']
                            )

            results = self.org_results(trials.trials, hyperparams, 'XGBoost')
            results_cumulative[model['name']] = {'score':results['training score'],'params' : results['parameters']}

        best_estimator = {}

        for key in results_cumulative.keys():
            if best_estimator == {} or results_cumulative[key]['score'] > best_estimator['score']:
                best_estimator = results_cumulative[key]
                best_estimator['booster'] = key

        inable_table = ['n_estimators','max_depth','num_leaves']
        for x in inable_table:
            if x in best_estimator['params'].keys():
                best_estimator['params'][x] = int(best_estimator['params'][x])

        self.best_model = self.models[best_estimator['booster']](**best_estimator['params']) 
        self.best_model.fit(X, y)   
        return self

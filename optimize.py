import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import GridSearchCV

class ModelSelector:
    def __init__(self,objective):
        
        
        self.figures = {}
        self.params = {
            "CAT": [
                    {
                        'n_estimators' : [i for i in range(50,275,25)]  
                    },
                    {
                        'learning_rate' : [0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.7,0.9]
                    },
                    {
                        'depth':range(3,10,2)
                    },
                    # {
                    #     'subsample':[i/10 for i in range(6,10)],
                    #     # 'rsm':[i/10 for i in range(6,10)]
                    # },
                    {
                        'l2_leaf_reg':[1e-5, 1e-2, 0.1, 1, 100]
                    }
                  
                ],
                "XGB": [
                {
                    'n_estimators' : [i for i in range(50,275,25)]
                },
                {
                    'learning_rate' : [0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.7,0.9]
                },
                {
                     'max_depth':range(3,10,2),
                     'min_child_weight': np.arange(0.5,6,0.5)
                },
                {
                    'gamma': [i/10.0 for i in range(0,5)]
                },
                {
                    'subsample':[i/10.0 for i in range(6,10)],
                    'colsample_bytree':[i/10.0 for i in range(6,10)]
                },
                {
                    'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
                }
                ],
                "LGBM": [
                    {
                        'n_estimators' : [i for i in range(50,275,25)]  
                    },
                    {
                        'learning_rate' : [0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.7,0.9]
                    },
                    {
                        'max_depth':range(3,10,2),
                        'num_leaves':np.arange(10,150,20)
                    },
                    {
                        'colsample_bytree':[i/10.0 for i in range(6,10)]
                    },
                    {
                        'reg_lambda':[1e-5, 1e-2, 0.1, 1, 100],
                        'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100],
                    },
                    {
                        'min_split_gain': [0.0001 * x for x in [10**i for i in range(4)]]
                    }
                ],
            }


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
                    if (score := gcv.best_estimator_.score(X,y)) > best_score:
                        best_score = score
                        self.best_model = gcv.best_estimator_
                else:
                    best_score = gcv.best_estimator_.score(X,y)
                    self.best_model = gcv.best_estimator_ 
            except:
                pass

            del learned_params
            
        return self


    def predict(self,X_test):
        return self.best_model.predict(X_test)


    def score(self, X, y):
        if self.best_model:
            return self.best_model.score(X,y)
        else:
            raise Exception("No fitted model available")

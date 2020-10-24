from hyperopt import hp
import numpy as np


grid_params = {
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

def get_bayes_params(cat = True, xgb = True, lgbm = True):
    """Return list of grids for specified gradient boosting libraries

    Args:
        cat (bool, optional): Whether to return CatBoost parameters. Defaults to True.
        xgb (bool, optional): Whether to return XGBoost parameters. Defaults to True.
        lgbm (bool, optional): Whether to return LightGBM parameters. Defaults to True.

    Returns:
        list: list of selected grids
    """

    xgb_grid =  {
            'name' : 'XGB',
            'n_estimators': hp.quniform('n_estimators_xgb', 50, 1000, 25),
            'max_depth': hp.quniform('max_depth_xgb', 1, 12, 1),
            'min_child_weight': hp.quniform('min_child_weight_xgb', 1, 6, 1),
            'gamma': hp.quniform('gamma_xgb', 0.5, 1, 0.05),
            'subsample': hp.quniform('subsample_xgb', 0.5, 1, 0.05),
            'learning_rate': hp.loguniform('learning_rate_xgb', np.log(.001), np.log(.3)),
            'colsample_bytree': hp.quniform('colsample_bytree_xgb', .5, 1, .1)

        }

    cat_grid =  {
            'name' : 'CAT',
            'n_estimators': hp.quniform('n_estimators_cat', 50, 1025, 25),
            'learning_rate': hp.loguniform('learning_rate_cat', np.log(0.005), np.log(0.3)),
            'depth': hp.quniform('depth_cat', 1, 16, 1),
            'border_count': hp.quniform('border_count_cat', 30, 220, 5), 
            'l2_leaf_reg': hp.quniform('l2_leaf_reg_cat', 1, 10, 1)
        }

    lgbm_grid = {
            'name' : 'LGBM',
            'learning_rate': hp.loguniform('learning_rate_lgbm', np.log(0.001), np.log(0.3)),
            'n_estimators': hp.quniform('n_estimators_lgbm', 50, 1200, 25),
            'max_depth': hp.quniform('max_depth_lgbm', 1, 15, 1),
            'num_leaves': hp.quniform('num_leaves_lgbm', 10, 150, 1),
            'feature_fraction': hp.uniform('feature_fraction_lgbm', .3, 1.0),
            'reg_lambda': hp.uniform('reg_lambda_lgbm', 0.0, 1.0),
            'reg_alpha': hp.uniform('reg_alpha_lgbm', 0.0, 1.0),
            'min_split_gain': hp.uniform('min_split_gain_lgbm', 0.0001, 0.1)
        }



    lista = []

    if lgbm:
        lista.append(lgbm_grid)
    if xgb:
        lista.append(xgb_grid)
    if cat:
        lista.append(cat_grid)

    bayes_params =  hp.choice('model_type', lista) 
    return bayes_params

    
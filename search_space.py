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


bayes_params = {
    "XGB" : {
            'n_estimators': hp.quniform('n_estimators', 50, 1000, 25),
            'max_depth': hp.quniform('max_depth', 1, 12, 1),
            'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
            'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
            'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
            'learning_rate': hp.loguniform('learning_rate', np.log(.001), np.log(.3)),
            'colsample_bytree': hp.quniform('colsample_bytree', .5, 1, .1)
        },
    "CAT": {
            'n_estimators': hp.quniform('n_estimators', 50, 1000, 25),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.005), np.log(0.3)),
            'depth': hp.quniform('depth', 1, 16, 1),
            'border_count': hp.quniform('border_count', 30, 220, 5), 
            'l2_leaf_reg': hp.quniform('l2_leaf_reg', 1, 10, 1)
        },
    "LGBM": {
            'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.3)),
            'n_estimators': hp.quniform('n_estimators', 50, 1200, 25),
            'max_depth': hp.quniform('max_depth', 1, 15, 1),
            'num_leaves': hp.quniform('num_leaves', 10, 150, 1),
            'feature_fraction': hp.uniform('feature_fraction', .3, 1.0),
            'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
            'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
            'min_split_gain': hp.uniform('min_split_gain', 0.0001, 0.1)
        }
}
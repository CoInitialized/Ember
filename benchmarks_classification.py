from ember.autolearn import Learner
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import numpy as np
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import tqdm
import os

# dataset_regression = r'C:\Users\cdami\Documents\Python Scripts\Ember\datasets\classification\seismic-bumps.csv'

# data = pd.read_csv(dataset_regression)
# X, y = data.drop(columns = ['class']), data['class']
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = 20)
# learner = Learner(objective='classification', X = X_train, y = y_train)
# learner.fit(cv = 3, optimizer = 'grid')

# xgb_default = XGBClassifier()
# xgb_default.fit(learner.feature_preprocessor.transform(X_train), learner.target_preprocessor.transform(np.array(y_train).reshape(-1,1)).ravel())
# results_default = xgb_default.predict(learner.feature_preprocessor.transform(X_test))

# results = learner.predict(X_test)
# y_test = learner.target_preprocessor.transform(np.array(y_test).reshape(-1,1)).ravel()
# print(accuracy_score(y_test, results.ravel()))
# print(accuracy_score(y_test, results_default.ravel()))

results = {

'name' : [],
'xgboost' : [],
'lightgbm' : [],
'catboost' : [],
'automl-grid' : []


}

failed_names = []

path = r'C:\Users\cdami\Documents\Python Scripts\Ember\datasets\classification'

names = os.listdir(path)

# names = ['auto-mpg.csv']

for name in tqdm.tqdm(names):
    try: 
        data = pd.read_csv(path + '/' +  name)
        X, y = data.drop(columns = ['class']), data['class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = 20)
        learner = Learner(objective='classification', X = X_train, y = y_train)
        learner.fit(cv = 3, optimizer = 'grid')
        _feature_preprocessor = learner.feature_preprocessor
        _target_preprocessor = learner.target_preprocessor
        y_test = _target_preprocessor.transform(np.array(y_test).reshape(-1,1)).ravel()
        score_auto = accuracy_score(y_test, learner.predict(X_test))
        xgb_default = XGBClassifier()
        cat_default = CatBoostClassifier(logging_level = 'Silent')
        lgbm_default = LGBMClassifier()
        X_train, X_test, y_train = _feature_preprocessor.transform(X_train), _feature_preprocessor.transform(X_test), _target_preprocessor.transform(np.array(y_train).reshape(-1,1)).ravel()
        print('training defaults')
        xgb_default.fit(X_train, y_train)
        cat_default.fit(X_train, y_train)
        lgbm_default.fit(X_train, y_train)
        score_xgb = accuracy_score(y_test, xgb_default.predict(X_test))
        score_cat = accuracy_score(y_test, cat_default.predict(X_test))
        score_lgbm = accuracy_score(y_test, lgbm_default.predict(X_test))
        del xgb_default, cat_default, lgbm_default, learner
        results['name'].append(name)
        results['xgboost'].append(score_xgb)
        results['lightgbm'].append(score_lgbm)
        results['catboost'].append(score_cat)
        results['automl-grid'].append(score_auto)
        print('RESULTS:')
        print(name, score_xgb, score_lgbm, score_cat, score_auto)
    except Exception as ex:
        print(f'{name} failed')
        print(ex)
        failed_names.append(name)


import json

with open('data.json', 'w') as fp:
    json.dump(results, fp)
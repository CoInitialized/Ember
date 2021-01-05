import pandas as pd
import numpy as np
from ember.impute import GeneralImputer
from ember.optimize import GridSelector, BayesSelector
from ember.utils import DtypeSelector
from sklearn.model_selection import train_test_split
from ember.preprocessing import Preprocessor, GeneralEncoder, GeneralScaler
from ember.optimize import BaesianSklearnSelector
from sklearn.metrics import r2_score, accuracy_score
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import tqdm
import datetime
import json
import neptune

objective = 'regression'

token = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiNTdiOWEyMDQtNWMxNi00MzY1LTk3N2ItMjY0MmI3Njk1NmUwIn0='
project_name = 'arkadiusz-czerwinski/eng'
neptune.init(project_qualified_name= project_name, # change this to your `workspace_name/project_name`
             api_token=token, # change this to your api token
            )


def preproces_data(X,y, target='class', objective='regression'):


    target_preprocessor = Preprocessor()
    target_preprocessor.add_branch('target')

    if y.dtype == np.object:

        target_preprocessor.add_transformer_to_branch('target', GeneralImputer('Simple', 'most_frequent'))
        target_preprocessor.add_transformer_to_branch('target', GeneralEncoder('LE'))
    else:
        if objective == 'classification':
            target_preprocessor.add_transformer_to_branch('target', GeneralImputer('Simple', 'most_frequent'))
        elif objective == 'regression':
            target_preprocessor.add_transformer_to_branch('target', GeneralImputer('Simple', 'mean'))
        else:
            pass

            ## features pipeline ##

    feature_preprocessor = Preprocessor()

    is_number = len(X.select_dtypes(include=np.number).columns.tolist()) > 0
    is_object = len(X.select_dtypes(include=np.object).columns.tolist()) > 0

    if is_object:
        feature_preprocessor.add_branch("categorical")
        feature_preprocessor.add_transformer_to_branch("categorical", DtypeSelector(np.object))
        feature_preprocessor.add_transformer_to_branch("categorical",
                                                       GeneralImputer('Simple', strategy='most_frequent'))
        feature_preprocessor.add_transformer_to_branch("categorical", GeneralEncoder(kind='OHE'))

    if is_number:
        feature_preprocessor.add_branch('numerical')
        feature_preprocessor.add_transformer_to_branch("numerical", DtypeSelector(np.number))
        feature_preprocessor.add_transformer_to_branch("numerical", GeneralImputer('Simple'))
        feature_preprocessor.add_transformer_to_branch("numerical", GeneralScaler('SS'))

    feature_preprocessor = feature_preprocessor.merge()
    target_preprocessor = target_preprocessor.merge()

    y = np.array(y).reshape(-1, 1)
    y = target_preprocessor.fit_transform(y).ravel()
    X = feature_preprocessor.fit_transform(X)
    return X, y

def change_df_column(df,to_change,new_name):
    df_columns = list(df.columns)
    for idx,column in enumerate(df_columns):
        if column == to_change:
            df_columns[idx] = new_name
    df.columns = df_columns

def get_lgbm_score(X_train,y_train,X_test,y_test):
    lgbm_default = LGBMRegressor()
    lgbm_default.fit(X_train, y_train)
    score_lgbm = accuracy_score(y_test, lgbm_default.predict(X_test))
    neptune.log_metric('lgbm', score_lgbm)
    return score_lgbm

def get_xgb_score(X_train,y_train,X_test,y_test):
    xgb_default = XGBRegressor()
    xgb_default.fit(X_train, y_train)
    score_xgb = accuracy_score(y_test, xgb_default.predict(X_test))
    neptune.log_metric('xgb', score_xgb)
    return score_xgb

def get_cat_score(X_train,y_train,X_test,y_test):
    cat_default = CatBoostRegressor(logging_level="Silent")
    cat_default.fit(X_train, y_train)
    score_cat = accuracy_score(y_test, cat_default.predict(X_test))
    neptune.log_metric('cat', score_cat)
    return score_cat
def get_gid_score(X_train,y_train,X_test,y_test,folds=5):
    model = GridSelector(objective,folds=folds, steps=6)
    model.fit(X_train, y_train)
    score = accuracy_score(y_test, model.predict(X_test))
    neptune.log_metric('grid', score)
    return score
def get_bayes_score(X_train,y_train,X_test,y_test,folds=5):
    model = BayesSelector(objective, cv=folds, max_evals=10)
    model.fit(X_train, y_train)
    score = accuracy_score(y_test, model.predict(X_test))
    neptune.log_metric('hyperopt', score)
    return score

def get_bayes_scikit_score(X_train,y_train,X_test,y_test,folds=5):
    model = BaesianSklearnSelector(objective, cv=folds, max_evals=25)
    model.fit(X_train, y_train)
    score = accuracy_score(y_test, model.predict(X_test))
    neptune.log_metric('skopt', score)
    return score


import os
def evaluate(path=r'datasets/regression'):
    global datasets
    failed_names = []
    scores = []
    names = os.listdir(path)
    datasets = [{"name":x,"target_column":"class"} for x in names]
    for dataset in tqdm.tqdm(datasets[:20]):
        try:
            neptune.create_experiment(name = str(datetime.datetime.today()).split()[0] + "_20_" + dataset['name'])
            data = pd.read_csv(path + '/' + dataset["name"])
            change_df_column(data, dataset['target_column'], 'class')
            X, y = data.drop(columns=['class']), data['class']
            X,y = preproces_data(X,y)
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2, stratify = y)
            score = {
                "bayes_scikit": get_bayes_scikit_score(X_train, y_train, X_test, y_test),
                'lgbm': get_lgbm_score(X_train,y_train,X_test,y_test),
                'xgb': get_xgb_score(X_train, y_train, X_test, y_test),
                'cat': get_cat_score(X_train, y_train, X_test, y_test),
                'bayes_hyperopt': get_bayes_score(X_train, y_train, X_test, y_test),
                "grid": get_gid_score(X_train, y_train, X_test, y_test),
                'name': dataset['name']
            }

            scores.append(score)
            print(score)

        except Exception as ex:
            print(f'{dataset["name"]} failed')
            print(ex)
            failed_names.append(dataset["name"])

    with open(str(datetime.datetime.today()).split()[0] + "_20" + ".json", 'w') as outfile:
        json.dump(scores, outfile)
    return scores


def evaluate_single():

    neptune.create_experiment(name = "LGBM first twenty")
    path = r'datasets/regression'
    names = os.listdir(path)
    datasets = [{"name":x,"target_column":"class"} for x in names]

    for dataset in tqdm.tqdm(datasets):
        data = pd.read_csv(path + '/' + dataset["name"])
        if data.count()[0] > 100:
            change_df_column(data, dataset['target_column'], 'class')
            X, y = data.drop(columns=['class']), data['class']
            X,y = preproces_data(X,y)
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
            # bayes = BayesSelector(objective,X_test = X_test, y_test = y_test, max_evals=10)
            # bayes_sklearn = BaesianSklearnSelector(objective,X_test = X_test, y_test = y_test, max_evals=10)
            # grid = GridSelector(objective)
            xgb_default = LGBMRegressor()
            xgb_default.fit(X_train, y_train)
            score_xgb = r2_score(y_test, xgb_default.predict(X_test))
            neptune.log_metric(dataset['name'], score_xgb)
if __name__ == '__main__':
    
    evaluate_single()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

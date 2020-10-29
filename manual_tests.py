from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from ember.preprocessing import Preprocessor
from ember.utils import DtypeSelector
from ember.optimize import GridSelector, BayesSelector
from sklearn.pipeline import make_pipeline
import pandas as pd
import numpy as np
from ember.impute import GeneralImputer
from ember.preprocessing import GeneralEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

datset_classification = r'Ember\datasets\classification\autos.csv'

data = pd.read_csv(datset_classification)

X, y = data.drop(columns = ['class']), data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, random_state = 42)

preprocessor = Preprocessor()
preprocessor.add_branch("categorical")
preprocessor.add_branch('numerical')

preprocessor.add_transformer_to_branch("numerical", DtypeSelector(np.number))
preprocessor.add_transformer_to_branch("numerical", GeneralImputer('Simple'))

preprocessor.add_transformer_to_branch("categorical", DtypeSelector(np.object))
preprocessor.add_transformer_to_branch("categorical", GeneralImputer('Simple', strategy='most_frequent'))
preprocessor.add_transformer_to_branch("categorical", GeneralEncoder(kind = 'LE'))

final = preprocessor.merge()

test = time.time()
# model = CatBoostClassifier(iterations = 1000, border_count = 350.0 ,logging_level='Silent')
# model = XGBClassifier(n_estimators=500) # 0.7884
# model = BayesSelector(objective = 'classification', X_test=final.fit(X_train).transform(X_test), y_test = y_test, max_evals = 100, cat = False)
model = BayesSelector(objective = 'classification', max_evals = 10, cat = False)
clf_pipe = make_pipeline(final, model) 
clf_pipe.fit(X_train, y_train)

print(accuracy_score(y_test, clf_pipe.predict(X_test)))

print(time.time() - test)

print(model.get_params())
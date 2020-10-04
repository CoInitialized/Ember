test_grid = {
                    
                        'n_estimators' : [i for i in range(50,275,25)]  
            }
                    



from preprocessing import Preprocessor
from selector import DtypeSelector
from optimize import ModelSelector
from sklearn.pipeline import make_pipeline
import pandas as pd
import numpy as np
from missing import GeneralImputer
from preprocessing import GeneralEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostClassifier

class kurwa_bzdura:

    def __init__(self):
        pass

    def fit(self, X, y):
        print(y.shape)
        self.gcv = GridSearchCV(estimator = CatBoostClassifier(), param_grid = test_grid, scoring='accuracy', cv=5, verbose = 2, n_jobs = -1)
        self.gcv.fit(X,y)
        return self
    def predict(self, X, y):
        self.gcv.predict(X)



dataset_regression = 'datasets/regression/mbagrade.csv'
datset_classification = 'datasets/classification/autos.csv'


data = pd.read_csv(datset_classification)


preprocessor = Preprocessor()
preprocessor.add_branch("categorical")
preprocessor.add_branch('numerical')

preprocessor.add_transformer_to_branch("numerical", DtypeSelector(np.number))
preprocessor.add_transformer_to_branch("numerical", GeneralImputer('Simple'))

preprocessor.add_transformer_to_branch("categorical", DtypeSelector(np.object))
preprocessor.add_transformer_to_branch("categorical", GeneralImputer('Simple', strategy='most_frequent'))
preprocessor.add_transformer_to_branch("categorical", GeneralEncoder(kind = 'LE'))

final = preprocessor.merge()

model = ModelSelector('classification')

clf_pipe = make_pipeline(final, model) 

X, y = data.drop(columns = ['class']), data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, random_state = 42)


clf_pipe.fit(X_train, y_train)

print(clf_pipe.score(X_test,y_test))
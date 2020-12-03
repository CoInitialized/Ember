# Ember - Automated Gradient Boosting Framework

- [Ember - Automated Gradient Boosting Framework](#ember---automated-gradient-boosting-framework)
  - [What is it](#what-is-it)
  - [How to use it](#how-to-use-it)
    - [<b>Submodules example</b>:](#bsubmodules-exampleb)
    - [<b>Automatic learner example</b>](#bautomatic-learner-exampleb)
  - [Further plans](#further-plans)


Work still in progress. When first version is ready - the package will be published to PyPi

## What is it

Ember is a project that aims to provide developer with easy to use interface to create machine learning pipelines.
From preprocessing to final optimized model.

It is based on [scikit-learn](https://scikit-learn.org/) and gradient boosting frameworks like:

- [CatBoost](https://xgboost.readthedocs.io/en/latest/)
- [XGBoost](https://catboost.ai/)
- [LightXGM](https://lightgbm.readthedocs.io/en/latest/)

It also uses some other libraries like

- [Category encoders](https://contrib.scikit-learn.org/category_encoders/index.html) - For more encoding options
- [hyperopt](https://github.com/hyperopt/hyperopt) - For automatic optimization of models

To your disposal are:

- Independent modules:
  - [optimize](ember/optimize.py)
  - [preprocessing](ember/preprocessing.py)
  - [impute](ember/impute.py)
  - [utils](ember/utils.py)
- Automatic Learner Module:
  - [autolearn](ember/autolearn.py)


## How to use it

Currently you need to clone the repository or download it as zip and then include the code in project manually. It should change soon so you will be able to use PyPi package.

### <b>Submodules example</b>:

In most cases you would like to use independent modules for quickly assembling your model.
This process is inspired by the way [keras](https://keras.io/) works.

```Python
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

dataset_classification = r'Ember\datasets\classification\autos.csv'

data = pd.read_csv(dataset_classification)

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

model = GridSelector('classification')

clf_pipe = make_pipeline(final, model) 
clf_pipe.fit(X_train, y_train)

print(accuracy_score(y_test, clf_pipe.predict(X_test)))
```

As you can see creating whole pipeline is very simple. If you were to skip imports it would be many times shorter than equivalent written in standard way.

### <b>Automatic learner example</b>

You also have an option to completely automate the process. In this case you can use ``Learner`` class from [autolearn](ember/autolearn.py) module.

```Python
from ember.autolearn import Learner
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

dataset_regression = r'Ember\datasets\regression\auto-price.csv'

data = pd.read_csv(dataset_regression)
X, y = data.drop(columns = ['class']), data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = 20)
learner = Learner(objective='regression', X = X_train, y = y_train)
learner.fit(cv = 5, optimizer = 'bayes', cat=False, speed=100)


results = learner.predict(X_test)
print(r2_score(y_test, results))

```

The way ``Learner`` creates and optimizes model can be controlled by providing different hyperparameters to its ``fit()`` method.

## Further plans

- More clearly comment modules - closer following [sklearn docs](https://scikit-learn.org/stable/modules/classes.html) convention
- Extend optimization module so it can be used in more general cases
- Build visualization module
- Build automatic dashboard
- <b>Publish as PyPi package</b>




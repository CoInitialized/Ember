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

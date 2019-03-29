import pandas as pd
from sklearn.ensemble import *
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import *

training_data = pd.read_csv('data/train.csv')

test_data = pd.read_csv('data/test.csv')

def make_Y(df):
	Y = pd.DataFrame(df['Survived'])
	Y.index = df.index
	return Y

def make_normalized_inputs(df):
	"""
	Normalize X and Y.
	"""
	ndf = df.copy()
	cols = ['Age', 'SibSp', 'Parch', 'Fare'] # todo other cols later
	X = ndf[cols]
	for c in cols:
		std = X[c].std()
		X[c] = X[c] / std
		mean = X[c].mean()
		X[c] = X[c].fillna(mean)

	X['sex'] = (ndf['Sex'] == 'male').astype(int)
	X['Pclass'] = ndf['Pclass']

	return X

Xtrain = make_normalized_inputs(training_data)
Ytrain = make_Y(training_data)

Xtest = make_normalized_inputs(test_data)

model = GradientBoostingClassifier(n_estimators=10, max_depth=5, max_features='auto')


params = {
	'max_depth': [x for x in range(1, 10)],
	'n_estimators': [x for x in range(1, 100)],
	'min_samples_leaf': [x for x in range(1, 40)],
	'max_features': ['auto', 'sqrt', 0.5]
}

iterations = 20
random_search = RandomizedSearchCV(model,
	param_distributions=params, n_iter=iterations,
	verbose=2, n_jobs=-1, scoring='accuracy')

# grid_search = GridSearchCV(model, params,
# verbose=2, n_jobs=-1, scoring='accuracy')
# grid_search.fit(Xtrain, Ytrain)
# best_model = grid_search.best_estimator_

random_search.fit(Xtrain, Ytrain)
best_model = random_search.best_estimator_


cvs = cross_val_score(best_model, Xtrain, Ytrain, scoring='accuracy')
print(sum(cvs) / len(cvs))


predictions = pd.DataFrame(best_model.predict(Xtest))
predictions.columns = ['Survived']
predictions.index = test_data['PassengerId']

predictions.to_csv('predictions.csv', index_label='PassengerId')

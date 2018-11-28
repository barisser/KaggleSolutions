import pandas as pd
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import cross_val_score, RandomizedSearchCV

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
		X[c] = X[c].fillna(-1)

	X['sex'] = (ndf['Sex'] == 'male').astype(int)

	return X

Xtrain = make_normalized_inputs(training_data)
Ytrain = make_Y(training_data)

Xtest = make_normalized_inputs(test_data)

model = rfc(n_estimators=10, max_depth=5)

r = 0.8
m=int(r*len(Xtrain))


params = {
	'max_depth': range(1, 20),
	'n_estimators': range(1, 400)
}
iterations = 20
random_search = RandomizedSearchCV(model,
	param_distributions=params, n_iter=iterations,
	verbose=2, n_jobs=-1)

random_search.fit(Xtrain, Ytrain)

best_model = random_search.best_estimator_


cvs = cross_val_score(best_model, Xtrain, Ytrain, scoring='accuracy')
print(sum(cvs) / len(cvs))


predictions = pd.DataFrame(best_model.predict(Xtest))
predictions['PassengerId'] = Xtest.index
predictions.columns = ['PassengerId', 'Survived']


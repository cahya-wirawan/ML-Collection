import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score
import pickle

# load data
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=",")
# split data into X and y
X = dataset[:,0:8]
Y = dataset[:,8]
# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
# l = dtrain.get_label()
# print(np.unique(l))

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
"""
# set xgboost params
params_grid = {
    'max_depth': [1, 2, 3, 4],
    'n_estimators': [i for i in range(1, 120, 2)],
    'learning_rate': np.linspace(1e-16, 1, 20)
}

params_fixed = {
    'objective': 'binary:logistic',
    'silent': 1
}
"""

# set xgboost params
params_grid = {
}

params_fixed = {
    'learning_rate': 0.1,
    'n_estimators': 100,
    'max_depth': 3,
    'objective': 'binary:logistic',
    'silent': 1
}

num_round = 30  # the number of training iterations

bst_grid = GridSearchCV(
    estimator=xgb.XGBClassifier(**params_fixed, seed=seed),
    param_grid=params_grid,
    cv=cv,
    scoring='accuracy'
)

bst_grid.fit(X_train, y_train)

# save model to file
pickle.dump(bst_grid, open("bst_grid-pima.pickle.dat", "wb"))

#print(bst_grid.cv_results_)
print("Best accuracy obtained: {0}".format(bst_grid.best_score_))
print("Parameters:")
for key, value in bst_grid.best_params_.items():
    print("\t{}: {}".format(key, value))

y_pred = bst_grid.predict(X_test)
predictions = [np.round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

labels = dtest.get_label()
correct = 0

for i in range(len(predictions)):
    if (labels[i] == predictions[i]):
        correct += 1

print('Predicted correctly: {0}/{1}'.format(correct, len(predictions)))
print('Error: {0:.4f}'.format(1-correct/len(predictions)))


# watchlist  = [(dtest,'test'), (dtrain,'train')]


# training and testing - numpy matrices
# bst = xgb.train(param, dtrain, num_round, watchlist)
# y_pred = bst.predict(dtest)
""" 
trees_dump = bst.get_dump(fmap='../data/featmap.txt', with_stats=True)
for tree in trees_dump:
    print(tree)
#xgb.plot_importance(bst, importance_type='gain', xlabel='Gain')
#predictions = [np.argmax(value) for value in y_pred]
predictions = [np.round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

labels = dtest.get_label()
correct = 0

for i in range(len(predictions)):
    if (labels[i] == predictions[i]):
        correct += 1

print('Predicted correctly: {0}/{1}'.format(correct, len(predictions)))
print('Error: {0:.4f}'.format(1-correct/len(predictions)))
"""
import pandas
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss

data = pandas.read_csv('gbm-data.csv')
y = data['Activity'].values
train = data.drop('Activity', axis=1).values
X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.8, random_state=241)
result = {}
"""
for learning_rate in [1, .5, .3, .2, .1]:
"""
for learning_rate in [.2]:
    classifier = GradientBoostingClassifier(n_estimators=250, verbose=True, random_state=241,
                                            learning_rate=learning_rate)
    classifier.fit(X_train, y_train)
    sdf_train = classifier.staged_decision_function(X_train)
    sdf_test = classifier.staged_decision_function(X_test)
    y_train_pred = []
    y_test_pred = []
    log_train = []
    log_test = []
    for i, pred in enumerate(sdf_train):
        y_train_pred.append(1 / (1 + np.exp(-pred)))
        log_train.append(log_loss(y_train, y_train_pred[i]))
    m = 19999
    ii = 0
    for i, pred in enumerate(sdf_test):
        y_test_pred.append(1 / (1 + np.exp(-pred)))
        log_test.append(log_loss(y_test, y_test_pred[i]))
        if log_test[i] < m:
            m = log_test[i]
            ii = i

    print(m, ii)
pass

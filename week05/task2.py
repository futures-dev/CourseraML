import pandas
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss

data = pandas.read_csv('gbm-data.csv')
train = data.drop(0).values
y = data[1:].values
X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.8, random_state=241)
classifier = GradientBoostingClassifier(n_estimators=250, verbose=True, random_state=241)
result = {}
for learning_rate in [1, .5, .3, .2, .1]:
    classifier.fit(X_train, y_train)
    sdf_train = classifier.staged_decision_function(X_train)
    sdf_test = classifier.staged_decision_function(X_test)
    hell = 1 / (1 + np.e ** (-classifier.predict(X_test)))
    log_loss_train = log_loss(y_train, classifier.predict(X_train))
    log_loss_test = log_loss(y_test, classifier.predict(X_test))
    result[learning_rate] = {
        'sdf_train': sdf_train,
        'sdf_test': sdf_test,
        'hell': hell,
        'log_loss_train': log_loss_train,
        'log_loss_test': log_loss_test}
pass

import pandas
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(tol=1e-5, C=10, max_iter=10000, intercept_scaling=0.1)
train = pandas.read_csv('data-logistic.csv')
y = train['a']
train.drop(train[[0]], 1, inplace=True)

logreg.fit(train, y)
roc_auc_score(y, logreg.class_weight)

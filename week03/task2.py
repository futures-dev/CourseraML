from sklearn import datasets
import numpy as np
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer

newsgroups = datasets.fetch_20newsgroups(
    subset='all',
    categories=['alt.atheism', 'sci.space']
)

vectorizer = TfidfVectorizer()
train = vectorizer.fit_transform(newsgroups.data)
y = newsgroups.target
# print train
# print vectorizer.get_feature_names()[12328]



y = newsgroups.target
print(np.asarray(y))
m = 0
grid = {'C': np.power(10.0, [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])}
fold = KFold(y.size, n_folds=5, shuffle=True, random_state=241)
svc = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(svc, grid, cv=5)
gs.fit(train, y)
cf = gs.best_estimator_.coef_
q = np.argsort(cf, axis=0)
print(q[::-1][0:10])

"""

m = 0
param = 0
for a in gs.grid_scores_:
    if a.mean_validation_score>m:
        m = a.mean_validation_score
        param = a.parameters

print(param)


"""

"""
svc = SVC(kernel='linear',random_state=241,C=10.0)
svc.fit(train,y)
ar = svc.support_
print ar
print ar[::1][0:10]
"""
while True:
    c = input()
    print (vectorizer.get_feature_names()[c])

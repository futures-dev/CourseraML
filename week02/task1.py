import pandas
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier

data = pandas.read_csv('wine.data')
kfold = cross_validation.KFold(n_folds=5, shuffle=True, random_state=42, n=177)
for k in range(1, 51):
    classifier = KNeighborsClassifier(n_neighbors=k)
    print(k)
    print(cross_validation.cross_val_score(classifier, data, cv=kfold))

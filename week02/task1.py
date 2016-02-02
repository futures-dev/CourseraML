import pandas
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale

data = pandas.read_csv('wine.data')
Y = data['class']
data.drop(data.columns[[0]], 1, inplace=True)
data = scale(data)
kfold = cross_validation.KFold(n_folds=5, shuffle=True, random_state=42, n=178)
m = []
for k in range(1, 51):
    classifier = KNeighborsClassifier(n_neighbors=k)
    m.append((k, cross_validation.cross_val_score(classifier, data, Y, cv=kfold, scoring='accuracy').mean()))
m.sort(key=lambda t: t[1], reverse=True)
print(m[0])
print(m[1])

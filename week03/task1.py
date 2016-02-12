import pandas
from sklearn.svm import SVC

svc = SVC(C=100000, random_state=241, kernel='linear')
train = pandas.read_csv('svm-data.csv')
y = train['a']
x = train[['b', 'c']]
svc.fit(x, y)
print(sorted(svc.support_ + 1))

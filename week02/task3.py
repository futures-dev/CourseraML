from sklearn import datasets
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsRegressor
from sklearn import cross_validation

boston = datasets.load_boston()
data = scale(boston.data)
kfold = cross_validation.KFold(n_folds=5, shuffle=False, random_state=42, n=506)
p = 1.0
m = -100
p0 = 0
for i in range(200):
    p += 0.045
    reg = KNeighborsRegressor(n_neighbors=5, weights='distance', metric='minkowski', p=p)
    temp = cross_val_score(reg, data, boston.target, scoring='mean_squared_error', cv=kfold).mean()
    if temp > m:
        m = temp
        p0 = p
print(p0)

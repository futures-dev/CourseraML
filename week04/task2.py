import pandas
from sklearn.decomposition import PCA

data_prices = pandas.read_csv('close_prices.csv')
data_index = pandas.read_csv('djia_index.csv')
data_prices.drop('date', axis=1, inplace=True)
for i in range(1, 20):
    pca = PCA(i)
    pca.fit_transform(data_prices)
    sum = 0
    for s in pca.explained_variance_ratio_:
        sum += float(s)
    print(i, sum)
"""
pca = PCA(10)
pca.fit_transform(data_prices)
new = pca.transform(data_prices)
first_component = pca.components_[0]
a = map(lambda x: x[0],new)
print(corrcoef(a,data_index[u'^DJI']))
t = zip(pca.components_[0],data_prices.columns)
print(max(t))
pass
"""

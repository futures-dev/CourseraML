from sklearn.tree import DecisionTreeClassifier
import pandas

data = pandas.read_csv('titanic.csv', index_col='PassengerId')
data['Sex'] = data['Sex'].astype('category')
sex_columns = data.select_dtypes(['category']).columns
data[sex_columns] = data[sex_columns].apply(lambda x: x.cat.codes)
data.drop(data.columns[[2, 5, 6, 7, 9, 10]], 1, inplace=True)
data.dropna(0, inplace=True)
Y = data['Survived']
data.drop(data.columns[[0]], 1, inplace=True)
print(data)
tree = DecisionTreeClassifier(random_state=241)
tree.fit(data, Y)
print(tree.feature_importances_)

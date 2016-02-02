import pandas
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler

train = pandas.read_csv('perceptron-train.csv')
test = pandas.read_csv('perceptron-test.csv')

per = Perceptron(random_state=241)
y_train = train['a']
y_test = test['a']
print(y_test)
train.drop(train.columns[[0]], 1, inplace=True)
test.drop(test.columns[[0]], 1, inplace=True)
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.fit_transform(test)
per.fit(train, y_train)
print(per.score(test, y_test))
per.fit(train_scaled, y_train)
print(per.score(test_scaled, y_test))

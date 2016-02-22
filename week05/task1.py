import pandas
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestRegressor

abalone_data = pandas.read_csv('abalone.csv')
abalone_data['Sex'] = abalone_data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))
Y = abalone_data['Rings']
train = abalone_data[
    ['Sex', 'Length', 'Diameter', 'Height', 'WholeWeight', 'ShuckedWeight', 'VisceraWeight', 'ShellWeight']]
for n in range(1, 51):
    clf = RandomForestRegressor(n_estimators=n, random_state=1)
    kFold = KFold(len(train), random_state=1, n_folds=5, shuffle=True)
    """
    for train_index, test_index in kFold:
        XTrain, XTest = train.ix[train_index],train.ix[test_index]
        YTrain, YTest = Y[train_index],Y[test_index]
        clf.fit(XTrain, YTrain)
        sc = r2_score(YTest,clf.predict(XTest))
        if sc>0.52:
            print(n)
    """
    sc = cross_validation.cross_val_score(clf, train, Y, scoring='r2', cv=kFold)
    if sc.mean() > 0.52:
        print(n)

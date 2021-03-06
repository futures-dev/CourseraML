import pandas
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import Ridge

train = pandas.read_csv('salary-train.csv')
test = pandas.read_csv('salary-test-mini.csv')

train['FullDescription'] = train['FullDescription'].apply(lambda x: x.lower())
train['FullDescription'] = train['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex=True)
test['FullDescription'] = test['FullDescription'].apply(lambda x: x.lower())
test['FullDescription'] = test['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex=True)

vectorizer = TfidfVectorizer(min_df=5)
t = vectorizer.fit_transform(train['FullDescription'])
t1 = vectorizer.transform(test['FullDescription'])
train['LocationNormalized'].fillna('nan', inplace=True)
train['ContractTime'].fillna('nan', inplace=True)
test['LocationNormalized'].fillna('nan', inplace=True)
test['ContractTime'].fillna('nan', inplace=True)

enc = DictVectorizer()
X_train_categ = enc.fit_transform(train[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test_categ = enc.transform(test[['LocationNormalized', 'ContractTime']].to_dict('records'))

new_data = hstack([t, X_train_categ])
new_test = hstack([t1, X_test_categ])

ridge = Ridge(alpha=1)
ridge.fit(new_data, train['SalaryNormalized'])
pr = ridge.predict(new_test)
print(pr)

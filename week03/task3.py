from __future__ import division
import pandas
import numpy as np
from sklearn.metrics import roc_auc_score

data = pandas.read_csv('data-logistic.csv')
train = data[['b', 'c']]
y = data['a']

l = y.size
w1 = 0
w2 = 0
cnt = 0
k = 0.1
C = 10
e = 100
while e > 0.00001 and cnt < 10000:
    a = 0
    b = 0
    for i in range(l):
        a += (y[i] * train['b'][i] * (
            1 - 1 / (1 + np.exp(-y[i] * (w1 * train['b'][i] + w2 * train['c'][i])))
        ))
        b += (y[i] * train['c'][i] * (
            1 - 1 / (1 + np.exp(-y[i] * (w1 * train['b'][i] + w2 * train['c'][i])))
        ))
    a *= k
    b *= k
    a /= l
    b /= l
    a -= k * C * w1
    b -= k * C * w2
    w11 = w1 + a
    w22 = w2 + b
    e = np.sqrt((w1 - w11) ** 2 + (w2 - w22) ** 2)
    w1 = w11
    w2 = w22
    cnt += 1

a = []
for i in range(l):
    a.append(1 / (1 + np.exp(-w1 * train['b'][i] - w2 * train['c'][i])))

print(roc_auc_score(y, a))

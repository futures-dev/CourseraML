import pandas
from sklearn.metrics import precision_recall_curve

classdata = pandas.read_csv('classification.csv')
scoresdata = pandas.read_csv('scores.csv')

tp = fp = fn = tn = 0

l = classdata['true'].size

"""
for i in range(l):
    x = classdata['true'][i]
    y = classdata['pred'][i]
    if x==1:
        if y==1:
            tp += 1
        else:
            fn += 1
    else:
        if y==1:
            fp += 1
        else:
            tn += 1

print(tp)
print(fp)
print(fn)
print(tn)
"""

"""
print(accuracy_score(classdata['true'],classdata['pred']))
print(precision_score(classdata['true'],classdata['pred']))
print(recall_score(classdata['true'],classdata['pred']))
print(f1_score(classdata['true'],classdata['pred']))
"""

"""
print(roc_auc_score(scoresdata['true'],scoresdata['score_logreg']))
print(roc_auc_score(scoresdata['true'],scoresdata['score_svm']))
print(roc_auc_score(scoresdata['true'],scoresdata['score_knn']))
print(roc_auc_score(scoresdata['true'],scoresdata['score_tree']))
"""
l = scoresdata['true'].size

logreg = precision_recall_curve(scoresdata['true'], scoresdata['score_logreg'])
score_svm = precision_recall_curve(scoresdata['true'], scoresdata['score_svm'])
score_knn = precision_recall_curve(scoresdata['true'], scoresdata['score_knn'])
score_tree = precision_recall_curve(scoresdata['true'], scoresdata['score_tree'])
print(score_tree)

max = 0
c = ''
for i in range(l):
    if i < logreg[1].size:
        if logreg[1][i] >= 0.7:
            if logreg[0][i] > max:
                max = logreg[0][i]
                c = 'score_logreg'
    if i < score_svm[1].size:
        if score_svm[1][i] >= 0.7:
            if score_svm[0][i] > max:
                max = score_svm[0][i]
                c = 'score_svm'
    if i < score_knn[1].size:
        if score_knn[1][i] >= 0.7:
            if score_knn[0][i] > max:
                max = score_knn[0][i]
                c = 'score_knn'

    if i < score_tree[1].size:
        if score_tree[1][i] >= 0.7:
            if score_tree[0][i] > max:
                max = score_tree[0][i]
                c = 'score_tree'

print(c)
print(max)

import pandas
data = pandas.read_csv('titanic.csv',index_col='PassengerId')
"""
nameStrings = data['Name']
names={}
for line in nameStrings:
    if line.find(', Mrs')>0 or line.find(', Miss')>0 or line.find(', Lady')>0 or line.find(', Mme')>0 or line.find(', Ms')>0 or line.find(', Mlle')>0 or line.find(', Countess')>0 :
        name = line[line.find('. ')+2:]
        if name in names:
            names[name]+=1
        else:
            names[name] = 1
for key in names.keys():
    print(key+'\t'+str(names[key]))
    """
print(data['Age'].mean())
print(data['Age'].median())
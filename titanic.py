# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 13:07:08 2017

@author: DIU
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
import sklearn.metrics as mt

def k_fold(model, x, y):
    scr = cross_val_score(model, x, y, cv=10, scoring='accuracy')
    return scr.mean()

train_data = pd.read_csv('path/titanic/titanic_train.csv')
test_data = pd.read_csv('path/titanic/titanic_test.csv')

fcols = ['Pclass', 'Sex', 'Fare']

x_train = train_data[fcols]
y_train = train_data.Survived


x_test = test_data[fcols]

pid = test_data['PassengerId']
pid = pid.values

#checking whether there is any nan values in the feature columns
"""print(x_train['Sex'].isnull().sum())
print(x_train['Age'].isnull().sum())
print(x_train['Fare'].isnull().sum())
print(x_train['Pclass'].isnull().sum())"""

"""k_range =[]
for i in range(31):
    k_range.append(i+1)

param_grid = dict(n_neighbors=k_range)    """

#as we have 177 nan values in the age columns, we will replace them with the median of age column
"""x_train['Age'] = x_train['Age'].fillna(x_train['Age'].median())
x_test['Age'] = x_test['Age'].fillna(x_test['Age'].median())"""
x_test['Fare'] = x_test['Fare'].fillna(x_test['Fare'].median())

#converting incompetible data type into competible. string to int
d = {'male':0, 'female':1}
x_train['Sex'] = x_train['Sex'].apply(lambda x:d[x])
x_test['Sex'] = x_test['Sex'].apply(lambda x:d[x])

#knn = KNeighborsClassifier()

#grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
#grid.fit(x_train, y_train)
#best = grid.best_params_
#best1 = best['n_neighbors']

"""knn2 = KNeighborsClassifier(n_neighbors=best1)
knn2.fit(x_train, y_train)
y_pred = knn2.predict(x_test)

score = cross_val_score(knn2, x_train, y_train, cv=11, scoring='accuracy')
print(score.mean())"""

mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(10,10,10,10), random_state=1)
tree = DecisionTreeClassifier()
rnd = RandomForestClassifier(n_estimators=325)
svc = SVC(kernel='rbf')

model = rnd
print(k_fold(model, x_train, y_train))
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
final = np.column_stack((pid,y_pred))
df = pd.DataFrame({'PassengerId':pid.tolist(), 'Survived':y_pred.tolist()})
#df.to_csv('result4.csv', sep=',', encoding='utf-8')


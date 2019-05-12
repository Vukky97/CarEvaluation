# Checking libraries version
import sklearn

from pandas.plotting import scatter_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.model_selection import train_test_split
from sklearn import datasets, ensemble, model_selection
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

print('Checking libraries version:')
import scipy
print('Scipy: {}'.format(scipy.__version__))
import numpy as np
print('Numpy: {}'.format(np.__version__))
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
import pandas as pd
print('pandas: {}'.format(pd.__version__))
#from sklearn.svm.libsvm import cross_validation
import matplotlib.pyplot as plt

#name colums
column_names = ['buy_price','maint_cost','doors','persons','lug','safety','class']

# load dataset
#CarDataset = pd.read_csv('cardata.csv',names = column_names)
df = pd.read_csv('cardata.csv',names=column_names,header=-1)
df.info()
df = df.replace('unacc', 1)
df = df.replace('acc', 2)
df = df.replace('good', 3)
df = df.replace('vgood', 4)
df = df.replace('low', 1)
df = df.replace('med', 2)
df = df.replace('high', 3)
df = df.replace('vhigh', 4)
df = df.replace('more', 5)
df = df.replace('small', 1)
df = df.replace('med', 2)
df = df.replace('big', 3)
df = df.replace('5more', 6)
#tooks me 2 hpur to find out
df = df.replace('2',2)
df = df.replace('3',3)
df = df.replace('4',4)


# basic test of dataset
#print(CarDataset.shape)
#print(CarDataset)
#print(CarDataset.describe())
#print(CarDataset.groupby('class').size())

print(df)

print(df.shape)

# data visualization

#amazing
#df.plot(kind='area')
#df.plot(kind='kde')
#df.plot(kind='box')
#plt.show()
print(df.describe())
print(df.groupby('class').size())
df.hist()
plt.show()




cars2 = df.values
X = cars2[:,0:6]
Y = cars2[:,6]
validation_size = 0.20
seed = 7
#scoring = 'accuarcy'
X_train, X_validation, Y_train, Y_validation = train_test_split(X,Y,test_size=validation_size,random_state=seed)


#algorithms: https://scikit-learn.org/stable/
models = []
models.append(('LR',LogisticRegression(solver='liblinear',multi_class='ovr')))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('NB',GaussianNB()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('SVM',SVC(gamma='auto')))
models.append(('RFC',RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)))
models.append(('CART',DecisionTreeClassifier()))

#scoring = sklearn.metrics.SCORES.keys()
#https://scikit-learn.org/stable/modules/model_evaluation.html
# scoring can be balanced_accuracy,accuracy, and so on
scoring='balanced_accuracy'
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10,random_state=seed)
    cv_results = model_selection.cross_val_score(model,X_train,Y_train,cv=kfold,scoring=scoring)
    results.append(cv_results)
    names.append(name)
    str = "%s: %f (%f)" % (name,cv_results.mean(),cv_results.std())
    print(str)

fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

dtc = DecisionTreeClassifier()
dtc.fit(X_train,Y_train)
predictions = dtc.predict(X_validation)
print(accuracy_score(Y_validation,predictions))
print(confusion_matrix(Y_validation,predictions))
print(classification_report(Y_validation,predictions))




# clf = ensemble.RandomForestClassifier(n_estimators=500)
# clf.fit(X_train,Y_train)
# RandomForestClassifier(
#             criterion='gini', max_depth=None, max_features='auto',
#             max_leaf_nodes=None, min_samples_leaf=1,
#             min_samples_split=2, n_estimators=500, n_jobs=1,
#             oob_score=False, random_state=None, verbose=0)
# resu = clf.score(X_validation,Y_validation)
# print(resu)

# cars = df.values
# X,y = cars[:,:6], cars[:,6]
# X,y = X.astype(int), y.astype(int)
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

# clf = ensemble.RandomForestClassifier(n_estimators=500)
# clf.fit(X_train,y_train)
# RandomForestClassifier(
#             criterion='gini', max_depth=None, max_features='auto',
#             max_leaf_nodes=None, min_samples_leaf=1,
#             min_samples_split=2, n_estimators=500, n_jobs=1,
#             oob_score=False, random_state=None, verbose=0)
# resu = clf.score(X_test,y_test)
# print(resu)

# from sklearn import preprocessing
# scaler = preprocessing.MinMaxScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.fit_transform(X_test)
# clf.fit(X_train_scaled,y_train)
# resu2 = clf.score(X_test_scaled,y_test)
# print(resu2)
#
#
#
# clf2 = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
# result = clf2.score(X_test, y_test)
# print(result)









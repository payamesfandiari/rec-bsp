'''
Created on Oct 30, 2015

@author: payam
'''

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from sklearn.externals.six import StringIO
import pydot_ng
from sklearn import tree


dataset_name = 'wall_follow/'
data_path = 'datasets_v1/{0}data'.format(dataset_name)
trueclass = 'datasets_v1/{0}trueclass'.format(dataset_name)
all_data = np.loadtxt(data_path, dtype='f')
# all_data = all_data[:,[23,1,2]]
pca = PCA(n_components=2)
all_data_X = pca.fit_transform(all_data)
true_labels = np.loadtxt(trueclass, dtype='i')

randomclass = 'datasets_v1/{0}random_class.{1}'.format(dataset_name,0)
train_labels = np.loadtxt(randomclass,dtype='i')
train_labels = train_labels[train_labels[:,1].argsort()]
train_data = all_data[train_labels[:,1],:]
ind = np.in1d(true_labels[:,1], train_labels[:,1], assume_unique=True)==False
test_data = all_data[ind,:]
test_labels = true_labels[ind]
train_labels = train_labels[:,0]


fig, ax = plt.subplots()


clf = DecisionTreeClassifier(random_state=0)
clf.fit(train_data,train_labels)
# f = open('wall_follow.dot','w')
# tree.export_graphviz(clf, out_file=f)


np.set_printoptions(precision=4, linewidth=400, suppress=True)
print(clf.score(test_data, test_labels[:,0]))
print("Tree F importance : ",clf.feature_importances_[np.argsort(clf.feature_importances_)])
print(np.argsort(clf.feature_importances_))
# print(np.argsort(clf.feature_importances_))

colors =  true_labels[:,0].tolist()
colors= ['b' if c==1 else 'r' for c in colors]

ax.scatter(all_data[:,0],all_data[:,1],c=colors)



linear = LinearSVC(C = 1)
linear.fit(train_data, train_labels)
print("W = ",linear.coef_[0][np.argsort(clf.feature_importances_,)])

# print(linear.coef_[0][23])
# print(clf.score(test_data,test_labels[:,0]))
print(linear.score(test_data,test_labels[:,0]))

F,pval = f_classif(train_data,train_labels)
# print(F)
# print(pval)
print("F Score : ",F[np.argsort(clf.feature_importances_,)])
print("pValue : ",pval[np.argsort(clf.feature_importances_,)])

plt.show()

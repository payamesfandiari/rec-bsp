import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from matplotlib.cm import cmap_d


dataset_name = 'test/'
data_path = 'datasets_v1/{0}data'.format(dataset_name)
trueclass = 'datasets_v1/{0}trueclass'.format(dataset_name)
all_data = np.loadtxt(data_path, dtype='f')
true_labels = np.loadtxt(trueclass, dtype='i')


randomclass = 'datasets_v1/{0}random_class.{1}'.format(dataset_name,0)
train_labels = np.loadtxt(randomclass,dtype='i')
train_labels = train_labels[train_labels[:,1].argsort()]
train_data = all_data[train_labels[:,1],:]
ind = np.in1d(true_labels[:,1], train_labels[:,1], assume_unique=True)==False
test_data = all_data[ind,:]
test_labels = true_labels[ind]
train_labels = train_labels[:,0]
test_labels = test_labels[:,0]
ones = np.ones((train_data.shape[0],1))
train_data =  np.concatenate((train_data,ones),axis=1)
 
ones = np.ones((test_data.shape[0],1))
test_data =  np.concatenate((test_data,ones),axis=1)
rows,cols = train_data.shape
 
train_labels[train_labels==0] = -1
test_labels[test_labels==0] = -1
 
fig, ax = plt.subplots()
colors =  train_labels[:].tolist()
colors= ['b' if c==1 else 'r' for c in colors]
 
ax.scatter(train_data[:,0],train_data[:,1],c=colors)
# 
# xx = np.linspace(0,13)
# w = np.asarray([  0.1290543,  0.1770358 ])
# w0 = -0.9757066
# print(w,w0)
# yy = -(w[0]/w[1]) * xx - (w0/w[1])    
# p = ax.plot(xx,yy,'r-')[0]
 
xx = np.linspace(0,13)
w = np.asarray([0.9906883 , 0.1361494 ])
w0 = -4.7303963
print(w,w0)
yy = -(w[0]/w[1]) * xx - (w0/w[1])    
p = ax.plot(xx,yy,'r-')[0]
# 
w = np.asarray([0.7073375, -0.706876  ])
w0 =-0.7080298 
yy = -(w[0]/w[1]) * xx - (w0/w[1])    
print(yy)
p = ax.plot(xx,yy,'b-')[0]
 
w = np.asarray([-0.3149295, -0.949115 ])
w0 =3.0004835 
yy = -(w[0]/w[1]) * xx - (w0/w[1])    
print(yy)
p = ax.plot(xx,yy,'g-')[0]
 
# 
ax.set_ylim(0,8)
ax.set_xlim(0,8)
# 
# plot_colors = "rb"
# plot_step = 0.02
# X = train_data
# y = train_labels
# 
# clf = DecisionTreeClassifier().fit(X, y)
# 
# # Plot the decision boundary
# plt.subplot(1, 1,1)
# 
# x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
# y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
#                      np.arange(y_min, y_max, plot_step))
# 
# Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)
# cs = plt.contourf(xx, yy, Z)
# 
# 
# plt.axis("tight")
# 
# # Plot the training points
# for i, color in zip(range(2), plot_colors):
#     idx = np.where(y == i)
#     plt.scatter(X[idx, 0], X[idx, 1], c=color, label=train_labels[i])

# plt.axis("tight")




# for i in range(1,len(AllW)):
#     w = AllW[i,0:2]
#     w0 = AllW[i,2]
#     print(w,w0)
#     yy = -(w[0]/w[1]) * xx - (w0/w[1])        
#     p.set_ydata(np.asarray(yy))
# 
#     plt.pause(0.5)

plt.show()

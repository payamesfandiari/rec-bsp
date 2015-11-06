'''
Created on Nov 1, 2015

@author: payam
'''
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from time import time
import operator
from BSP_v2 import bsp
from collections import deque
from BinaryTree import BinaryTree


STANDARD = False

def predict(X,W):
    wx = np.sum(X*W,axis=1)       
    ywx = [1 if x > 0 else -1 for x in wx.ravel()]
    return ywx 
    

def objective(X,Y,W,C=1):
    wx = np.sum(X*W,axis=1)    
    ywx = Y*wx
    error = np.sum([x for x in ywx<=0]) / float(len(ywx))
    ywx[ywx > 0] = 0
    margin = np.sum(-ywx) / np.sqrt(np.dot(W,W))
#     print(error)
#     print(margin)    
    return (error ,margin,margin - C*error)

def partition (W,X,Y,side):
    WX  =np.sum(X*W,axis = 1)
    YWX = Y*WX
    left = side[WX < 0]
    right = side[WX >= 0]
    if left.size == 0 or right.size == 0:
        return None,None
    if np.sum([x for x in YWX[WX < 0]<=0]) == 0:
        left = None
    if np.sum([x for x in YWX[WX >= 0]<=0]) == 0:
        right = None
        
    return left,right


dataset_name = 'breast_cancer/'
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
if STANDARD:
#     scalar = StandardScaler().fit(train_data)
    scalar = MinMaxScaler().fit(train_data)
    train_data = scalar.transform(train_data)
    test_data = scalar.transform(test_data)


ones = np.ones((test_data.shape[0],1))
test_data =  np.concatenate((test_data,ones),axis=1)
test_labels[test_labels==0] = -1

all_w,all_obj,all_error,all_margin = bsp(train_data=train_data,train_labels=train_labels,C=1,ilsitr=100,ils_percent=0.9)
inds = np.argsort(all_obj.ravel())

# 
for i in range(10):
    error,margin,obj = objective(test_data, test_labels,all_w[inds[i],:])
    print("for W {0}: , the test error is : {1}({3}), the training error is : {2}".format(i,error,all_obj[inds[i]],all_error[inds[i]]))
#     print(all_w[inds[i]])

W_1 = all_w[inds[0]]
tree = BinaryTree()
tree.put(None, W_1)
ones = np.ones((train_data.shape[0],1))
X_x =  np.concatenate((train_data,ones),axis=1)
indicies = np.arange(len(train_labels))
left , right = partition(W=W_1, X=X_x, Y=train_labels,side = indicies)
left_key = 'l'
right_key = 'r'
levels = deque([(left,left_key),(right,right_key)])

while levels:
    (side,key) = levels.popleft()
    print("Looking at {0}".format(key))
    if side == None:
        tree.put(key,None)
        continue
    all_w,all_obj,all_error,all_margin = bsp(train_data=train_data[side,:],train_labels=train_labels[side],C=1,ilsitr=10)    
    inds = np.argsort(all_obj.ravel())
    print("Error for {0} W is {1}".format(key,all_error[inds[0]]))
    W = all_w[inds[0]]
    l,r = partition(W,X_x[side,:], train_labels[side],side)
    if l != None and r != None:
        if len(l)==1 and len(r) != 1:
            levels.append((r,key))
            continue
        elif len(r)==1 and len(l) != 1:
            levels.append((l,key))
            continue
        elif len(r)==1 and len(l) == 1:
            continue
    tree.put(key,W)
    levels.append((l,key+'l'))
    levels.append((r,key+'r'))

# tree.traverse()


# 
# Ws  = []
# while True:
# 
# 
# 
# 
# 
# 
# 
# 
# left_w,left_obj,left_error,left_margin = bsp(train_data=left,train_labels=left_labels,C=0.1,ilsitr=10)
# inds = np.argsort(left_obj.ravel())
# 
# 
# # for i in range(10):
# #     error,margin,obj = objective(test_data, test_labels,left_w[inds[i],:])
# #     print("for W {0}: , the test error is : {1}({3}), the training obj is : {2}".format(i,error,left_obj[inds[i]],left_error[inds[i]]))
# #     print(left_w[inds[i]])
# 
# print()
# W_l = left_w[inds[0]]
# out_l = predict(test_data, W=W_l)
# 
# right_w,right_obj,right_error,right_margin = bsp(train_data=right,train_labels=right_labels,C=0.1,ilsitr=10)
# inds = np.argsort(right_obj.ravel())
# 
# # 
# # for i in range(10):
# #     error,margin,obj = objective(test_data, test_labels,right_w[inds[i],:])
# #     print("for W {0}: , the test error is : {1}({3}), the training obj is : {2}".format(i,error,right_obj[inds[i]],right_error[inds[i]]))
# #     print(right_w[inds[i]])
# 
# 
# W_r = right_w[inds[0]]
# print()
# out_r = predict(test_data, W=W_r)
# 
# error = 0
# for i in range(len(test_labels)):
#     if np.sum(test_data[i]*W_1) > 0:
#         if np.sum(test_data[i]*W_r)>0 : 
#             if test_labels[i] < 0 :
#                 error += 1 
#         else:
#             if test_labels[i] > 0 :
#                 error += 1
#     else:
#         if np.sum(test_data[i]*W_l)>0 : 
#             if test_labels[i] < 0 :
#                 error += 1 
#         else:
#             if test_labels[i] > 0 :
#                 error += 1
#         
#         
# print(error)
# print(error / len(test_labels))



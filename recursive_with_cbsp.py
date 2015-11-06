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
import subprocess as sub
import write_bsp as wbsp
from BinaryTree import BinaryTree

STANDARD = False

def getW(out):
    readw =False
    readw0 = False
    w = []
    w0 = []
    for line in out.splitlines():                
        if 'End' in line:
            readw = True
            continue
        if readw:
            w.append(line.split())
            readw = False
            readw0 = True
            continue
        if readw0:
            w0.append(line.split())
            readw0 = False
            readw = False
            continue
    w = np.concatenate((np.asanyarray(w,dtype='f'),np.asanyarray(w0, dtype='f')),axis=1)
    return w

def predict(X,W):
    wx = np.sum(X*W,axis=1)       
    ywx = [1 if x > 0 else -1 for x in wx.ravel()]
    return ywx 
    
def bestW(W,X,Y):
    err = []
    for w in W:
        wx = np.sum(X*w,axis=1)    
        ywx = Y*wx
        error = np.sum([x for x in ywx<=0]) / float(len(ywx))
        err.append(error)
    err = np.asarray(err)
    inds = np.argsort(err)
    
    return W[inds[0]],err[inds[0]]
    
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


dataset_name = 'wall_follow/'
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

train_labels[train_labels==0] = -1
test_labels[test_labels==0] = -1
ones = np.ones((train_data.shape[0],1))
X_x =  np.concatenate((train_data,ones),axis=1)
ones = np.ones((test_data.shape[0],1))
test_data =  np.concatenate((test_data,ones),axis=1)

for kk in range(10):
    print("***",kk)
    bsp_file = "{0}.bsp".format(data_path)
    wbsp.write_bsp(bsp_file,train_data,train_labels)
    
    out = sub.check_output('./bsp {0} {1} {2} 100 1 .001 10 1 10000 1 10 {3}'.format(train_data.shape[0],train_data.shape[1],bsp_file,10000),shell=True)
    out = out.decode('ascii')
    # print(out)
    allW = getW(out)
    W,err = bestW(allW, X_x, train_labels) 
    original_predictions = predict(test_data, W)
    
    # 
    # all_w,all_obj,all_error,all_margin = bsp(train_data=train_data,train_labels=train_labels,C=1,ilsitr=100,ils_percent=0.9)
    # inds = np.argsort(all_obj.ravel())
    #  
    # # 
    # for i in range(10):
    #     error,margin,obj = objective(test_data, test_labels,all_w[inds[i],:])
    #     print("for W {0}: , the test error is : {1}({3}), the training error is : {2}".format(i,error,all_obj[inds[i]],all_error[inds[i]]))
    # #     print(all_w[inds[i]])
    #  
    # W_1 = all_w[inds[0]]
    tree = BinaryTree()
    tree.put(None, W)
    
    indicies = np.arange(len(train_labels))
    left , right = partition(W=W, X=X_x, Y=train_labels,side = indicies)
    left_key = 'l'
    right_key = 'r'
    levels = deque([(left,left_key),(right,right_key)])
     
    while levels:
        (side,key) = levels.popleft()
        print("Looking at {0}".format(key))
        if side == None:
            tree.put(key,None)
            continue
        wbsp.write_bsp(bsp_file,train_data[side,:],train_labels[side])
        out = sub.check_output('./bsp {0} {1} {2} 100 1 .001 10 1 10000 1 10 {3}'.format(len(train_data[side,:]),train_data.shape[1],bsp_file,10000),shell=True)
        out = out.decode('ascii')
        # print(out)
        allW = getW(out)
        W,err = bestW(allW, X_x[side,:], train_labels[side]) 
        
    #     all_w,all_obj,all_error,all_margin = bsp(train_data=train_data[side,:],train_labels=train_labels[side],C=1,ilsitr=10)    
    #     inds = np.argsort(all_obj.ravel())
#         print("Error for {0} W is {1}".format(key,err))
    #     W = all_w[inds[0]]
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
    
    predictions = []
    for x in test_data:
        node = tree.root
        lbl = 0
        while node.W != None:
            if np.sum(x*node.W) < 0:
                lbl = -1 
                node = node.left
            elif np.sum(x*node.W) >= 0:
                lbl = 1 
                node = node.right
    
        predictions.append(lbl)
    # tree.traverse()
#     print("True Labels for Test : {0}".format(test_labels))
#     print("Original BSP : {0}".format(original_predictions))
#     print("RBSP : {0}".format(predictions))
    print("Recursive Error : {0}".format((len(test_labels) - np.sum([test_labels==predictions]))/len(test_labels)))
    print("BSP Error {0}".format((len(test_labels) - np.sum([test_labels==original_predictions]))/len(test_labels)))

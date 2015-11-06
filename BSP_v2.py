'''
Created on Oct 28, 2015

@author: payam
'''

import numpy as np
import operator


DEBUG = False
WGT = False
np.set_printoptions(precision=7, linewidth=400, suppress=True)


def build_stump_1d(x,y,w):
    sorted_xyw = np.array(sorted(zip(x,y,w), key=operator.itemgetter(0)))
    xsorted = sorted_xyw[:,0]
    wy = sorted_xyw[:,1]*sorted_xyw[:,2]
    score_left = np.cumsum(wy)
    score_right = np.cumsum(wy[::-1])
    score = -score_left[0:-1:1] + score_right[-1:0:-1]
    Idec = np.where(xsorted[:-1]<xsorted[1:])[0]
    if len(Idec)>0:  # determine the boundary
        ind, maxscore = max(zip(Idec,abs(score[Idec])),key=operator.itemgetter(1))
        err = 0.5-0.5*maxscore # compute weighted error
        threshold = (xsorted[ind] + xsorted[ind+1])/2 # threshold
        s = np.sign(score[ind]) # direction of -1 -> 1 change
    else:  # all identical; todo: add random noise?
        err = 0.5
        threshold = 0
        s = 1
    return (err, threshold, s)

def objective(X,Y,W,C=1):
    WX = np.sum(X*W,axis=1)
    ywx = Y*WX
    error = np.sum([x for x in ywx<=0]) / float(len(ywx))
    ywx[ywx > 0] = 0
    margin = np.sum(ywx) / np.sqrt(np.dot(W,W))
    return (error ,margin,margin - C*error)

def normalize(W):
#     return W
    return W/np.sqrt(np.dot(W,W))


def bsp(train_data,train_labels,C=1,ils_percent=.6,ilsitr=100,ils_thresh=1.0e-4):
    
    ones = np.ones((train_data.shape[0],1))
    train_data =  np.concatenate((train_data,ones),axis=1)

    
    rows,cols = train_data.shape
    
    train_labels[train_labels==0] = -1
    w = normalize(np.random.rand(cols))     
    wx = np.sum(train_data*w,axis=1)
    best_error,best_margin,best_obj= objective(X=train_data, Y=train_labels, W=w, C=C)    
    prev_obj = float('-inf')
    prev_inner_obj = 0
    if DEBUG:
        print("Initial Values : objective : {0} , error : {1}, margin : {2}".format(best_obj,best_error,best_margin))
    ## CREATE A RANDOM PERMUTAITION OF COLS
    cols_permute = np.random.permutation(cols)

    itr = 0
    all_w = np.empty([ilsitr,cols])
    all_error = np.empty([ilsitr,1])
    all_margin = np.empty([ilsitr,1])
    all_obj = np.empty([ilsitr,1])
    
    
    if WGT :
        train_weights = np.ones([rows,1])
        train_weights[train_labels==1] = (np.sum([train_labels==1]) /rows)
        train_weights[train_labels==-1] = (np.sum([train_labels==-1]) /rows) 
    else:
        train_weights = np.ones([rows,1])/rows

    
    while(itr < ilsitr):
        
        while(abs(prev_obj-best_obj) > ils_thresh):
            prev_obj = best_obj       
            
            for j in cols_permute:       
        #         while(True):
                prev_inner_obj = best_obj           
                
                non_zero_ind_j = np.nonzero(train_data[:,j])  
                
                if(len(non_zero_ind_j[0]) < 5):
                    continue
                
                e,alpha,d=build_stump_1d(np.divide(wx[non_zero_ind_j[0]],train_data[non_zero_ind_j[0],j]), train_labels[non_zero_ind_j[0]], train_weights[non_zero_ind_j[0]])
                    #print(e,alpha,d)
                w[j] = w[j] - d*alpha
                    #print(w)
                error ,margin,best_obj = objective(X=train_data, Y=train_labels,W=w,C=C )    
        #         if(abs(overall_inner_obj-overall_obj) < 0.0001):
        #             break
                if best_obj <= prev_inner_obj :
                    w[j] = w[j] + d*alpha
                    best_obj = prev_inner_obj
                else:
                    best_error = error
                    best_margin = margin
                    if DEBUG:
                        print("W[{0}] was updated with {1}*{2}".format(j,d,alpha))
                        print("W = {0}".format(w))
                    
                w = normalize(w)
                wx = np.sum(train_data*w,axis=1) 
            if DEBUG:
                print("Previous Obj = {0}, Current : {1} ".format(prev_obj,best_obj))
        #     w = np.divide(w,np.sqrt(np.dot(w,w)))
    #     e,m,ob = objective(X=train_data, Y=train_labels, W=w)
    #     print("for W itr : {0}, error = {1}, margin = {2}, obj = {3}".format(itr,e,m,ob))
        if DEBUG:
            print("Final Values for itr : {6}: objective : {0}({1}) , error : {2}({3}), margin : {4}({5})".format(best_obj,best_obj,best_error,best_error,best_margin,best_margin,itr))
    #     print("")
    #     print("Final W for itr {1}: {0}".format(w,itr))
        all_w[itr] = w
        all_error[itr] = best_error
        all_margin[itr] = best_margin
        all_obj[itr] = best_obj
        cols_permute = np.random.permutation(cols)
        for i in range(0,int(ils_percent*cols)):
            w[cols_permute[i]] += np.random.rand(1)
        
        w = normalize(w)
        wx = np.sum(train_data*w,axis=1)
        best_error,best_margin,best_obj= objective(X=train_data, Y=train_labels, W=w,C=C)    
        prev_obj = float('-inf')
        prev_inner_obj = 0 
        itr+=1
    return (all_w,all_obj,all_error,all_margin)
    
    
    





    
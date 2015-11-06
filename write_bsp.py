'''
Created on Nov 3, 2015

@author: payam
'''


import numpy as np


def write_bsp(file_path,data,labels):
    f = open(file_path,'w')
    for x,y in zip(data,labels):
        print(y,end=' ', file=f)
        for d in x:
            print("{0} ".format(d),end=' ', file=f)
        print('',file=f)
    f.close()


def write_liblinear(file_path,data,labels):
    f = open(file_path,'w')
    for x,y in zip(data,labels):
        print(y,end=' ', file=f)
        i = 1
        for d in x:
            print("{0}:{1} ".format(i,d),end=' ', file=f)
            i += 1
        print('',file=f)
    f.close()
    
def write_arff(file_path,data,labels,name):
    f = open(file_path,'w')
    print("@RELATION {0}".format(name),file=f)
    for i in range(data.shape[1]):
        print("@ATTRIBUTE {0} NUMERIC".format(i),file=f)
    
    print("@ATTRIBUTE class {-1,1}",file=f)
    print("@DATA",file=f)
    
    for x,y in zip(data,labels):        
        for d in x:
            print("{0},".format(d),end='', file=f)
        print(y, file=f)
    f.close()
    
    

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
train_labels[train_labels==0] = -1
test_labels[test_labels==0] = -1


write_arff(data_path+".arff", train_data, train_labels, "wall_follow")
write_arff(data_path+"test.arff", test_data, test_labels, "wall_follow")


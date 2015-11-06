'''
Created on Nov 3, 2015

@author: payam
'''
import numpy as np 

class data(object):
    
    def __init__(self,data_path,true_path,STD = False):
        self.allData = None
        self.trueClass = None
        self.train = None
        self.test = None
        self.trainLabels = None
        self.testLabels = None
        self.STD = STD
        self.__init(data_path,true_path)
        
    def __init(self,dp,tp):
        self.allData = np.loadtxt(dp, dtype='f')
        self.trueClass = np.loadtxt(tp, dtype='i')
    
    def loadRC(self,rc_path):
        train_labels = np.loadtxt(rc_path,dtype='i')
        train_labels = train_labels[train_labels[:,1].argsort()]
        self.train = self.allData[train_labels[:,1],:]
        ind = np.in1d(self.trueClass[:,1], train_labels[:,1], assume_unique=True)==False
        self.test = self.allData[ind,:]
#         test_labels = true_labels[ind]
        train_labels = train_labels[:,0]
#         test_labels = test_labels[:,0]

class formatter(object):
    '''
    classdocs
    '''


    def __init__(self, params):
        '''
        Constructor
        '''
        
        
        
        
        
        
        

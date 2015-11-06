'''
Created on Nov 3, 2015

@author: payam
'''
class Node(object):
    '''
    classdocs
    '''

    def __init__(self,key,W,left=None,right=None,parent=None):
        '''
        Constructor
        '''
        self.key = key
        self.W = W
        self.left = left
        self.right = right
        self.parent = parent
    
    def hasLeftChild(self):
        return self.left
    
    def hasRightChild(self):
        return self.right
    
    def isLeftChild(self):
        return self.parent and self.parent.leftChild == self

    def isRightChild(self):
        return self.parent and self.parent.rightChild == self

    def isRoot(self):
        return not self.parent

    def isLeaf(self):
        return not (self.rightChild or self.leftChild)
    def __iter__(self):
        if self:
            yield self.W
            if self.hasLeftChild():
                for elem in self.left:
                    yield elem
            if self.hasRightChild():
                for elem in self.right:
                    yield elem


class BinaryTree(object):
    '''
    classdocs
    '''

    def __init__(self):
        self.root = None
        self.size = 0 
        
    
    def put(self,key,W):
        if self.root:
            self.__put(key, W, self.root)
        else:
            self.root = Node(key,W)
        self.size += 1
            
            
    
    def __put(self,key,W,currentNode):
        if key[0] == 'l':
            if currentNode.hasLeftChild():
                self.__put(key[1:], W, currentNode.left)
            else:
                currentNode.left = Node(key,W,parent=currentNode)
        elif key[0]== 'r':
            if currentNode.hasRightChild():
                self.__put(key[1:], W, currentNode.right)
            else:
                currentNode.right = Node(key,W,parent=currentNode)
    
    
    def __setitem__(self,k,w):
        self.put(key=k, W=w)
    
    def getroot(self):
        return self.root
        
    def traverse(self):    
        thisLevel = [self.root]
        while thisLevel:
            nextLvl = list()
            for n in thisLevel:
                print(n.key,"-->",n.W,end=' ')
                if n.hasLeftChild():
                    nextLvl.append(n.left)
                if n.hasRightChild():
                    nextLvl.append(n.right)
            thisLevel = nextLvl
            print()
        
    
        
    




import matplotlib.pyplot as mat
import numpy as np
import time

#KNN search algorithm using Numpy, using euclidean distances
def k_nearest_neighbors(X, Y, k):
    return np.argsort( np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=2))[:,0:k]

#Another function to Compute euclidean distance using Numpy
def k_nearest_neighbour_2(Test,Training,k):
    x_square = np.sum(Training ** 2, axis=1, keepdims=True)
    y_square = np.sum(Test ** 2, axis=1)
    xy = np.dot(Training, Test.T)
    dist = np.sqrt(x_square - 2 * xy + y_square)
    return dist


#Decide the classification of the k nearest neighbours and plot them
def Classify(TrainingSet,TestSet,k):
    ClassIndexes = k_nearest_neighbors(TestSet[:,0:2],TrainingSet[:,0:2],k)
    SuccessComparision = (np.sum(TrainingSet[ClassIndexes, 2],axis=1)>0) == (TestSet[:,2]>0)
    return(np.sum(SuccessComparision)/TestSet.shape[0])

#Function to load the data
def LoadData(Filename):
    data = np.loadtxt(Filename, dtype=np.object, comments='#', delimiter=None)
    OriginalData = data[:, 0:3].astype(np.float)
    return OriginalData

#Load the training and test data
TestData = LoadData('data2-test.dat')
TrainingData = LoadData('data2-train.dat')

#Perform the knn test for each individual k
for k in range(1,20):
    start = time.clock()
    print("\n Accuracy Rate For k = ",k," Nearest Neighbours = ",Classify(TrainingData[:,:],TestData[:,:],k))
    end = time.clock()
    print("\n Execution Time= ",(end - start))


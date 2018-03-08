import matplotlib.pyplot as mat
import numpy as np
import time as tm
import sys

#Function to get the euclidean distance between 2 points
def EuclideanDistance(X,Y):
    return (( (X[0]-Y[0])**2 + (X[1]-Y[1])**2)**0.5 )

#Various Functions to split the data along a particular dimension d
#1. Split along dimension d at median
def SplitAlongMedian(X,d):
    median_d = np.median(X[:,d])
    Left  = X[X[:, d] <= median_d,:]
    Right = X[X[:, d] >  median_d,:]

    if Left.shape[0] >= 1:
       index = np.argmax(Left[:,d])
       median_element = Left[index, :]
       Left = np.delete(Left, (index), axis=0)
    elif Right.shape[0] >= 1:
       index = np.argmin(Right[:, d])
       median_element = Right[index, :]
       Right = np.delete(Right, (index), axis=0)

    return median_element,Left,Right

#2. Split along dimension d at mean
def SplitMidPoint(X,d):
    midpoint = ( np.max(X[:,d]) + np.min(X[:,d]) ) / 2
    Left  = X[X[:,d] <= midpoint,:]
    Right = X[X[:,d] >  midpoint,:]

    if Left.shape[0] >= 1:
       index = np.argmax(Left[:, d])
       midpoint_element = Left[index, :]
       Left = np.delete(Left, (index), axis=0)
    elif Right.shape[0] >= 1:
       index = np.argmin(Right[:, d])
       midpoint_element = Right[index, :]
       Right = np.delete(Right, (index), axis=0)

    return midpoint_element, Left, Right

#Various functions to select split dimetions
# 1.Dimension selection based on higher variance
def DimensionAlongHigherVariance(X,param1=0,param2=0):
    return np.argmax( np.var(X[:,0:2],axis=0) )

# 2.Dimension selection based on X and Y alternation, the mod function : Round Robin Fashion
def DimensionAlongRoundRobinXY(X,K,depth):
    return (depth % K)

# Class to rrepresent a tree node
class TreeNode:
      def __init__(self,Point,Parent,SplitDimension,PointX,PointY):
          self.Point = Point
          self.Parent = Parent
          self.SplitDimension = SplitDimension
          self.PointX = PointX
          self.PointY = PointY
          self.left = None
          self.right = None

# Class to construct and manipulate a KD Tree
class KD_Tree:
      #Read the points from file and set up the list of points
      def __init__(self,FileName,FunctionSplit,SelectSplitDimention,K,mat):
          self.Data = self.LoadData(FileName)
          self.K = K
          self.FunctionSplit = FunctionSplit
          self.SelectSplitDimention = SelectSplitDimention
          self.mp = mat
          self.Leafx=[]
          self.Leafy=[]
          self.PartitionLines = []
          self.TreeNode = self.CreateKDTree\
                          ( self.Data,None,
                            self.mp.xlim(xmin=np.min(self.Data[:, 0]) - 10, xmax=np.max(10 + self.Data[:, 0])),
                            self.mp.ylim(ymin=np.min(self.Data[:, 1]) - 10, ymax=np.max(10 + self.Data[:, 1])),
                            0
                          )

      def LoadData(self,Filename):
          data = np.loadtxt(Filename, dtype=np.object, comments='#', delimiter=None)
          OriginalData = data[:, 0:3].astype(np.float)
          return OriginalData

      #Get the dimensions of the partitioned space
      def ModifyBox(self,SplitPoint,RangeX,RangeY,dimension):
          if dimension==0:
             return [SplitPoint[0],SplitPoint[0]],[RangeY[0],RangeY[1]],(RangeX[0],SplitPoint[0]),(SplitPoint[0],RangeX[1]),RangeY,RangeY
          if dimension==1:
             return [RangeX[0],RangeX[1]],[SplitPoint[1],SplitPoint[1]],RangeX,RangeX,(RangeY[0],SplitPoint[1]),(SplitPoint[1],RangeY[1])

      #Create a KD Tree from the K-D data points
      # Axis = 0 implies along x-axis, 1 implies along y-axis, and so on in case of more dimentions
      def CreateKDTree(self,Data,Parent,RangeX,RangeY,depth=0):
          if Data.shape[0] == 0: # Base Case : Empty Data
              return None
          if Data.shape[0] == 1: # Base case : When only one point is there in the list: No need to partition
             self.Leafx.append(Data[0,0])
             self.Leafy.append(Data[0,1])
             return TreeNode(Data[0,:],Parent,-1,(),())

          SplitDimension = (self.SelectSplitDimention)(Data,self.K,depth) #Selecting the dimension of splitting
          SplitPoint,SplitLeft,SplitRight = (self.FunctionSplit)(Data,SplitDimension) #Selecting the splitPoint and the Splitted parts
          #Get the new measurements for partitioning
          PointX,PointY,RangeX1,RangeX2,RangeY1,RangeY2 = self.ModifyBox(SplitPoint,RangeX,RangeY,SplitDimension)
          #Partition plot
          self.PartitionLines.append((PointX,PointY))
          #Split the plane and create subtrees
          Node = TreeNode(SplitPoint,Parent,SplitDimension,PointX,PointY)  # Create a new node , containing this splitpoint and the dimension it is going to split
          Node.left = self.CreateKDTree(SplitLeft,Node,RangeX1,RangeY1,depth+1)
          Node.right = self.CreateKDTree(SplitRight,Node,RangeX2,RangeY2,depth+1)
          return Node

      #Display the plotted KD Tree
      def ShowKDSpace(self,Info,flag):
          self.mp.title(Info)
          self.mp.legend()
          for Pair in self.PartitionLines:
              self.mp.plot(Pair[0], Pair[1], 'g', linewidth=0.5)
          self.mp.plot(self.Data[:, 0], self.Data[:, 1], 'bo',markersize=1.5,label='Partitioning Points')
          self.mp.plot(self.Leafx,self.Leafy,'ro',markersize=1.5,label='Leaf Points')
          if flag == True:
             self.mp.savefig("Plot-" + Info + ".pdf", facecolor='w', edgecolor='w',
                                   papertype=None, format='pdf', transparent=False,
                                   bbox_inches='tight', pad_inches=0.1)
             self.mp.show()

      # Display the plotted KD Tree
      def ShowKNNProgress(self, Info, ProgressLines,SphereRadius,Point,flag):
          fig, ax = mat.subplots()
          mat.xlim(xmin=np.min(self.Data[:, 0]) - 10, xmax=np.max(10 + self.Data[:, 0]))
          mat.ylim(ymin=np.min(self.Data[:, 1]) - 10, ymax=np.max(10 + self.Data[:, 1]))
          mat.title(Info)
          mat.legend()
          for Pair in self.PartitionLines:
              mat.plot(Pair[0], Pair[1], 'g', linewidth=0.5)
          for Pair in ProgressLines:
              mat.plot(Pair[0], Pair[1], 'r', linewidth=0.5)
          mat.plot([Point[0]],[Point[1]],'o',color='magenta',markersize=2.5)
          mat.plot(self.Data[:, 0], self.Data[:, 1], 'bo', markersize=1.5, label='Partitioning Points')
          mat.plot(self.Leafx, self.Leafy, 'ro', markersize=1.5, label='Leaf Points')
          c = mat.Circle((Point[0],Point[1]), SphereRadius, color='g', fill=False)
          ax.add_artist(c)
          if flag == True:
             mat.show()

      #Function for test purposes...stack based DFS traversal
      def StackTraversal(self,Node,Point,List):
          CurrentDistance = 999999999999
          CurrentNode = Node
          BestNode = Node
          Stack=[]
          done = False
          while done == False:
                if CurrentNode is not None:
                   Stack.append(CurrentNode)
                   CurrentNode = CurrentNode.left
                else:
                   if len(Stack) > 0 :
                      CurrentNode = Stack.pop()
                      d = EuclideanDistance(CurrentNode.Point,Point)
                      if d < CurrentDistance:
                          CurrentDistance = d
                          BestNode = CurrentNode
                      CurrentNode = CurrentNode.right
                   else:
                       done=True
          return BestNode,CurrentDistance,[]

      #Perform DFS traversal of the KD Tree
      def DFS(self,Node,Point,List):
          if Node is not None:
             #Get Euclidean distance
             CurrentDistance = EuclideanDistance(Node.Point,Point)
             #Check if Node is a leaf Node
             if (Node.left is None) and (Node.right is None) and (Node.SplitDimension == -1):
                # Return the distance and the currentNode
                # The leaf node has been found which is the partition within which the Query Point lies
                # CurrentDistance contains the Radius of the HyperSphere, around which the HyperRectangle Test will be done
                return Node,CurrentDistance,List
             #Append the Co-ordinates of this point for plotting partition
             List.append((Node.PointX, Node.PointY))

             #Check for backtracking the subtrees if any intersection is found of the hyperrectangle with the hypersphere
             if Node.Point[Node.SplitDimension] >= Point[Node.SplitDimension] and Node.left is not None :
                CurrentBest,BestDistance,x = self.DFS(Node.left,Point,List)
                if Node.right is not None and ( abs(Node.Point[0] - Point[0]) <= BestDistance or abs(Node.Point[1] - Point[1]) <= BestDistance):
                   SphereNode, SphereDistance, x = self.DFS(Node.right, Point, x)
                   if SphereDistance < BestDistance:
                      BestDistance = SphereDistance
                      CurrentBest  = SphereNode

             elif Node.right is not None:
                CurrentBest,BestDistance,x = self.DFS(Node.right,Point,List)
                if Node.left is not None and ( abs(Node.Point[0] - Point[0]) <= BestDistance or abs(Node.Point[1] - Point[1]) <= BestDistance):
                   SphereNode, SphereDistance, x = self.DFS(Node.left, Point, x)
                   if SphereDistance < BestDistance:
                      BestDistance = SphereDistance
                      CurrentBest = SphereNode

             if CurrentDistance < BestDistance:
                CurrentBest = Node
                BestDistance = CurrentDistance

          return CurrentBest,BestDistance,x


      #Perform KNN on a test point by traversing the KD Tree created
      def PerFormKNN(self,testFile):
          TestPass=0.0
          TotalCases =0.0
          TestData = self.LoadData(testFile)
          start = tm.clock()
          print("\n ")


          for i in range(0,TestData.shape[0]):
              Node = self.TreeNode
              P = TestData[i,:]
              NearestPoint,ShortestDistance,ProgressLines = self.DFS(Node,P,[])
              #print(NearestPoint,ShortestDistance,ProgressLines)
              #self.ShowKNNProgress('1-NN in progress', ProgressLines,ShortestDistance,P,True)
              TotalCases += 1.0
              #print("\n ",TestData[i,2],NearestPoint.Point[2])
              if TestData[i,2] == NearestPoint.Point[2]:
                 TestPass += 1.0
          end = tm.clock()
          return (TestPass/TotalCases)*100,(end-start)


T1 = KD_Tree('data2-train.dat',SplitAlongMedian,DimensionAlongRoundRobinXY,2,mat)
T1.ShowKDSpace('Median Split + Dimension Along Round Robin',True)
accuracy,time = T1.PerFormKNN('data2-test.dat')
print("\nAccuracy on Tree1 = ",accuracy)
print("\nTime taken for Tree1 = ",time)

T2 = KD_Tree('data2-train.dat',SplitMidPoint,DimensionAlongRoundRobinXY,2,mat)
T2.ShowKDSpace('MidPoint Split + Dimension Along Round Robin',True)
accuracy,time = T2.PerFormKNN('data2-test.dat')
print("\nAccuracy on Tree2 = ",accuracy)
print("\nTime taken for Tree2 = ",time)

T3 = KD_Tree('data2-train.dat',SplitAlongMedian,DimensionAlongHigherVariance,2,mat)
T3.ShowKDSpace('Median Split + Dimension Along Higher Variance',True)
accuracy,time = T3.PerFormKNN('data2-test.dat')
print("\nAccuracy on Tree3 = ",accuracy)
print("\nTime taken for Tree3 = ",time)

T4 = KD_Tree('data2-train.dat',SplitMidPoint,DimensionAlongHigherVariance,2,mat)
T4.ShowKDSpace('MidPoint Split + Dimension Along HigherVariance',True)
accuracy,time = T4.PerFormKNN('data2-test.dat')
print("\nAccuracy on Tree4 = ",accuracy)
print("\nTime taken for Tree4 = ",time)

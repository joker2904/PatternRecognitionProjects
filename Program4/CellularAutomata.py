import numpy as np
import scipy as sp

def invert(x):
    if x == 0:
        return 1
    else:
        return x

#Return the dot product of PseudoInverse and y by naive method
def NaiveComputeWeight(X,Y):
    return np.dot( np.dot(np.linalg.inv(np.dot(X.T, X)), X.T),Y)

### SUBTASK II
## Class to construct the combinatorial products
class GenerateProducts:
      def __init__(self,m):
          self.m = m
          #Generate the list in string format and write it to a file for Log purposes
          self.ProductSet = []
          self.GetProductCombinationMatrix_ListFormat(m, [], 1)
          self.PrintStringFormat()

          #Generate the Product matrix of 2^m combinations of products
          self.ProductMatrix = [[0]*m]
          self.GetProductCombinationMatrix_MatrixFormat(m, [0]*m, 0,0,1,self.ProductMatrix)
          self.ProductMatrix = np.matrix(self.ProductMatrix)

          self.X = [[-1]*m]
          self.GetProductCombinationMatrix_MatrixFormat(m, [-1]*m, 0,-1,1,self.X)
          self.X = np.matrix(self.X)

      #Return a product matrix of each combinat
      def MakeProductMatrix(self,binary1,binary2):
          a = []
          for Row in self.ProductSet:
              Product = [binary1] * self.m
              for j in Row:
                  Product[j-1] = binary2
              a.append(Product)
          a = np.array(a)
          return a

      #Return combinatorial product
      def ReturnCombinatorialProduct(self,xVector):
          auxiliary = np.multiply(self.ProductMatrix,xVector)
          f = np.vectorize(invert)
          auxiliary = f(auxiliary)
          auxiliary = np.prod(auxiliary,axis=1).reshape((1,auxiliary.shape[0]))
          return auxiliary

      def ConstructFeatureMatrix(self):
          a = self.ReturnCombinatorialProduct(self.X[0])
          for i in range(1,len(self.X)):
              a = np.vstack((a,self.ReturnCombinatorialProduct(self.X[i])))
          return a

      #Return the combination in List OF Indexes Format
      def GetProductSubset(self):
          return self.ProductMatrix,self.X

      #Return all the combinations in string Format
      def PrintStringFormat(self):
          fileio = open("Subsets.txt","w+")
          fileio.write("\n--------------------------------")
          fileio.write('\n | 1')
          for Products in self.ProductSet:
              row = '\n | '
              if len(Products) == 0:
                 row += '1'
              else:
                 for i in Products:
                     row += str('x'+str(i)+' ')
          fileio.write(row)
          fileio.write("\n--------------------------------")
          fileio.close()

      #Generate all the combinations
      def GetProductCombinationMatrix_ListFormat(self,NumberOfVariables,ProductSet,CurrentIndex):
          for i in range(CurrentIndex,NumberOfVariables+1): # Loop invariant : CurrentIndex <= NumberOfVariables
              if CurrentIndex == NumberOfVariables+1:
                 return
              ProductSet.append(i)
              self.ProductSet.append(ProductSet[:])
              self.GetProductCombinationMatrix_ListFormat(NumberOfVariables,ProductSet,i+1)
              ProductSet.pop()


      def GetProductCombinationMatrix_MatrixFormat(self,NumberOfVariables,ProductVector,CurrentIndex,binaryValue1,binaryValue2,Final):
          for i in range(CurrentIndex,NumberOfVariables): # Loop invariant : CurrentIndex <= NumberOfVariables
              if CurrentIndex == NumberOfVariables:
                 return
              ProductVector[i] = binaryValue2
              Final.append(ProductVector[:])
              self.GetProductCombinationMatrix_MatrixFormat(NumberOfVariables,ProductVector,i+1,binaryValue1,binaryValue2,Final)
              ProductVector[i] = binaryValue1

P = GenerateProducts(3)
FeatureDesignMatrix = P.ConstructFeatureMatrix()
print("\n FeatureDesignMatrix :: \n",FeatureDesignMatrix)



#### BASIC CALCULATION :: SUBTASK I
#Calculate for rule 110
X = np.matrix([(1.0, +1.0, +1.0, +1.0),
              (1.0, +1.0, +1.0, -1.0),
              (1.0, +1.0, -1.0, +1.0),
              (1.0, +1.0, -1.0, -1.0),
              (1.0, -1.0, +1.0, +1.0),
              (1.0, -1.0, +1.0, -1.0),
              (1.0, -1.0, -1.0, +1.0),
              (1.0, -1.0, -1.0, -1.0)])

y = np.matrix([-1.0,+1.0,+1.0,+1.0,-1.0,+1.0,+1.0,-1.0]).T
w_star_basic = NaiveComputeWeight(X,y)
w_start_pinv = np.dot(np.linalg.pinv(X),y)
y_basic = np.dot(X,w_star_basic)
y_pinv = np.dot(X,w_start_pinv)

w_feature = NaiveComputeWeight(FeatureDesignMatrix,y)
y_latest = np.dot(FeatureDesignMatrix,w_feature)

print("\n For Rule 110 :")
print("\n y = ",y.T)
print("\n w_basic = ",w_star_basic.T)
print("\n w_pinv = ",w_start_pinv.T)
print("\n y_basic = ",y_basic.T)
print("\n y_pinv = ",y_pinv.T)
print("\n y_latest = ",y_latest.T)

#Calculate for rule 126
X = np.matrix([(1.0, +1.0, +1.0, +1.0),
              (1.0, +1.0, +1.0, -1.0),
              (1.0, +1.0, -1.0, +1.0),
              (1.0, +1.0, -1.0, -1.0),
              (1.0, -1.0, +1.0, +1.0),
              (1.0, -1.0, +1.0, -1.0),
              (1.0, -1.0, -1.0, +1.0),
              (1.0, -1.0, -1.0, -1.0)])

y = np.matrix([-1.0,+1.0,+1.0,+1.0,+1.0,+1.0,+1.0,-1.0]).T
w_star_basic = NaiveComputeWeight(X,y)
w_start_pinv = np.dot(np.linalg.pinv(X),y)
y_basic = np.dot(X,w_star_basic)
y_pinv = np.dot(X,w_start_pinv)
w_feature = NaiveComputeWeight(FeatureDesignMatrix,y)
y_latest = np.dot(FeatureDesignMatrix,w_feature)
print("\n For Rule 126 :")
print("\n y = ",y.T)
print("\n w_basic = ",w_star_basic.T)
print("\n w_pinv = ",w_start_pinv.T)
print("\n y_basic = ",y_basic.T)
print("\n y_pinv = ",y_pinv.T)
print("\n y_latest = ",y_latest.T)


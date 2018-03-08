import numpy as np
import matplotlib.pyplot as mp
import numpy.polynomial as pol

#Return the dot product of PseudoInverse and y by naive method
def NaiveComputeWeight(X,Y):
    return np.dot( np.dot(np.linalg.inv(np.dot(X.T, X)), X.T),Y)

#Naive method to get VanderMond Matrix
def VanderMondMatrix(degree,X):
    Matrix = np.array([1] * X.shape[0]).reshape(X.shape[0],1)
    for d in range(1,degree+1):
        Matrix = np.hstack((Matrix, pow(X,d).reshape(X.shape[0],1)))
    return(Matrix)

#Load the data
def LoadData():
    # read data as 2D array of data type 'object'
    data = np.loadtxt('whData.dat', dtype=np.object, comments='#', delimiter=None)

    # read height and weight data into 2D array (i.e. into a matrix)
    OriginalData = data[:, 0:2].astype(np.float)
    # read gender data into 1D array (i.e. into a vector)
    Gender = data[:, 2]

    # removing negative and zeros from both columns
    X = OriginalData[OriginalData[:, 1] > 0, :]
    X = X[X[:, 0] > 0, :]

    # Get the list of outliers
    Outliers = OriginalData[OriginalData[:, 1] > 0, :]
    Outliers = Outliers[Outliers[:, 0] < 0, :]
    return X,Outliers[:,-1]

Sample,Outliers = LoadData()

fileio = open("Calculations.txt","w+")

##############################################################################################
###### NUMPY pinverse and Vandermonde ##############
fileio.write("\n\n\n :::::::::::::::::: Numpy Pinv Calculations ::::::::::::::::::")
mp.xlim(xmin=140,xmax=210)
mp.ylim(ymin=30,ymax=150)
mp.grid(True)
mp.xlabel('Height ( in cm )----->')
mp.ylabel('Weight (in kg)----->')
x = np.linspace(100,210,200)
mp.title('Numpy Pinv Fit')
mp.plot(Sample[:,1],Sample[:,0],'o',color='black')

VanderMondeX1 = pol.polynomial.polyvander(Sample[:,1],1)
Weight1 = np.dot( np.linalg.pinv(VanderMondeX1) , Sample[:,0])
fileio.write("\n Weight Matrix for Degree 1 polynomial="+str(Weight1))
x1 = pol.polynomial.polyvander(x,1)
y1 = np.dot( x1, np.reshape(Weight1,(Weight1.shape[0],1))  )
mp.plot(x1,y1,'green',label = 'deg=1')
OutlierMatrix = pol.polynomial.polyvander(Outliers,1)
Outlier_Weights = np.dot( OutlierMatrix , np.reshape(Weight1,(Weight1.shape[0],1)) )
mp.plot(np.reshape(Outliers,(1,3,)),np.reshape(Outlier_Weights,(1,3)),'o','green',label = "Outliers")
fileio.write( "\n\n Calculated Outlier Weights = "+ str(Outlier_Weights) )
fileio.write( "\n\n Outlier Heights = "+str(Outliers))

VanderMondeX5 = pol.polynomial.polyvander(Sample[:,1],5)
Weight5 = np.dot( np.linalg.pinv(VanderMondeX5) , Sample[:,0])
fileio.write("\n Weight Matrix for Degree 5 polynomial="+str(Weight5))
x5 = pol.polynomial.polyvander(x,5)
y5 = np.dot( x5, np.reshape(Weight5,(Weight5.shape[0],1))  )
mp.plot(x5,y5,'blue',label = "deg=5")
OutlierMatrix = pol.polynomial.polyvander(Outliers,5)
Outlier_Weights = np.dot( OutlierMatrix , np.reshape(Weight5,(Weight5.shape[0],1)) )
mp.plot(np.reshape(Outliers,(1,3,)),np.reshape(Outlier_Weights,(1,3)),'o','blue',label = "Outliers")
fileio.write( "\n\n Calculated Outlier Weights = "+ str(Outlier_Weights) )
fileio.write( "\n\n Outlier Heights = "+str(Outliers))

VanderMondeX10 = pol.polynomial.polyvander(Sample[:,1],10)
Weight10 = np.dot( np.linalg.pinv(VanderMondeX10) , Sample[:,0])
fileio.write("\n Weight Matrix for Degree 10 polynomial="+str(Weight10))
x10 = pol.polynomial.polyvander(x,10)
y10 = np.dot( x10, np.reshape(Weight10,(Weight10.shape[0],1))  )
mp.plot(x10,y10,'red',label = "deg=10")
OutlierMatrix = pol.polynomial.polyvander(Outliers,10)
Outlier_Weights = np.dot( OutlierMatrix , np.reshape(Weight10,(Weight10.shape[0],1)) )
mp.plot(np.reshape(Outliers,(1,3,)),np.reshape(Outlier_Weights,(1,3)),'o','red',label = "Outliers")
fileio.write( "\n\n Calculated Outlier Weights = "+ str(Outlier_Weights) )
fileio.write( "\n\n Outlier Heights = "+str(Outliers))
mp.legend()
mp.show()

######################################################################
###### Direct Computation and Vandermonde
fileio.write("\n :::::::::::::::::: Naive Calculations( Basic one) ::::::::::::::::::")
mp.xlim(xmin=140,xmax=210)
mp.ylim(ymin=30,ymax=150)
mp.grid(True)
mp.xlabel('Height ( in cm )----->')
mp.ylabel('Weight (in kg)----->')
x = np.linspace(100,210,200)
mp.title('Direct Computation Fit')
mp.plot(Sample[:,1],Sample[:,0],'o',color='black')

VanderMondeX1 = pol.polynomial.polyvander(Sample[:,1],1)
Weight1 = NaiveComputeWeight(VanderMondeX1, Sample[:,0])
fileio.write("\n Weight Matrix for degree 1 polynomial="+str(Weight1))
x1 = pol.polynomial.polyvander(x,1)
y1 = np.dot( x1, np.reshape(Weight1,(Weight1.shape[0],1))  )
mp.plot(x1,y1,'blue')
OutlierMatrix = pol.polynomial.polyvander(Outliers,1)
Outlier_Weights = np.dot( OutlierMatrix , np.reshape(Weight1,(Weight1.shape[0],1)) )
mp.plot(np.reshape(Outliers,(1,3,)),np.reshape(Outlier_Weights,(1,3)),'o','blue',label = "Outliers")
fileio.write( "\n\n Calculated Outlier Weights = "+ str(Outlier_Weights) )
fileio.write( "\n\n Outlier Heights = "+str(Outliers))

VanderMondeX5 = pol.polynomial.polyvander(Sample[:,1],5)
Weight5 = NaiveComputeWeight(VanderMondeX5 , Sample[:,0])
fileio.write("\n Weight Matrix for degree 5 polynomial ="+str(Weight5))
x5 = pol.polynomial.polyvander(x,5)
y5 = np.dot( x5, np.reshape(Weight5,(Weight5.shape[0],1))  )
mp.plot(x5,y5,'green')
OutlierMatrix = pol.polynomial.polyvander(Outliers,5)
Outlier_Weights = np.dot( OutlierMatrix , np.reshape(Weight5,(Weight5.shape[0],1)) )
mp.plot(np.reshape(Outliers,(1,3,)),np.reshape(Outlier_Weights,(1,3)),'o','green',label = "Outliers")
fileio.write( "\n\n Calculated Outlier Weights = "+ str(Outlier_Weights) )
fileio.write( "\n\n Outlier Heights = "+str(Outliers))

VanderMondeX10 = pol.polynomial.polyvander(Sample[:,1],10)
Weight10 = NaiveComputeWeight(VanderMondeX10 , Sample[:,0])
fileio.write("\n Weight Matrix for degree 10 polynomial ="+str(Weight10))
x10 = pol.polynomial.polyvander(x,10)
y10 = np.dot( x10, np.reshape(Weight10,(Weight10.shape[0],1))  )
mp.plot(x10,y10,'orange')
OutlierMatrix = pol.polynomial.polyvander(Outliers,10)
Outlier_Weights = np.dot( OutlierMatrix , np.reshape(Weight10,(Weight10.shape[0],1)) )
mp.plot(np.reshape(Outliers,(1,3,)),np.reshape(Outlier_Weights,(1,3)),'o','orange',label = "Outliers")
fileio.write( "\n\n Calculated Outlier Weights = "+ str(Outlier_Weights) )
fileio.write( "\n\n Outlier Heights = "+str(Outliers))
mp.legend(loc='Best')
mp.show()

##########################################################################
###### NUMPY leastsquare and Vandermonde ##############

fileio.write("\n :::::::::::::::::: Numpy LEastSquare Calculations ::::::::::::::::::")
mp.xlim(xmin=140,xmax=210)
mp.ylim(ymin=30,ymax=150)
mp.xlabel('Height ( in cm )----->')
mp.ylabel('Weight (in kg)----->')
mp.grid(True)
x = np.linspace(100,210,200)
mp.title('Numpy LeastSquare Fit')
mp.plot(Sample[:,1],Sample[:,0],'o',color='black')

VanderMondeX1 = pol.polynomial.polyvander(Sample[:,1],1)
Weight1s = np.linalg.lstsq( VanderMondeX1 , Sample[:,0])[0]
fileio.write("\n Weight Matrix for degree 1 polynomial ="+str(Weight1s))
x1 = pol.polynomial.polyvander(x,1)
y1 = np.dot( x1, np.reshape(Weight1s,(Weight1s.shape[0],1))  )
mp.plot(x1,y1,'blue',label='d=1')
OutlierMatrix = pol.polynomial.polyvander(Outliers,1)
Outlier_Weights = np.dot( OutlierMatrix , np.reshape(Weight1s,(Weight1s.shape[0],1)) )
mp.plot(np.reshape(Outliers,(1,3,)),np.reshape(Outlier_Weights,(1,3)),'o','blue',label = "Outliers")
fileio.write( "\n\n Calculated Outlier Weights = "+ str(Outlier_Weights) )
fileio.write( "\n\n Outlier Heights = "+str(Outliers))

VanderMondeX5 = pol.polynomial.polyvander(Sample[:,1],5)
Weight5s = np.linalg.lstsq( VanderMondeX5 , Sample[:,0])[0]
fileio.write("\n Weight Matrix for degree 5 polynomial ="+str(Weight5s))
x5 = pol.polynomial.polyvander(x,5)
y5 = np.dot( x5, np.reshape(Weight5s,(Weight5s.shape[0],1))  )
mp.plot(x5,y5,'green',label='d=5')
OutlierMatrix = pol.polynomial.polyvander(Outliers,5)
Outlier_Weights = np.dot( OutlierMatrix , np.reshape(Weight5s,(Weight5s.shape[0],1)) )
mp.plot(np.reshape(Outliers,(1,3,)),np.reshape(Outlier_Weights,(1,3)),'o','green',label = "Outliers")
fileio.write( "\n\n Calculated Outlier Weights = "+ str(Outlier_Weights) )
fileio.write( "\n\n Outlier Heights = "+str(Outliers))

VanderMondeX10 = pol.polynomial.polyvander(Sample[:,1],10)
Weight10s = np.linalg.lstsq( VanderMondeX10 , Sample[:,0])[0]
fileio.write("\n Weight Matrix for degree 10 polynomial ="+str(Weight10s))
x10 = pol.polynomial.polyvander(x,10)
y10 = np.dot( x10, np.reshape(Weight10s,(Weight10s.shape[0],1))  )
mp.plot(x10,y10,'orange',label='d=10')
OutlierMatrix = pol.polynomial.polyvander(Outliers,10)
Outlier_Weights = np.dot( OutlierMatrix , np.reshape(Weight10s,(Weight10s.shape[0],1)) )
mp.plot(np.reshape(Outliers,(1,3,)),np.reshape(Outlier_Weights,(1,3)),'o','orange',label = "Outliers")
fileio.write( "\n\n Calculated Outlier Weights = "+ str(Outlier_Weights) )
fileio.write( "\n\n Outlier Heights = "+str(Outliers))
mp.legend(loc='Best')
mp.show()



######### Uing Polyfit ##################################
fileio.write("\n :::::::::::::::::: Numpy Polyfit Calculations ::::::::::::::::::")
mp.xlim(xmin=140,xmax=210)
mp.ylim(ymin=30,ymax=150)
mp.grid(True)

mp.xlabel('Height ( in cm )----->')
mp.ylabel('Weight (in kg)----->')
x = np.linspace(100,210,200)
mp.title('Using PolyFit')
mp.plot(Sample[:,1],Sample[:,0],'o',color='black')

Weight1s = np.polyfit( Sample[:,1] ,Sample[:,0],1)
fileio.write("\n Weight Matrix for degree 1 polynomial="+str(Weight1s))
p1 = np.poly1d(Weight1s)
mp.plot(x,p1(x),'blue',label='d=1')
mp.plot(np.reshape(Outliers,(1,3,)),np.reshape(p1(Outliers),(1,3)),'o','blue',label = "Outliers")

Weight5s = np.polyfit(Sample[:,1] ,Sample[:,0],5)
fileio.write("\n Weight Matrix for degree 5 polynomial ="+str(Weight5s))
p5 = np.poly1d(Weight5s)
mp.plot(x,p5(x),'green',label='d=5')
mp.plot(np.reshape(Outliers,(1,3,)),np.reshape(p5(Outliers),(1,3)),'o','green',label = "Outliers")

Weight10s = np.polyfit( Sample[:,1] ,Sample[:,0],10)
fileio.write("\n Weight Matrix for degree 10 polynomial ="+str(Weight10s))
p10 = np.poly1d(Weight10s)
mp.plot(x,p10(x),'orange',label='d=10')
mp.plot(np.reshape(Outliers,(1,3,)),np.reshape(p10(Outliers),(1,3)),'o','orange',label = "Outliers")
mp.legend(loc='Best')
mp.show()
'''

##############################################################################################
###### NUMPY pinverse and Vandermonde ##############
fileio.write("\n\n\n ::::::::::::::::::Scaled Numpy Pinv Calculations ::::::::::::::::::")
mp.xlim(xmin=140,xmax=210)
mp.ylim(ymin=30,ymax=350)
mp.grid(True)
mp.xlabel('Height ( in cm )----->')
mp.ylabel('Weight (in kg)----->')
x = np.linspace(100,210,200)
mp.title('Numpy Pinv Fit')
mp.plot(Sample[:,1],Sample[:,0],'o',color='black')

Sample[:,1]=0.1*Sample[:,1]
Sample[:,0]=0.1*Sample[:,0]
#print(Sample)

VanderMondeX1 = pol.polynomial.polyvander(Sample[:,1],1)
Weight1 = np.dot( np.linalg.pinv(VanderMondeX1) , Sample[:,0])
fileio.write("\n Weight Matrix for Degree 1 polynomial="+str(Weight1))
x1 = pol.polynomial.polyvander(x,1)
y1 = np.dot( x1, np.reshape(Weight1,(Weight1.shape[0],1))  )
print(x1,y1)
mp.plot(x1,y1,'green',label = 'deg=1')
OutlierMatrix = pol.polynomial.polyvander(Outliers,1)
Outlier_Weights = np.dot( OutlierMatrix , np.reshape(Weight1,(Weight1.shape[0],1)) )
mp.plot(np.reshape(Outliers,(1,3,)),np.reshape(Outlier_Weights,(1,3)),'o','green',label = "Outliers")
fileio.write( "\n\n Calculated Outlier Weights = "+ str(Outlier_Weights) )
fileio.write( "\n\n Outlier Heights = "+str(Outliers))

VanderMondeX5 = pol.polynomial.polyvander(Sample[:,1],5)
Weight5 = np.dot( np.linalg.pinv(VanderMondeX5) , Sample[:,0])
fileio.write("\n Weight Matrix for Degree 5 polynomial="+str(Weight5))
x5 = pol.polynomial.polyvander(x,5)
y5 = np.dot( x5, np.reshape(Weight5,(Weight5.shape[0],1))  )
print(x5,y5)
mp.plot(x5,y5,'blue',label = "deg=5")
OutlierMatrix = pol.polynomial.polyvander(Outliers,5)
Outlier_Weights = np.dot( OutlierMatrix , np.reshape(Weight5,(Weight5.shape[0],1)) )
mp.plot(np.reshape(Outliers,(1,3,)),np.reshape(Outlier_Weights,(1,3)),'o','blue',label = "Outliers")
fileio.write( "\n\n Calculated Outlier Weights = "+ str(Outlier_Weights) )
fileio.write( "\n\n Outlier Heights = "+str(Outliers))

VanderMondeX10 = pol.polynomial.polyvander(Sample[:,1],10)
Weight10 = np.dot( np.linalg.pinv(VanderMondeX10) , Sample[:,0])
fileio.write("\n Weight Matrix for Degree 10 polynomial="+str(Weight10))
x10 = pol.polynomial.polyvander(x,10)
y10 = np.dot( x10, np.reshape(Weight10,(Weight10.shape[0],1))  )
print(x10,y10)
mp.plot(x10,y10,'red',label = "deg=10")
OutlierMatrix = pol.polynomial.polyvander(Outliers,10)
Outlier_Weights = np.dot( OutlierMatrix , np.reshape(Weight10,(Weight10.shape[0],1)) )
mp.plot(np.reshape(Outliers,(1,3,)),np.reshape(Outlier_Weights,(1,3)),'o','red',label = "Outliers")
fileio.write( "\n\n Calculated Outlier Weights = "+ str(Outlier_Weights) )
fileio.write( "\n\n Outlier Heights = "+str(Outliers))
mp.legend()
mp.show()
'''


fileio.close()




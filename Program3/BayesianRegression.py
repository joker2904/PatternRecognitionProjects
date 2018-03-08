import numpy as np
import numpy.polynomial as pol
import matplotlib.pyplot as mp
import scipy.sparse.linalg as sla

def ComputeBayesianRegression(X,sigma_2,sigma_0_2,y):
    return np.dot(np.dot(np.linalg.inv( np.dot(X.T,X) + (sigma_2)/(sigma_0_2) * np.identity(X.shape[1]) ),X.T),y)
    #return sla.lsqr(X, y, damp=((sigma_2)/(sigma_0_2)))[0]
    #return sla.lsmr(X, y, damp=((sigma_2)/(sigma_0_2)))[0]

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
    return X,Outliers[:,1]

Sample,Outliers = LoadData()
VanderMondeX = pol.polynomial.polyvander(Sample[:,1],5)


MLE_Theta =  np.dot( np.linalg.pinv(pol.polynomial.polyvander(Sample[:,1],5)) , Sample[:,0])
SigmaList = [np.var(Sample[:,0]),1.0,2.0,3.0,504.0,7000.0]


for sigmaValue in SigmaList:
    mp.xlim(xmin=140,xmax=210)
    mp.ylim(ymin=30,ymax=150)
    mp.xlabel('Height ( in cm )----->')
    mp.ylabel('Weight (in kg)----->')
    x = np.linspace(100,210,200)
    mp.title('Bayesian Fit for SigmaSquare ='+str(sigmaValue))
    mp.plot(Sample[:,1],Sample[:,0],'o',color='black',label='Data Points')
    MAP_Theta = ComputeBayesianRegression( VanderMondeX, sigmaValue, 3.0, Sample[:,0] )
    VX = pol.polynomial.polyvander(x,5)
    y_MAP = np.dot( VX,np.reshape(MAP_Theta,(6,1)) )
    y_MLE = np.dot( VX,np.reshape(MLE_Theta,(6,1)) )
    mp.plot(x,y_MAP,'orange',label='Bayesian Fit')
    mp.plot(x,y_MLE,'pink',  label='Maximum Likelihood Fit')

    OutlierMatrix = pol.polynomial.polyvander(Outliers,5)
    Outlier_Weights_MAP = np.dot( OutlierMatrix , np.reshape(MAP_Theta,(MAP_Theta.shape[0],1)) )
    Outlier_Weights_MLE = np.dot( OutlierMatrix , np.reshape(MLE_Theta,(MLE_Theta.shape[0],1)) )

    print("\n Outlier Heights = ",sigmaValue,Outliers)
    print("\n Weights by MAP =",Outlier_Weights_MAP.T)
    print("\n Weights by MLE =",Outlier_Weights_MLE.T)
    mp.plot(np.reshape(Outliers,(1,3)),np.reshape(Outlier_Weights_MAP,(1,3)),'o',color='red',label = "Outliers by MAP")
    mp.plot(np.reshape(Outliers,(1,3)),np.reshape(Outlier_Weights_MLE,(1,3)),'o',color='magenta',label = "Outliers by MLE")
    mp.legend()
    mp.show()

    y_predicted = np.dot(VanderMondeX, np.reshape(MAP_Theta, (6, 1)))
    #print(y_predicted)
    print("\n Weights predicted by MAP ::")
    for i,j in zip(Sample[:,1],y_predicted):
        print("\n Height = ",i," , Weight =",j)



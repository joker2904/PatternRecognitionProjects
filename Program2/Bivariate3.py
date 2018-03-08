import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.mlab import bivariate_normal
import scipy.stats as linregress
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy import interpolate
from scipy.stats import linregress
if __name__ == "__main__":

    # read data as 2D array of data type 'object'
    data = np.loadtxt('whData.dat', dtype=np.object, comments='#', delimiter=None)

    # read height and weight data into 2D array (i.e. into a matrix)
    X = data[:, 0:2].astype(np.float)

    missingValue =  X[X[:, 0] < 0, :]
    missingValue = missingValue[:,1]
    # removing negative and zeros from both columns
    X = X[X[:, 1] > 0, :]
    X = X[X[:, 0] > 0, :]

    X = X[np.argsort(X[:, 1])]
    height = X[:, 1]
    weight = X[:, 0]
    mean1 = np.mean(height)
    mean2 = np.mean(weight)

    dev1 = np.std(height)
    dev2 = np.std(weight)
    p = np.cov(height,weight)[0][1] / (dev1 * dev2)

    var1 = np.var(height)



    var2 = np.var(weight)
    sigma = np.array([[var1,p*dev1*dev2],[p*dev1*dev2,var2]])
    inv = np.linalg.inv(sigma)
    mu = np.array([mean1, mean2])


    def calcExp1(height, weight, mu):
        xPart = np.array(height - mu[0])
        yPart = np.array(weight - mu[1])
        xPartPow = xPart ** 2
        yPartPow = yPart ** 2
        sum = - (((xPartPow / var1) + (yPartPow / var2) - ((2 * p * xPart * yPart)/ (dev1 * dev2)))) / (2 * (1 - (p ** 2)))

        sum1 = (-1.0 * (xPartPow / (var1) + yPartPow / (var2) - 2 * p * xPart * yPart / (dev1 * dev2))) / 2 * (
        1 - p ** 2)
        exp = np.exp(sum)
        fun = exp / (2 * np.pi * dev1 * dev2 * np.sqrt(1 - (p ** 2)))
        return fun

    fn = calcExp1(height, weight, mu)

    temp = p * dev1 * dev2

    plt.plot(height, weight, 'o', label="data")

    x = np.random.uniform(150, 200, 100000)
    y = np.random.uniform(40, 95, 100000)
    z = calcExp1(x, y, mu)
    Z = bivariate_normal(x, y, dev1, dev2, mu[0], mu[1], temp)
    xi = np.linspace(150, 190, 100)
    yi = np.linspace(40, 110, 100)
    ## grid the data.
    zi = griddata((x, y), Z, (xi[None, :], yi[:, None]), method='cubic')
    # contour the gridded data, plotting dots at the randomly spaced data points.

    plt.contour(xi, yi, zi, 10, linewidths=1, colors='k')
    plt.xlim(160, 190)
    plt.ylim(50, 95)
    plt.xlabel("Heights------>")
    plt.ylabel("Weights------>")
    predictedValue = mean2 + (p * dev2 * (missingValue[:] - mean1) )/ dev1
    plt.plot(missingValue, predictedValue, 'o', label="outliers")
    slope, intercept, r_value, p_value, std_err = linregress(missingValue, predictedValue)
    x = np.linspace(160,250,1000)
    plt.plot(x,intercept+(slope*x))
    plt.legend()
    print("The values are", missingValue,"\n",predictedValue)
    plt.show()






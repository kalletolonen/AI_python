import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import os

#Data imports
filedir = os.path.abspath("C:\Python_AI\Exercise_4")
filename = "quadreg_data.csv"
filepath = os.path.join(filedir, filename)
data_pd = pd.read_csv(filepath,skiprows=0,names=["x","y"])
print(data_pd)

xpd = np.array(data_pd[["x"]])
ypd = np.array(data_pd[["y"]])
xpd = xpd.reshape(-1,1)
ypd = ypd.reshape(-1,1)

print("xpd", "\n",xpd)
print("ypd", "\n",ypd)

#Set up the polynomial regression model of appropriate degree
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(xpd)
pol_reg = LinearRegression()
pol_reg.fit(X_poly, ypd)

#Here X_poly is an array which contains powers of x as columns. 
# The powers are increasing from 0 to 2 and they are needed in order to use sklearn for 
# quadratic regression. 
# Same procedure for x is needed when predicting values of qudratic model. We see it here 
# when we plot the results

plt.scatter(xpd, ypd, color='red')
xval = np.linspace(-1,1,10).reshape(-1,1)
plt.plot(xval, pol_reg.predict(poly_reg.fit_transform(xval)), color='blue')
plt.show()

#Show coefficients
print(pol_reg.coef_)
print("c=",pol_reg.intercept_)
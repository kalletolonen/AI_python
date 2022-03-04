import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

#Data imports
filedir = os.path.abspath("C:\Python_AI\Exercise_4")
filename = "quadreg_data.csv"
filepath = os.path.join(filedir, filename)
data_pd = pd.read_csv(filepath,skiprows=0,names=["x","y"])

#print(data_pd)

xpd = np.array(data_pd[["x"]])
ypd = np.array(data_pd[["y"]])
n = xpd.size

xbar = np.mean(xpd)
ybar = np.mean(ypd)

xpd = xpd.reshape((n,))
ypd = ypd.reshape((n,))

#Next we compute the five sums appearing above

Sxx = np.sum(xpd**2)-n*xbar**2
Sxy = np.dot(xpd,ypd)-n*xbar*ybar
Sxx2 = np.sum(xpd**3)-xbar*np.sum(xpd**2)
Sx2y = np.sum(xpd**2*(ypd))-ybar*np.sum(xpd**2)
Sx2x2 = np.sum(xpd**4)-(np.sum(xpd**2)**2)/n

a = (Sx2y*Sxx-Sxy*Sxx2)/(Sxx*Sx2x2-Sxx2**2)
b = (Sxy*Sx2x2-Sx2y*Sxx2)/(Sxx*Sx2x2-Sxx2**2)
c = ybar-b*xbar-a*np.sum(xpd**2)/n

#print(a)
#print(b)
#print(c)

#Using these numbers the fitted parabola can be plotted as
x = np.linspace(-1,1,100)
y = a*x**2 + b*x +c
plt.plot(x,y)
plt.scatter(xpd,ypd,color="black")
plt.show()
#Also, we may use the formula of this parabola to predict y for any given value(s) x.

#The performance of the model is quantified using RMSE and R^2 value. They become
yhat = a*xpd**2 + b*xpd +c
RMSE = np.sqrt(np.sum((ypd-yhat)**2)/n)
print(RMSE)
R2 = 1-np.sum((ypd-yhat)**2)/np.sum((ypd-ybar)**2)
print(R2)
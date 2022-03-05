import numpy as np
from sklearn import linear_model
from sklearn import metrics
import matplotlib.pyplot as plt
import os

#Data imports
filedir = os.path.abspath("C:\Python_AI\Exercise_4")
filename = "linreg_data.csv"
filepath = os.path.join(filedir, filename)
my_data = np.genfromtxt(filepath, delimiter=',')

#Read in data and convert it into 2D arrays
xp = my_data[:,0]
yp = my_data[:,1]
print(xp)
xp = xp.reshape(-1,1)
yp = yp.reshape(-1,1)

#Model is created and trained in just two lines of code
regr = linear_model.LinearRegression()
regr.fit(xp, yp) # fitting the model=training the model

#The coefficieints a and b are now attributes of regr object
print(regr.coef_,regr.intercept_)

#Making predictions is done as follows
xval = np.full((1,1),0.5)
yval = regr.predict(xval)
print(yval)

#Plotting of the regression line can be done by first predicting y-values for some appropriate x-values
xval = np.linspace(-1,2,20).reshape(-1,1)
yval = regr.predict(xval)
plt.plot(xval,yval) # this plots the line
plt.scatter(xp,yp,color="red")
plt.show()

#Regarding the metrics of accuracy, they are readily available in sklearn
yhat = regr.predict(xp)
print('Mean Absolute Error:', metrics.mean_absolute_error(yp, yhat))  
print('Mean Squared Error:', metrics.mean_squared_error(yp, yhat))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(yp, yhat)))
print('R2 value:', metrics.r2_score(yp, yhat))
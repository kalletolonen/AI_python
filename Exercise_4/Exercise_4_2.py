#Consider the data from the file weight-height.csv.

#1) Inspect the dependence between height and weight using a scatter plot. 
# You may use either of the variables as independent variable.

#2) Choose appropriate model for the dependence
#-Linear regression

#3) Perform regression on the data using your model of choice

#4) Plot the results

#5) Compute RMSE and R2 value

#6) Assess the quality of the regression (visually and using numbers) in your own words.
#The quality of the regression model is good, according to R2, around 85% (depends on the
# training/test-set slicing) of dependent variability 
# can be explained by the model. You can clearly see this with the plotting
#as the data forms a coherent visual pattern. RMSE would help in choosing
# the best regression model for the task.

#You are not required to split the dataset into training and testing sets. 
#Of course you are completely free to experiment it here already.
#It is recommended that you use the module sklearn for all your computations.

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

#Data imports
filedir = os.path.abspath("C:\Python_AI\Exercise_4")
filename = "weight-height.csv"
filepath = os.path.join(filedir, filename)
df = pd.read_csv(filepath,skiprows=0)

#Split the dataset to 2 DataFrames for sklearn
x = df[['Height']]
y = df[['Weight']]

#Scatter plot to determine the model for dependence (Linear Regression)
plt.scatter(x,y)
plt.scatter(x,y,color="r",marker="o",label="Points")
plt.title('Scatter plot for weight/height-correlation')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.show()

#Linear regression of the data
#splitting the data into training and testing portions with 80/20 ratio. 
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

#Compare the training and testing sets
#print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
plt.scatter(x_train,y_train)
plt.scatter(x_test,y_test,color="red")
plt.legend(["train","test"])
plt.xlabel("Height")
plt.ylabel("Weght")
plt.title("Dataset splitting - a comparison of training and testing data")
plt.show()

#Train (fit) the model using training data only
lm = linear_model.LinearRegression()
model = lm.fit(x_train, y_train)

#The predictions
#to_numpy is crucial for this to work! (atleast in 2022/03/05)
plt.scatter(x,y)
plt.plot(x_train.to_numpy(), lm.predict(x_train.to_numpy()), color = "green")
plt.plot(x_test.to_numpy(),lm.predict(x_test.to_numpy()),color="red")
plt.title("Prediction")
plt.show()

#For RMSE
xpd = np.array(df[['Weight']])
ypd = np.array(df[['Height']])

n = xpd.size
xbar = np.mean(xpd)
ybar = np.mean(ypd)
xpd = xpd.reshape((n,))
ypd = ypd.reshape((n,))

#Next we compute the above five sums for RMSE
Sxx = np.sum(xpd**2)-n*xbar**2
Sxy = np.dot(xpd,ypd)-n*xbar*ybar
Sxx2 = np.sum(xpd**3)-xbar*np.sum(xpd**2)
Sx2y = np.sum(xpd**2*(ypd))-ybar*np.sum(xpd**2)
Sx2x2 = np.sum(xpd**4)-(np.sum(xpd**2)**2)/n
a = (Sx2y*Sxx-Sxy*Sxx2)/(Sxx*Sx2x2-Sxx2**2)
b = (Sxy*Sx2x2-Sx2y*Sxx2)/(Sxx*Sx2x2-Sxx2**2)
c = ybar-b*xbar-a*np.sum(xpd**2)/n

#R2 score & RMSE are computed for the test data
print("R2= ",lm.score(x_test,y_test)) # using linear model

yhat = a*xpd**2 + b*xpd +c
RMSE = np.sqrt(np.sum((ypd-yhat)**2)/n)
print("RMSE= ", RMSE)
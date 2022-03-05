#use 80/20 division between training and testing sets

import pandas as pd
from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

filedir = os.path.abspath("C:\Python_AI\Exercise_4")
filename = "Admission_Predict.csv"
filepath = os.path.join(filedir, filename)
df = pd.read_csv(filepath,skiprows=0,delimiter=",")

#print(df.head())
#print(df.index)

#Of the many possible variables (columns) we pick the CGPA column to be our independent variable 
# and Chance of Admit to be our dependent variable. 
# So we would like to investigate it CGPA can be used to predict the chance of
#  a person to be admitted to university.

X = df[['CGPA']]
y = df[['Chance of Admit ']]


#Now it is time to do the splitting into training and testing portions with 80/20 ratio. 
# This can be done in one line as
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

#print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
#plt.scatter(X_train,y_train)
#plt.scatter(X_test,y_test,color="red")
#plt.legend(["train","test"])
#plt.xlabel("CGPA")
#plt.ylabel("Chance of Admit")
#plt.title("Dataset splitting")
#plt.show()

#It is worthwhile to note that the splitting is random so it does not follow any particular pattern
#  and is very likely to change every time it is done. You can check this by printing
#  the heads of training and testing sets
#print(y_train.head())
#print(y_test.head())

#Next we train (fit) the model using training data only
print(X_train.head())
print(y_train.head())

lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)

#The predictions are best observed as a line plot with the full dataset present
#to_numpy is crucial for this to work! (atleast in 2022/03/04)
plt.scatter(X,y)
plt.plot(X_train.to_numpy(), lm.predict(X_train.to_numpy()), color = "green")
plt.plot(X_test.to_numpy(),lm.predict(X_test.to_numpy()),color="red")
plt.title("Prediction")
plt.show()

#Finally the R2 score is computed for the test data
print("R2=",lm.score(X_test,y_test)) # using linear model
print("R2=",metrics.r2_score(y_test,lm.predict(X_test.to_numpy())))# using sklearn

#R2 score = “(total variance explained by model) / total variance.” So if it is 100%, the 
# two variables are perfectly correlated, # i.e., with no variance at all. 
# A low value would show a low level of correlation, meaning a regression model
#  that is not valid, but not in all cases.
#Investigate the model for predicting Boston house prices by adding more explanatory variables to it in addition to RM and LSTAT.
#a) Which variable would you add next? Why?
#I added PTRATION (-.51) because it's the next most impactful variable according to the heat map.

#b) How does adding it affect the model's performance? Compute metrics and compare to having just RM and LSTAT.
#Adding it made a meaningful difference to the performance of the model. The performance difference was between 
# 4.5%(Test) and 6.3%(Training) in RMSE-score and the difference in R2-score was between 4.1%(Test) and 4.5(Training)%.
# My conclusion was that adding a third variables made the model slightly better. 

#d) Does it help if you add even more variables?
#I added an additional variable to the model to find out if it would make a diffence. I chose ZN, since it had the second largest 
#positive correlation in the heat map. Adding more variables did make the model better, but only around 0.1%. I think that linear regression is just the 
# wrong model to use for predicting anything meaningful from this data. 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = load_boston()

#House prices are recorded in MEDV variable and that is our target variable for prediction
#convert the dataset into pandas data frame The target variable can also be added to data frame
df = pd.DataFrame(data.data, columns=data.feature_names)
df['MEDV'] = data.target
#print(df.head())

#The house prices are distributed quite normally albeit with few outliers
plt.hist(df['MEDV'],25)
plt.xlabel("MEDV")
plt.show()

#Since we are aiming for multiple regression we need to check the data for multicollinearity. 
# To this end we plot a heat map of the correlation coefficients
sns.heatmap(data=df.corr().round(2), annot=True)
plt.show()

#The dependence of the target variable and independent variables is best observed using the scatter plots.
plt.subplot(2,2,1)
plt.scatter(df['RM'],df['MEDV'])
plt.xlabel("RM")
plt.ylabel("MEDV")

plt.subplot(2,2,2)
plt.scatter(df['LSTAT'],df['MEDV'])
plt.xlabel("LSTAT")
plt.ylabel("MEDV")

plt.subplot(2,2,3)
plt.scatter(df['PTRATIO'],df['MEDV'])
plt.xlabel("PTRATIO")
plt.ylabel("MEDV")

plt.subplot(2,2,4)
plt.scatter(df['ZN'],df['MEDV'])
plt.xlabel("ZN")
plt.ylabel("MEDV")

plt.show()

#prepare training and testing data for multiple linear regression for the original model and a model with 3 and 4 variables
X = pd.DataFrame(df[['LSTAT','RM']], columns = ['LSTAT','RM'])
y = df['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=5)

X2 = pd.DataFrame(df[['LSTAT','RM', 'PTRATIO']], columns = ['LSTAT','RM', 'PTRATIO'])
y2 = df['MEDV']
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size = 0.2, random_state=5)

X3 = pd.DataFrame(df[['LSTAT','RM', 'PTRATIO','ZN']], columns = ['LSTAT','RM', 'PTRATIO','ZN'])
y3 = df['MEDV']
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size = 0.2, random_state=5)

#Train the linear models using the same commands as in the case of simple linear regression. 
lm = LinearRegression()
lm.fit(X_train, y_train)

lm2 = LinearRegression()
lm2.fit(X2_train, y2_train)

lm3 = LinearRegression()
lm3.fit(X3_train, y3_train)

#Model evaluation is performed for training and testing sets separately.
y_train_predict = lm.predict(X_train)
rmse = (np.sqrt(mean_squared_error(y_train, y_train_predict)))
r2 = r2_score(y_train, y_train_predict)

y_test_predict = lm.predict(X_test)
rmse_test = (np.sqrt(mean_squared_error(y_test, y_test_predict)))
r2_test = r2_score(y_test, y_test_predict)

print("Original model training set with 2 variables (RMSE): ", rmse, "R2: ",r2)
print("Original model testing set with 2 variables: ", rmse_test, "R2: ",r2_test)

#Model evaluation for 3 variables
y2_train_predict = lm2.predict(X2_train)
rmse2 = (np.sqrt(mean_squared_error(y2_train, y2_train_predict)))
r2_2 = r2_score(y2_train, y2_train_predict)

y2_test_predict = lm2.predict(X2_test)
rmse_test2 = (np.sqrt(mean_squared_error(y2_test, y2_test_predict)))
r2_test2 = r2_score(y2_test, y2_test_predict)

print("Modified model training set with 3 variables (RMSE): ", rmse2, "R2: ",r2_2)
print("Modified model testing set with 3 variables: ", rmse_test2, "R2: ",r2_test2)

#Model evaluation for 4 variables
y3_train_predict = lm3.predict(X3_train)
rmse3 = (np.sqrt(mean_squared_error(y3_train, y3_train_predict)))
r2_3 = r2_score(y3_train, y3_train_predict)

y3_test_predict = lm3.predict(X3_test)
rmse_test3 = (np.sqrt(mean_squared_error(y3_test, y3_test_predict)))
r2_test3 = r2_score(y3_test, y3_test_predict)

print("Modified model training set with 4 variables (RMSE): ", rmse3, "R2: ",r2_3)
print("Modified model testing set with 4 variables: ", rmse_test3, "R2: ",r2_test3)

print ("Diff 2 to 3 variables: \n","RMSE: ", rmse/rmse2*100-100,"%\n","RMSE_Test: ",rmse_test/rmse_test2*100-100,"\n","R2: ",r2/r2_2*100-100,"%\n","R2_Test: ",r2_test/r2_test2*100-100,"%"    )
print ("Diff 3 to 4 variables: \n","RMSE: ", rmse2/rmse3*100-100,"%\n","RMSE_Test: ",rmse_test2/rmse_test3*100-100,"\n","R2: ",r2_2/r2_3*100-100,"%\n","R2_Test: ",r2_test2/r2_test3*100-100,"%"    )
#The results indicate that our model does not work very well in this case. 

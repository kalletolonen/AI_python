import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_boston
#from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = load_boston()

#California is an alternative dataset, which is ethically sustainable
#data = fetch_california_housing()

#Print variable data keys
#print(data.keys())

#Description of variables
#print(data.DESCR)

#House prices are recorded in MEDV variable and that is our target variable for prediction
#convert the dataset into pandas data frame The target variable can also be added to data frame

df = pd.DataFrame(data.data, columns=data.feature_names)
df['MEDV'] = data.target
print(df.head())

#The house prices are distributed quite normally albeit with few outliers
plt.hist(df['MEDV'],25)
plt.xlabel("MEDV")
plt.show()

#Since we are aiming for multiple regression we need to check the data for multicollinearity. 
# To this end we plot a heat map of the correlation coefficients
sns.heatmap(data=df.corr().round(2), annot=True)
plt.show()

#The dependence of the target variable and independent variables is best observed using the scatter plots.
plt.subplot(1,2,1)
plt.scatter(df['RM'],df['MEDV'])
plt.xlabel("RM")
plt.ylabel("MEDV")

plt.subplot(1,2,2)
plt.scatter(df['LSTAT'],df['MEDV'])
plt.xlabel("LSTAT")
plt.ylabel("MEDV")
plt.show()

#Both variables seem to have quite linear infuence on the target variable. Some deviations of this
#  trend may, however, cause problems in predictions.

#prepare training and testing data for multiple linear regression.
X = pd.DataFrame(df[['LSTAT','RM']], columns = ['LSTAT','RM'])
y = df['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=5)

#Train the linear model using the same commands as in the case of simple linear regression. Now the only difference is 
# that X_train contains two columns, one for each explanatory variable.

lm = LinearRegression()
lm.fit(X_train, y_train)

#Model evaluation is performed for training and testing sets separately.
y_train_predict = lm.predict(X_train)
rmse = (np.sqrt(mean_squared_error(y_train, y_train_predict)))
r2 = r2_score(y_train, y_train_predict)

y_test_predict = lm.predict(X_test)
rmse_test = (np.sqrt(mean_squared_error(y_test, y_test_predict)))
r2_test = r2_score(y_test, y_test_predict)

print(rmse,r2)
print(rmse_test,r2_test)
#The results indicate that our model does not work very well in this case. 
# Partly this is due to the fact that RM and LSTAT are quite strongly correlated so
#  there is some amount of multicollinearity present in this model.
#  But for now it is the best we can do with two explanatory variables.
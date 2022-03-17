import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("winequality-white.csv",delimiter=";")
print(df.head())

#Variables' correlation is now examined in absolute values to more easily identify key variables.

sns.heatmap(data=df.corr().round(2).abs(), annot=True)
plt.show()

#This time we look at columns which contain light colors as they correspond to high absolute correlation. 
# We pick the alcohol content and amount of residual sugar to explain the density of the wine.
#  Density is often regarded as a major factor in wine quality.
#  These explanatory variables have positive and negative, rather linear too, correlation with the density.

plt.subplot(1,2,1)
plt.scatter(df['residual sugar'],df['density'])
plt.xlabel("residual sugar")
plt.ylabel("density")

plt.subplot(1,2,2)
plt.scatter(df['alcohol'],df['density'])
plt.xlabel("alcohol")
plt.ylabel("density")
plt.show()

#form the training and testing data using these variables
X = pd.DataFrame(df[['residual sugar','alcohol']])
y = df['density']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=5)

#Training the model
lm = LinearRegression()
lm.fit(X_train, y_train)

#model evaluation
y_train_predict = lm.predict(X_train)
rmse = (np.sqrt(mean_squared_error(y_train, y_train_predict)))
r2 = r2_score(y_train, y_train_predict)

y_test_predict = lm.predict(X_test)
rmse_test = (np.sqrt(mean_squared_error(y_test, y_test_predict)))
r2_test = r2_score(y_test, y_test_predict)

print("RMSE: ",rmse,"R2: ",r2)
print("RMSE_test_set: ",rmse_test,"R2_test_set: ",r2_test)

#The basic principle in regression has been to minimize the total error with respect to training data.
#  Unfortunately, low training error does not guarantee good expected performance.
#  This phenomenon is called overfitting.

#When looking at the expected error E on test data it can be shown that it can be decomposed as
#E = Bias^^2 + Variance + noise

#Here noise is any uncertainty inherent in the problem and it is not in our control.
#  Bias is the difference between average prediction in our model and
#  the true value which we are trying to predict.
#  High bias means that we pay less attention to training data and
#  oversimplify the model (called underfitting). It implies larger errors on
#  training and testing data.

#Variance is the variability of prediction for given data points.
#  High variance pays a lot of attention to training data and generalizes
#  poorly on new data the model has never seen before. In other word,
#  models with high variance work very nicely for training data but produce
#  larger errors on test data.

#Often the key problem in regression is to find a good balance between 
# bias and variance.

#Simple models with a few parameters tend to have high bias and low variance. 
# Complex models with several parameters have low bias and high variance.
#  One needs to find an optimal balance between overfitting and underfitting.

#The complexity of the model influences bias and variance in opposite manner. 
# Therefore we cannot make them both small at the same time.
#  This is the tradeoff that one must be aware of when dealing with
#  regression problems.
# 
#  In fact, the dilemma with bias and variance concerns also other forms of ML.

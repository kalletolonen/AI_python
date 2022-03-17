#LASSO stands for "Least Absolute Shrinkage and Selection Operator"
#In LASSO regression we minimize the expression

#where α>0 controls the weight of the added penalty term. 
# Increasing α will decrese the slope b and it is actually allowed
#  to become equal to zero. In ridge regression the slope
#  only approaches, but never equals, zero.

#This subtle difference has major consequences in the frame of
#  multiple variables. LASSO regression can exclude useless variables
#  while reducing variance and simplifying model interpretation.
#  If all variables are useful
#  then ridge regression will probably do better.

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv('diamonds.csv')
print(df.head())

#For variables we pick all the numerical columns
X = df[['carat', 'depth', 'table', 'x', 'y', 'z']]
y = df[['price']]

#Split to training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)

#We run the LASSO regressor over multiple values of alpha 
# and measure the R2 score for test data.
alphas = [0.1,0.2,0.3,0.4,0.5,1,2,3,4,5,6,7,8]
scores = []
for alp in alphas:
    lasso = linear_model.Lasso(alpha=alp)
    lasso.fit(X_train, y_train)
    print(lasso.coef_.round(2))
    sc = lasso.score(X_test, y_test)
    scores.append(sc)
    print("alpha=",alp," lasso score:", sc)

#The scores are easily plotted as a function of alpha
plt.plot(alphas, scores)
plt.xlabel("alpha")
plt.ylabel("R2 score")
plt.show()

#We see that the R2 score peaks at around alpha=3.
#  More detailed examination near this value may improve the score more.

#Looking at the model coefficients we observe that
#  LASSO regressor returns zero values for two variables.
#  This is interpreted as these variables being useless in model predictions.
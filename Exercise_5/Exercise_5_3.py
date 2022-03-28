# Consider car performance data from the file Auto.csv.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn import linear_model

# 1) Read data into pandas dataframe
# Data imports
filedir = os.path.abspath("C:\Python_AI\Exercise_5")
filename = "Auto.csv"
filepath = os.path.join(filedir, filename)
df = pd.read_csv(filepath, skiprows=0)
print(df.head())

# 2) Setup multiple regression X and y to predict 'mpg' of cars using all the variables except 'mpg', 'name' and 'origin'
print(df.keys())

X = pd.DataFrame(df[['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year']], columns=[
                 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year'])
y = df['mpg']

# 3) Split data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5)

# 4) Implement both ridge regression and LASSO regression using several values for alpha.
# Ridge Regression

alphas = np.linspace(0,3,50) 
r2values = []
for alp in alphas:
    rr = Ridge(alpha=alp)
    rr.fit(X_train, y_train)
    r2_test = r2_score(y_test.to_numpy(), rr.predict(X_test.to_numpy()))
    r2values.append(r2_test)

plt.plot(alphas,r2values)
plt.show()

#Lasso regression
alphas = [0.1,0.2,0.3,0.4,0.5,1,2,3,4,5,6,7,8]
scores = []
for alp in alphas:
    lasso = linear_model.Lasso(alpha=alp)
    lasso.fit(X_train, y_train)
    print(lasso.coef_.round(2))
    sc = lasso.score(X_test, y_test)
    scores.append(sc)
    print("alpha=",alp," lasso score:", sc)

#The scores are plotted as a function of alpha
plt.plot(alphas, scores)
plt.xlabel("alpha")
plt.ylabel("R2 score")
plt.show()

# 5) Search optimal value for alpha (in terms of R2 score) by fitting the models with training data and computing the score using testing data
# 6) Plot the R2 scores for both regressors as functions of alpha
# 7) Identify, as accurately as you can, the value for alpha which gives the best score

#the alpha value peaks around 0.2-0.3 in terms of R2.

#More detailed look
alphas = [0.2,0.21,0.22,0.23,0.24,0.25,0.3,0.35,0.4]
scores = []
for alp in alphas:
    lasso = linear_model.Lasso(alpha=alp)
    lasso.fit(X_train, y_train)
    print(lasso.coef_.round(2))
    sc = lasso.score(X_test, y_test)
    scores.append(sc)
    print("alpha=",alp," lasso score:", sc)

#The scores are plotted as a function of alpha
plt.plot(alphas, scores)
plt.xlabel("alpha")
plt.ylabel("R2 score")
plt.show()


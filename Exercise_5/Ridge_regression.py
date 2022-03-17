#Ridge regression is designed to mitigate the bias-variance dilemma
#The main idea in ridge regression is to introduce small amount of bias
#  so that the regression line does not fit so well into training data
#  but it works better with testing data. This reduces variance.

#In general, ridge regression is shrinking parameters 
# (and so by reducing variance) 
# compared with usual linear regression.

from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd

#We use the single variable data from CSV file for demonstration purposes.
df = pd.read_csv("ridgereg_data.csv")
x = df[['x']]
y = df[['y']]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=5)

#We loop over several values for alpha and plot the fitted line in each case
#to_numpy is crucial as of 2022/07/03
#for alp in [0,1,5,10,20,30,50,100,1000]:
 #   rr = Ridge(alpha=alp)
  #  rr.fit(X_train, y_train)
   # plt.scatter(X_train,y_train)
    #plt.plot(X_train.to_numpy(),rr.predict(X_train.to_numpy()),color="red")
    #plt.title("alpha="+str(alp))
    #plt.show()

#to find optimal value for α one could use cross-validation 
# but we will simply try different values for α and compute
#  the R2 score for each choise

alphas = np.linspace(0,3,50) # past 4 the score decreases significantly
r2values = []
for alp in alphas:
    rr = Ridge(alpha=alp)
    rr.fit(X_train, y_train)
    r2_test = r2_score(y_test.to_numpy(), rr.predict(X_test.to_numpy()))
    r2values.append(r2_test)


plt.plot(alphas,r2values)
plt.show()

#Another value from ridge regression can be formulates as follows.
#  Note that simple linear regression needs two data points to
#  optimally select the two unknown parameters in the model.
#  For multiple linear regression over two variables the model
#  has three parameters (intercept and two slopes) and so
#  three data points are needed to determine them. In general
# , a model with n parameters needs n data points. So, for large
#  n we need a lot of data to fit the model. But ridge regression can be
#  used in this case since its' extra penalty term puts more
#  favor on small parameter values.

#In conclusion we may say that ridge regression has more bias
#  but less variance than usual linear regression.
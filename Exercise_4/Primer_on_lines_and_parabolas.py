#Regression means the prediction of a continuous dependent variable based on data

#In simple linear regression the goal is to investigate the relationship between an independent
#  variable (call it x) and a dependent variable (call it y).

#Independent variables are also called features, predictors, explanatory variables or inputs. 
# Dependent variables are also called target variable, response variable or output.

import numpy as np
import matplotlib.pyplot as plt

x = np.array([0,1,2,3,4])
y = np.array([1,3,5,7,9])
plt.scatter(x,y)
plt.plot(x,y) 
plt.show()


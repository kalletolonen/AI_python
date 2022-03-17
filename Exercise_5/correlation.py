import pandas as pd

#Correlation varies between -1 and 1. 1 Means that there is strong positive correlation and -1 strong negative correlation.
# Values close to 0 means there is no correlation. 
df = pd.read_csv("weight-height.csv")
print(df.corr())

#When performing multiple linear regression one needs to be aware that some explanatory 
# (independent) variables might be highly correlated. This is called multicollinearity. 
# Such variables should not be both used as explanatory variables in the regression model.

#The reason is that a variable's effect on target variable is hard to distinguish from 
# the effect of another variable if they are correlated (positive of negative correlation) 
# since a change in one variable is associated in another variable (that's what correlation is).

#On the other hand independent variables who have strong correlation with target variable are
#  good candidates for multiple linear regression.
#Consider the Dataset 50_Startups.csv which contains data for companies' profit etc.
#0) Read the dataset into pandas dataframe paying attention to file delimeter.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#Data imports
filedir = os.path.abspath("C:\Python_AI\Exercise_5")
filename = "50_Startups.csv"
filepath = os.path.join(filedir, filename)
df = pd.read_csv(filepath,skiprows=0)
print(df.head())

#1) Identify the variables inside the dataset
#Print variable data keys
print(df.keys())
#Index(['R&D Spend', 'Administration', 'Marketing Spend', 'State', 'Profit'], dtype='object')

#2) Investigate the correlation between the variables
#The profits are distributed almost normally
plt.hist(df['Profit'],15)
plt.xlabel("Profit")
plt.show()

# Next I plot a heat map of the correlation coefficients
sns.heatmap(data=df.corr().round(2), annot=True)
plt.show()

#3) Choose appropriate variables to predict company profit. Justify your choice.
#According to the heat map, there is a strong correlation in R&D and Marketing Spend with regards to profit.
#That's why I chose the variables 'Marketing Spend' and 'R&D Spend' as my variables. 

#4) Plot explanatory variables against profit in order to confirm (close to) linear dependence
plt.subplot(1,2,1)
plt.scatter(df['Marketing Spend'],df['Profit'])
plt.xlabel("Marketing Spend")
plt.ylabel("Profit")

plt.subplot(1,2,2)
plt.scatter(df['R&D Spend'],df['Profit'])
plt.xlabel("R&D Spend")
plt.ylabel("Profit")
plt.show()

#There seems to be a linear dependence in the data.

#5) Form training and testing data (80/20 split)
X = pd.DataFrame(df[['Marketing Spend','R&D Spend']], columns = ['Marketing Spend','R&D Spend'])
y = df['Profit']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=5)

#6) Train linear regression model with training data
lm = LinearRegression()
lm.fit(X_train, y_train)

#7) Compute RMSE and R2 values for training and testing data separately
#The result suggest that there's something seriously wrong with the rmse-calculation and that the r2-score implies that 
#the model is quite good at predicting profits. I tried dropping the state-column, but it didn't affect the results.
y_train_predict = lm.predict(X_train)
rmse = (np.sqrt(mean_squared_error(y_train, y_train_predict)))
r2 = r2_score(y_train, y_train_predict)
 
y_test_predict = lm.predict(X_test)
rmse_test = (np.sqrt(mean_squared_error(y_test, y_test_predict)))
r2_test = r2_score(y_test, y_test_predict)

print("RMSE_train: ", rmse, "R2: ",r2)
print("RMSE_test: ", rmse_test, "R2: ",r2_test)

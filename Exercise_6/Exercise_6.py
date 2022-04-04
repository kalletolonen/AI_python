import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os


#1) Read in the CSV file using pandas. Pay attention to file delimeter.
#Inspect the resulting dataframe with respect to column names and variable type.

#Data imports
filedir = os.path.abspath("C:\Python_AI")
filename = "bank.csv"
filepath = os.path.join(filedir, filename)
df = pd.read_csv(filepath,skiprows=0, delimiter=";")
#print(df.head())
#print(df.keys())

#2) Pick data from the following columns to a second dataframe 'df2': y, job, marital, default, housing, poutcome
df2 = pd.DataFrame(df[['y', 'job', 'marital', 'default', 'housing', 'poutcome']], columns=[
                 'y', 'job', 'marital', 'default', 'housing', 'poutcome'])
#print(df2.head())

#3) Convert categorical variables to dummy numerical values using the command
df3 = pd.get_dummies(df2,columns=['job','marital','default','housing','poutcome'])
#print(df3.head())

#4) Produce a heat map of correlation coefficients for all variables in df3. Describe the amount
#  of correlation between variables in your own words.
sns.heatmap(data=df3.corr().round(2), annot=True)
plt.show()

#There are obvious negative correlations between values, as in "marital_single/marital_married",
#  "default_yes/defaul_no" and "poutcome_failure/poutcome_unkown". 
# Otherwise there isn't exactly a strong correlation between any variables (>0.3)

#5) Select column called 'y' of df3 as target variable y, and all the remaining columns for explanatory variables X
X = df3.iloc[:, 1:24].values
y = df3.iloc[:, 0].values
#print(X)
#print(y)

#6) Split dataset into training and testing sets with 75/25  ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=0)

#7) Setup logistic regression model, train it with training data and predict on testing data.
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#8) Print confusion matrix (or use heat map if you want) and accuracy score for logistic regression model.
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
#print("Log. regression: ")
#print(cnf_matrix)

metrics.plot_confusion_matrix(model, X_test, y_test)
plt.show()

#9) Repeat steps 7 and 8 for k-nearest neighbors model. Use k=3, for example, or experiment with different values.
#7
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

#8
metrics.plot_confusion_matrix(classifier, X_test, y_test)
plt.show()

cnf_matrix2 = metrics.confusion_matrix(y_test, y_pred)


#10) Compare results between the two models.
print("Log. regression: ")
print(cnf_matrix)
print("KNN: ")
print(cnf_matrix2)

#Logistics regression seems to be a bit better at predicting true negative labels, but otherwise 
# the models are similar in their abilities.



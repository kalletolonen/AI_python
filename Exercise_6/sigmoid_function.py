#sklearn module in Python provides again high-level interface for
#  logistic regression problems. The interface is used very similary to
#  linear regression (simple or multiple). Let us illustrate this
#  using a dataset containing students' exam scores and information
#  if student was admitted to university or not.

#data imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
import seaborn as sns
import os

filedir = os.path.abspath("C:\Python_AI")
filename = "exams.csv"
filepath = os.path.join(filedir, filename)
df = pd.read_csv(filepath,skiprows=0)
print(df.head())

#As the data contains three columns we pick the feature variables and target variables by column indexing
X = df.iloc[:, 0:2]
y = df.iloc[:, -1]

#filter the rows based on the value in 3rd column (admit yes/no)
admit_yes = df.loc[y == 1]
admit_no = df.loc[y == 0]

#prepare a scatter plot of both classes (exam1 vs. exam2)
plt.scatter(admit_no.iloc[:,0],admit_no.iloc[:,1],label="admit no")
plt.scatter(admit_yes.iloc[:,0],admit_yes.iloc[:,1],label="admit yes")
plt.xlabel("exam1")
plt.ylabel("exam2")
plt.legend()
#This scatter plot shows clearly how student needs to do reasonably well in both
#  exams in order to get admitted to university.
#plt.show()

#For training of our machine learning model we do the usual data splitting into training and testing sets
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.25,random_state=0)
print(X_train.shape)

#Model is constructed and trained
model = LogisticRegression()
model.fit(X_train, y_train)

#to evaluate the model we predict on the testing set
y_pred = model.predict(X_test)

#Our first evaluation involves the confusion matrix, both in numerical and visual form
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)

metrics.plot_confusion_matrix(model, X_test, y_test)
plt.show()

#compute some other metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))

#plot the predicted labels
y_test2 = y_test.to_numpy()
idx1 = np.logical_and(y_pred == 1, y_test2 == 1)
idx2 = np.logical_and(y_pred == 1, y_test2 == 0)
idx3 = np.logical_and(y_pred == 0, y_test2 == 0)
idx4 = np.logical_and(y_pred == 0, y_test2 == 1)
X1 = X_test.loc[idx1]
X2 = X_test.loc[idx2]
X3 = X_test.loc[idx3]
X4 = X_test.loc[idx4]

plt.scatter(X1.iloc[:,0],X1.iloc[:,1],label="pred yes correct",marker="+",color="blue")
plt.scatter(X2.iloc[:,0],X2.iloc[:,1],label="pred yes incorrect",marker="o",color="blue")
plt.scatter(X3.iloc[:,0],X3.iloc[:,1],label="pred no correct",marker="+",color="red")
plt.scatter(X4.iloc[:,0],X4.iloc[:,1],label="pred yes incorrect",marker="o",color="red")

plt.xlabel("exam1")
plt.ylabel("exam2")
plt.legend()
plt.title("Predicted")
plt.show()
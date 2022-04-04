import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sb
import os

filedir = os.path.abspath("C:\Python_AI")
filename = "iris.csv"
filepath = os.path.join(filedir, filename)
df = pd.read_csv(filepath, skiprows=0)
#print(df.head())

#use all four numerical variables as features and species column as target
X = df.iloc[:, 0:4].values
y = df.iloc[:, 4].values
print(X)
print(y)

#Split to training and testing sets again
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=1)

#Setup KNN classifier, train it, and make predictions. We use k=5 here
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

#Evaluate the model using predictions of the test set. We form the confusion matrix
metrics.plot_confusion_matrix(classifier, X_test, y_test)
plt.show()

#ask sklearn to give us a classification report
print(classification_report(y_test, y_pred))

#To see how other values of k perform we check it using a loop and for each value of k we record the mean error rate
error = []
for k in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    error.append(np.mean(y_pred != y_test))

plt.plot(range(1, 20), error, marker='o', markersize=10)
plt.xlabel('k')
plt.ylabel('Mean Error')
plt.show()
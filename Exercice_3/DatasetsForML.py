import pandas as pd
import os

###
#pip install pandas
#pip install scikit-learn
#pip install numpy
###

#https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf


#import csv to dataframe
filedir = os.path.abspath("C:/Python_AI/Exercice_3")
filename = "iris.csv"
filepath = os.path.join(filedir, filename)

df = pd.read_csv(filepath)

#Examining the file
print(df.head()) #print the start of the file
#print(df.tail()) #print the end of the file
#print(df.describe()) #print statistical data from the file

#print(df.dtypes) #Data types
#print(df.index) #Data indexes
#print(df.columns) #column names & data type of the table(?)
#print(df.values) #Only values

#Data sorting
#df2 = df.sort_values('sepal.width',ascending=False) # does not sort in-place
#print(df2)

#slice data frames
#print(df[['sepal.width']]) #slice one column by name
#print(df[['sepal.width','sepal.length']]) #slice two columns by name
#print(df[2:4]) #slice rows by index, exclusive

#Slicing by rows and columns at the same time uses the functions loc() or iloc(). Slicing can be combined 
# with assignment operator to assign new values to existing data records.

#print(df.loc[2:4,['petal.width','petal.length']]) #slice rows by index and columns by name
#print(df.iloc[2:4,[0,1]]) # slice row and columns by index

#FIX DATA TYPES
#df['sepal.width']=df['sepal.width'].astype('category')

#Filtering of the data can be done using logical conditions or isin() function
#print(df.loc[df['sepal.width']>3]) # slicing with logical condition
#print(df[df['variety'].isin(['Setosa'])])

#New columns can be added to a data frame by giving it a name and using assignment operator 
#df["sepal.area"] = df["sepal.length"] * df["sepal.width"]
#df['zeros'] = 0.0
#print(df)

#Removing columns
#df = df.drop(['zeros'],axis=1)
#print("df after drop","\n",df)

#For renaming columns there are two possibilities. Either rename one column 
# (which can be done in-place) or give all columns new names
#df.rename(columns = {'sepal.area':'sep_ar'},inplace=True)
#print(df.head())
#df.columns = ['col1','col2','col3','col4','col5','col6']
#print(df.head())

#To add a row to a data frame takes an intermediate 
# step with the function Series()
to_append = [7.0,4.0,5.5,6.6,"Iris-setosa"]
a_series = pd.Series(to_append, index = df.columns)
#df = df.append(a_series, ignore_index=True)
#print(df)

#Looping the edataframe
#for ind, row in df.iterrows():
#    print(ind,row['variety'])

#Finally, data frames are easy to save to CSV files using
df.to_csv("iris_new.csv")
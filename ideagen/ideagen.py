import pandas as pd
import os
import random

#Data imports
filedir = os.path.abspath("C:\Python_AI\ideagen")
filename = "megatrends.csv"
filepath = os.path.join(filedir, filename)
df = pd.read_csv(filepath)

filename = "hype_cycle.csv"
filepath = os.path.join(filedir, filename)
df2 = pd.read_csv(filepath)

filename = "personas.csv"
filepath = os.path.join(filedir, filename)
df3 = pd.read_csv(filepath)

#let's turn on the INNOVATION LOOP.
innovate = True
while innovate:
    #Grab some randomized entries from source dataframes
    trend = (df.iat[random.randint(1,len(df)-1),0])
    hype = (df2.iat[random.randint(1,len(df2)-1),0])
    persona = (df3.iat[random.randint(1,len(df3)-1),0])
    print (persona, "\n")
    print (trend, "\n")
    print (hype, "\n")
    usrinput = input("Write your innovation (type q to quit): ")

    if usrinput != "q":
        #creates a new DataFrame from values
        df4 = pd.DataFrame({
        #grab the headers from source DF's
        list(df.columns.values.tolist())[0] : [trend],
        list(df2.columns.values.tolist())[0] : [hype],
        list(df3.columns.values.tolist())[0] : [persona],
        "Innovation" : [usrinput]},
        index = [1])

        #Append to idealist.csv, add headers if the .csv doesn't already exist
        df4.to_csv(os.path.join(filedir, 'idealist.csv'),index=False, mode='a', header=(not os.path.exists(os.path.join(filedir, 'idealist.csv'))))
    else:
        innovate = False
#0) Write a for loop which repeats the steps 1)-3) below for values of n ranging as
#500,1000,2000,5000,10000, 15000, 20000, 50000, 100000

#1) Use numpy to simulate throwing of two dice n times. Compute the sum of the dice.

#2) Use numpys histogram() function to compute the frequencies as

#h,h2 = np.histogram(s,range(2,14))
#where s contains the sum.
#3) Use matplotlib's bar function to plot the histogram as
#plt.bar(h2[:-1],h/n)
#and show the value of n in the title.

#4) What do you observe? You may need to run the loop a few times to see it.
#The data gets smoother as the amount of data increases

#5) How is this related to "regression to the mean"?
#With a large enought sample size, things tend to regress to the mean, ie. the most likely/common outcome/occurance

import numpy as np
import matplotlib.pyplot as plt
import random

#list including the numbers of throws for each round
rangeArray = [500,1000,2000,5000,10000, 15000, 20000, 50000, 100000]

rounds = 0
dieArray = []

#outer for-loop to loop throught the rangeArray [1-9]
for i in rangeArray:
    #inner for-loop to execute n-times (the # of rounds in given array/list location)
    for n in range(rangeArray[rounds]):
        #casting the dies
        die1 = random.randint(1,6)
        die2 = random.randint(1,6)
        #adding the result to a new list
        dieArray.append(die1+die2)
    #inner loop exits and the formed list is converted to np.array        
    a = np.array([dieArray])
    #plotting to a histogram
    h,h2 = np.histogram(a,range(2,14))
    plt.bar(h2[:-1],h/rangeArray[rounds])
    plt.title(str(rangeArray[rounds]) + " Rounds")
    plt.show()
    dieArray = []
    rounds += 1    
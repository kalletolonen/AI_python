#Read from CSV-file weight-height.csv to numpy-table information about the lengths and weights (in inches and pounds) of a group of students. Collect the lengths for the variable "length" and the weights for the variable "weight" by cutting the table.
#Convert lengths from inches to centimeters and weights from pounds to kilograms.
# Finally, calculate the means, medians, standard deviations, and variances of the lengths and weights.
#Extra: Draw a histogram pattern of the lengths
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("weight-height.csv",delimiter=",", dtype=str, skip_header=1)

dataB = data

print(dataB[0][1], "inches")
print(dataB[0][2], "pounds")

for i in dataB:
    i[1] = float(i[1])*2.54 #convert values to cm
    i[2] = float(i[2])*0.453592 #convert values to kg

print(dataB[0][1], "cm", dataB[0][1].dtype)
print(dataB[0][2], "kg")

#mean
weightList = []
heightList = []

for i in dataB:
    heightList.append(float(i[1]))
    weightList.append(float(i[2]))
    
print(heightList[0])
print("The mean of heights is ", np.mean(heightList), " cm.")
print("The mean of weights is ", np.mean(weightList), " kg.")

#median
print("The median of heights is ", np.median(heightList), " cm.")
print("The median of weights is ", np.median(weightList), " kg.")

#standard deviation
print("The standard deviation of height is ", np.std(heightList))
print("The standard deviation of weight is ", np.std(weightList))

#variations
print("The variance of height is ", np.var(heightList))
print("The variance of weight is ", np.var(weightList))

#histogram
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Height & weight histograms')
ax1.hist(heightList, bins=50)
ax1.set(xlabel='Height', ylabel='frequency')
ax2.hist(weightList, bins=50)
ax2.set(xlabel='Weight')
plt.show()
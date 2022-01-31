#Read from CSV-file weight-height.csv to numpy-table information about the lengths and weights (in inches and pounds) of a group of students. Collect the lengths for the variable "length" and the weights for the variable "weight" by cutting the table.
#Convert lengths from inches to centimeters and weights from pounds to kilograms.
# Finally, calculate the means, medians, standard deviations, and variances of the lengths and weights.
#Extra: Draw a histogram pattern of the lengths
import numpy as np
import matplotlib as plt

data = np.genfromtxt("weight-height.csv",delimiter=",", dtype=str, skip_header=1)

dataB = data

print(dataB[0][1], "inches")
print(dataB[0][2], "pounds")

for i in dataB:
    i[1] = np.double(i[1])*2.54 #convert values to cm
    i[2] = np.double(i[2])*0.453592 #convert values to kg

print(dataB[0][1], "cm")
print(dataB[0][2], "kg")

#mean
weightList = []
print(len(dataB))
for i in dataB:
    weightList.append(i[1])

print(len(weightList))

#median

#standard deviation

#variations

#histogram
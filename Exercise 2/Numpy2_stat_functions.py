import numpy as np

#Many statistical indicators can be calculated from the numerical values in the table. The sum is calculated with the command sum (), which can optionally be given as the second argument the number of the axis over which the sum is calculated. E.g.
A = np.array([[1,2,3],[4,5,6]])
#print(np.sum(A))
#print(np.sum(A,0), "column sums, summed along rows") # column sums, summed along rows
#print(np.sum(A,1), "row sums, summed along the columns") # row sums, summed along the columns

#The product is calculated similarly with the prod () command.
#print(np.prod(A))
#print(np.prod(A,0))
#print(np.prod(A,1))

#There are also cumulative variants cumsum () and cumprod () for sum and result.
#The largest and smallest elements are found with the min () / max () commands. E.g.
#print(np.min(A))
#print(np.min(A,0))
#print(np.min(A,1))
#print(np.max(A))
#print(np.max(A,0))
#print(np.max(A,1))

#The average of the elements is calculated with the function mean (), which can also be assigned to a specific axis.
#print(np.mean(A))
#print(np.mean(A,0))
#print(np.mean(A,1))

#The median, ie the "middle" value of the numerical values, is determined by the function median ()
#print(np.median(A))
#print(np.median(A,0))
#print(np.median(A,1))

#The variance and its standard root deviation measure how far the numbers are from their mean. For these, there are functions var () and std ()
#print(np.std(A))
#print(np.std(A,0))
#print(np.std(A,1))
#print(np.var(A))
#print(np.var(A,0))
#print(np.var(A,1))




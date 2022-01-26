import numpy as np

#The numpy library supports the following key data types: int8, int16, int32, int64, float16, float32, float64, uint8, uint16, uint32, uint64,  where numbers indicate the number of bits used and uint means an unsigned integer.
#These can be specified when creating a table so that the items in the table are of the desired data type. E.g.

A = np.array([1,2,3,4,5,6],dtype="int8")

print(A)
#print(A.dtype) # check data type

#A.astype("int16") # change data type

print(A.itemsize)
print(A.nbytes)
A2 = np.array([1,2,3,4,5,6],dtype="int64")
print(A2.nbytes) #different data types require different amounts of memory!

#With an unsigned integer type (uint), you should be careful, because when it encounters a negative number, an underflow occurs and the results can be catastrophic.
A = np.array([1,2,3,4,5,6],dtype="uint8")
#e.g. A-7 will crash the program

#However, the advantage of the uint type is that the same number of bits can represent twice as many positive numbers as the int type, because the int type prepares to handle roughly the same number of negative and positive numbers. The range of figures that can be displayed for different data types is shown in the following table.

#Complete documentation on data types can be found at
#https://numpy.org/doc/stable/reference/arrays.dtypes.html





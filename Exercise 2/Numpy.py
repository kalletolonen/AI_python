import numpy as np
x = np.array([1,3,4,5])
A = np.array([[1,3],[4,5]])

np.shape(x)
np.shape(A)

print(x)
print(A)

print("-------------------\n")
Z = np.zeros(5)
print(Z)
np.shape(Z)
print(Z, "aasdf")

Z2 = np.zeros((4,5)) # notice double brackets. Stores 0-digits to an 4*5 array
print(Z2)
np.shape(Z2)

Y = np.ones((2,3)) # stores number 1-digits to an array of 2 high and 3 wide
print(Y)

F=np.full((7,8),11) # stores number 11-digits to an array of 7*8
print(F)

x = np.linspace(0,5,10) #Generoi 0-5 väliin 10 tasaisin välein olevaa numeroa
print(x)

x2 = np.arange(0,5,0.2) #generoi 0-5 väliin 0.2 välein numerot. Loppuarvoon ei päästä.
print(x2)

a = 1
b = 6
amount = 50
nopat = np.random.randint(a,b+1,amount)
print(nopat) #generates random integers between a and b. "Note how the endpoint is not included here, so the second parameter must be b + 1. Normally distributed random numbers are obtained with the command randn (n), where n indicates the amount of numbers to be generated."

x = np.random.randn(100) #The random () command produces random numbers evenly distributed over a semi-open interval [0.0, 1.0). So these are decimal numbers. 
print(x)

x = np.random.random(10)
print(x)

#The numbers of dimensions and elements in the table are determined by the attributes ndim and size. E.g.
x.size
x.ndim
A.size
A.ndim

#Array can be read from the file with the genfromtxt command, which is given the name of the CSV file as a parameter. Additional parameters can be given as a value delimiter and whether the header line at the beginning of the file is skipped. E.g.
#data = np.genfromtext("data.csv",delimeter=",",skip_header=1)

#You can change the format of the table with the reshape (n, m) command, where n and m represent the new format of the table. However, the deformation must not change the number of embryos. E.g.
A = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
np.shape(A)
A.reshape(3,4)
A.reshape(2,3,2)
A.reshape(6,2)

#As the shape changes, it is important to keep in mind the order that the numpy uses in the transformation. The conversion is done on a row major order basis.
#You can repeat a row, column, or any table multiple times with the repeat () command. E.g.
A = np.repeat([[1,2,3]],4,axis=0)
B = np.repeat([[1],[2],[3]],3,axis=1)

#you should only make copies of tables, so that they don't affect each other
B = A.copy()

#Cutting a one-dimensional array is done in the usual way with square brackets by indexing. Remember that places are indexed from zero!
#The two-dimensional array is indexed (cut) by two indexes within one square bracket:
A = np.array([[1,2,3],[4,5,6]])
print(A[0,0])
print(A[0,1])

#The three-dimensional array is cut with three indexes, respectively.
#A negative index works in the same way as with lists, i.e. it counts from the end. A colon is used to cut an entire row or column. E.g.
print(A[:,0]) # fisrt row, ":"reads all rows
print(A[0,:]) # first column, ":" read all columns

#The colon can also be used to create index spaces with the start: end: step syntax, so that the intersection no longer targets the end. The index spacing can be used to index both rows and columns. E.g.
A = np.array([1,2,3,4,5,6,7,8,9])
print(A[0:6:1])
print(A[0:6:2])

#Items can be updated with a combination of an signing statement and a cut. E.g.
A = np.array([[1,2,3],[4,5,6]])
A[0,0] = 17
A[1,:] = [11,12,13]
print(A)

#Tables can be stacked with the vstack () command. E.g.
new = np.vstack((A,A))
print(new)

#This method can therefore be used to add horizontal rows to the beginning or end of a table. However, stackable tables must be compatible in shape. Horizontal stacking connects tables side by side. E.g.
new2 = np.hstack((A,A))
print(new2)

#Deleting a row and column is done with the delete command, to which information about the index of the row / column to be deleted and the dimension of the deletion operation must be passed (0 = for rows, 1 = for columns).
B = np.delete(A,[0],0)
#C = np.delete[A,[1],1]
print(B)

#A single item cannot be removed as it would make holes in the matrix.
#The table can be traversed (iterated) in many different ways. For example, one line at a time
A = np.array([1,2,3,4,5,6])
A = A.reshape(2,3)
n,m = np.shape(A)
for i in range(n):
    print("Rivi",i,"on",A[i,:])

#column by column
for j in range(m): 
    print("Column",j,"on",A[:,j])

#alkio kerrallaan
for i in range(n):
    for j in range(m):
        print("Element",i,j,"on",A[i,j])

#Another way to iterate the table item by item is to use a single iteration structure. This is especially useful for high-dimension arrays
for a in np.nditer(A):
    print(a)
#Notice how here again the order goes to the lines above (dimension / axis 0).

#As with regular numbers, calculations can also be performed on matrices. Addition and subtraction work on the tables item by item, ie the counterparts in the same place in the two matrices are added together (or subtracted from each other). Note that the list structure in Python does not support this kind of itemized calculations, Numpy library is needed. E.g.
A = np.array([[1,2],[3,4]])
B = np.array([[3,9],[4,-1]])
print(A+B, "Addition")
print(A-B, "minus")
print(A*B, "multiplication")
print(A/B, "Division")

#This way, you can create a large matrix that contains the desired number for each item (or use the full () command). E.g.
T = 5*np.ones((10,10))

print(T)

#These calculations, of course, require that the matrices A and B are the same size (shape the same) or that the other is a mere number. E.g.
print(A, "before")
print(B, "before")
print(A-1)
print(B+2)
print(A**2)

#There is also a completely unique multiplication (matrix product, matrix multiplication) for matrices, which does not work element by element but corresponds to the concept of multiplication in linear algebra. The matrix product AB is defined if the number of columns in A is the same as the number of rows in B (the exact definition can be found in Wikipedia, for example).
#Numpy's matmul () command or operator @ performs this multiplication. E.g.

print(np.matmul(A,B)) # tai A @ B

#For one-dimensional tables (i.e. vectors), dot () produces the point product (scalar product) of the vectors. In that case, the compatibility of the forms of income factors is not taken into account.
print(x.dot(x))

#It is also possible to calculate the matrix product between the matrix and the vertical vector, in which case an elementary product would not even be possible. E.g.
b = np.array([[5],[7]])
#print(np.matmul(A,print(b)))

#So note that matmul () / dot () does not compute an elementary product but something completely different (related to a linear algebra).
#In addition to the basic calculations, the application of elementary functions to the matrix is ​​done item by item. E.g.
np.sqrt(A) # the square root of each item
np.sin(A)
np.cos(A)
np.tan(A)
np.log(A)
np.exp(A)
np.log10(A)
np.log2(A)

#It is important to use the numpy library and not the math library, which also has elementary functions, but they do not work with matrices.

#Comparison operators also target matrices by item. E.g.
A > 1
A <= 0

#The result is a matrix formed from the truth values ​​True and False, to which the functions any () / all () can be assigned, which tells whether one of the truth values ​​is True and, respectively, whether all the values ​​are True. E.g.

print(np.all(A>1))
print(np.any(A>1))

#Operators can be used at the same time as the cut to select only items from a table that fulfill any of the conditions. E.g.
A = np.array([1,2,3,4,5,6,7])
B = A[A>3]
print(B)

#In practice, the solution of the group of equations is not done through the inverse matrix because it is quite a heavy operation for large matrices. Instead, other methods are used, we won’t go into details here. However, it is noted that numpy can solve a group of equations with the command solve (). For example, in the case of the previous situation
A = np.array([[2,1],[-4,3]])
b = np.array([11,3])
X = np.linalg.solve(A,b)
print(X)

#If necessary, the inverse matrix is ​​determined by the command inv () and can be used to check that the solution is correct.
Ainv = np.linalg.inv(A)
print(np.matmul(Ainv,b))



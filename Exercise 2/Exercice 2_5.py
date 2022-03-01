#Calculate the inverse matrix of matrix A and  check with the matrix product that 
# both AA−1 and A−1A produce a unit matrix with ones in diagonals and zeros elsewhere.
import numpy as np;

A = np.array([[1,2,3],[0,1,4],[5,6,0]])
print(A)
A_prod = np.prod(A)

#Inverse of matrix A
A_inv = np.linalg.inv(A)
print("The inverse of Matrix A is: ", "\n", A_inv)
print("A@A_inv: ", "\n", A@A_inv)


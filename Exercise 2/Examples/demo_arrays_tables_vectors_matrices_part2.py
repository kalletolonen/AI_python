#

import numpy as np


A = np.array([[1,2],[3,4]])
print(A)
B = np.array([[3,9],[4,-1]])
print(B)

# Basic Calculations
C = A + B
print("C=",C)
D = A - B
print("D=",D)
E = A * B
print("E=",E)
F = A / B
print("F=",F)
A2 = A**2
print("A2=",A2)


G = 5*A # 5A= A+A+A+A+A
print("G=",G)
I = 5*np.ones((2,3))
print("I=",I)
AA = 2 + A + 1
print("AA=",AA)

#########################
# Matrix Algebra
##########################
deta = np.linalg.det(A) # determinant
print("det(A)=",deta)

C = np.matmul(A,B) # matrix multiplication
print("C=",C)
print("E=",E)
C2 = A @ B
print("C2=",C2)

x = np.array([1,2,3])
y = np.array([0,1,0])
d = np.dot(x,y)
print("d=",d)
d2 = x*y
print("d2=",d2)

##########################################
# Basic functions
print("sqrt(A)=",np.sqrt(A))
print("sin(A)=",np.sin(A))
print("exp(A)=",np.exp(A))
print("log(A)=",np.log(A))

#################################
# Comparation
T = A > 1
print("T=",T)

print(np.all(T)) # all true?
print(np.any(T)) # one true?

# Cutting using logical condition
B = A[A < 3]
print("B=",B)


############################
# Statistic functions
############################
print("A=",A)
s = np.sum(A)
print("Sum is ",s)
print("Colum Sum is ",np.sum(A,axis=0))
print("Row Sum is ",np.sum(A,axis=1))

print("Product is ",np.prod(A))

print("Min is ",np.min(A))
print("Max is ",np.max(A))

print("Average is ",np.mean(A))
print("Median is ",np.median(A))

print("Standard Devision is ",np.std(A))
print("Variance is ",np.var(A))

sat = np.random.randn(1000)
print(sat.mean())
print(sat.std())

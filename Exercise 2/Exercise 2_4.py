import numpy as np
import matplotlib.pyplot as plt

A = np.array([[1,2],[3,4]])
B = np.array([[-1,1],[5,7]])

print(A)
print(B)

deta = np.linalg.det(A) # determinant A
print("det(A)=",deta)
detb = np.linalg.det(B) # determinant A
print("det(B)=",detb)

print(np.linalg.det(A @ B)) # determinant (AB) = det (A) * Det(B)
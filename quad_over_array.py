from scipy.integrate import quad
from scipy.linalg import expm
import numpy as np

def integrand(X, A, B):
    return np.dot(expm(A*X),expm(B*X))


A = np.array([[1, 2], [3, 4]])
B = np.array([[1, 2], [3, 4]])

#I= quad(integrand, 0, 1, args=(A,B))

def integrand(X, A, B, ix=None):
    """ pass ix=None to return the matrix, ix = 0,1,2,3 to return an element"""
    output = np.dot(expm(A*X),expm(B*X))
    if ix is None:
        return output
    i, j = ix//2, ix%2
    return output[i,j]
I= np.array([quad(integrand, 0, 1, args=(A,B, i))[0] for i in range(4)]).reshape(2,2)
print(I)

def integrand(x, a):
    return x**(a-1)*np.exp(-x)
a=3
I = quad(integrand, 0, np.inf, args=(a+1))   
print(I[0])
# I = a!

import warnings
warnings.filterwarnings("ignore")
x=np.arange(-1.0,5.0,0.1)
print(x)
y=list([quad(integrand, 0, np.inf, args=(a))[0] for a in x])
#print(x,y)

def fact(x):
    return np.sqrt(2*np.pi)*np.exp(-x)*x**(x+1/2) # about x!

n=10
print(fact(n))

I = quad(integrand, 0, np.inf, args=(n+1))   
print(I[0])
    


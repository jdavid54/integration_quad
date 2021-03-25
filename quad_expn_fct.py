# https://docs.scipy.org/doc/scipy/reference/tutorial/integrate.html
import numpy as np
from scipy.integrate import quad

def integrand(x, a, b):   #integre sur x; a,b = arguments
    return a*x**2 + b
a = 2
b = 1
I = quad(integrand, 0, 1, args=(a,b))
print('résultat',I[0])
print('précision',I[1])


def integrand(t, n, x):  #integre sur t, n,x arguments de integrale resultante
    return np.exp(-x*t) / t**n

def expint(n, x):
    return quad(integrand, 1, np.inf, args=(n, x))[0]

vec_expint = np.vectorize(expint)
expn2 = vec_expint(3, np.arange(1.0, 4.0, 0.5))
print(expn2)
#array([ 0.1097,  0.0567,  0.0301,  0.0163,  0.0089,  0.0049])

import scipy.special as special
expn1 = special.expn(3, np.arange(1.0,4.0,0.5))
print(expn1)
#array([ 0.1097,  0.0567,  0.0301,  0.0163,  0.0089,  0.0049])

n = 7
result = quad(lambda x: expint(n, x), 0, np.inf)
print(result)
#(0.33333333324560266, 2.8548934485373678e-09)

In = 1.0/n   # In = 1/n
print(In)
#0.333333333333

print(In - result[0])
#8.77306560731e-11




# double intégration dblquad
from scipy.integrate import dblquad
def I(n):
    return dblquad(lambda t, x: np.exp(-x*t)/t**n,
                   0, np.inf,     # bornes extérieures pour t
                   lambda x: 1, lambda x: np.inf)   # bornes intérieures pour x

print(I(4))
#(0.2500000000043577, 1.29830334693681e-08)
print(I(3))
#(0.33333333325010883, 1.3888461883425516e-08)
print(I(2))
#(0.4999999999985751, 1.3894083651858995e-08)

area = dblquad(lambda x, y: x*y, 0, 0.5, lambda x: 0, lambda x: 1-2*x)
print(area)
#(0.010416666666666668, 1.1564823173178715e-16)

from scipy.integrate import nquad
N = 5
def f(t, x):
    return np.exp(-x*t) / t**N

result = nquad(f, [[1, np.inf],[0, np.inf]])
print(result)
#(0.20000000000002294, 1.2239614263187945e-08)

# définition de bornes par fonction
def f(x, y):
    return x*y

def bounds_y():
    return [0, 0.5]

def bounds_x(y):
    return [0, 1-2*y]

print(nquad(f, [bounds_x, bounds_y]))
#(0.010416666666666668, 4.101620128472366e-16)

# simps
def f1(x):
   return x**2

def f2(x):
   return x**3

x = np.array([1,3,4])
y1 = f1(x)

from scipy.integrate import simps
I1 = simps(y1, x)
print(I1)
#21.0

y2 = f2(x)
I2 = simps(y2, x)
print(I2)
#61.5 and not correct 63.75
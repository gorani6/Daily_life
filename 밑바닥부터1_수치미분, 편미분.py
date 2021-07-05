def numerical_diff(f, x):
    h = 1e-4 #0.0001
    return (f(x+h) - f(x-h)) / (2*h)


def function_1(x):
    return 0.01*x**2 + 0.1*x

import numpy as np
import matplotlib.pylab as plt

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
plt.show()

def funtion_2(x):
    return x[0]**2 + x[1] **2

def funtion_tmp1(x0):
    return x0*x0 + 4.0**2.0

print(numerical_diff(funtion_tmp1, 3.0))

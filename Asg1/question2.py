#!/usr/local/bin/python3

import numpy as np
import math

# Part (a)
x = 1e-10
f = np.exp(x)-1. # accurate to 6 digits

x = 1e-5
f = np.exp(x)-1. # accurate to 11 digits

# Part (b)
x = 1e-10
f = x + .5*x*x
error = (x*x*x)/6.
rel_error = error/f     # error = 1.6666e-21

# Part (c)
epsilonlst = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15]

for epsilon in epsilonlst:
    if (np.abs(x) < epsilon):
        f = x + .5*x*x
        error = (x*x*x)/6.
        rel_error = error/f
        print("epsilon: " + str(epsilon) + " rel error: " + str(rel_error))
    else:
        f = np.exp(x)-1.
        rel_error = (x - f) / x
        print("epsilon: " + str(epsilon) + " rel error: " + str(rel_error))


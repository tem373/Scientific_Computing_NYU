import math
import numpy as np
import matplotlib.pyplot as plt

###################### QUESTION 2 ######################
# part (b)
# Solving for p
def convergence_analysis(true_E, E_n):
    rn = math.log(abs(E_n - true_E))    # base e
    print(rn)

def secant_method(E_nminus1,E_n,M,e):
    f_nminus1 = (E_nminus1 - e*math.sin(E_nminus1))
    f_n = (E_n - e*math.sin(E_n))
    E_nplus1 = E_n - (f_n * (E_n - E_nminus1))/(f_n - f_nminus1)
    epsilon = 0.01
    while (abs(E_n - E_nplus1) > epsilon):
        E_nplus1 = secant_method(E_n,E_nplus1,M,e)
        convergence_analysis(true_E, E_nplus1)
    return E_nplus1

# part (c)
def direct_iteration(E_n,M,e):
    E_nplus1 = M + e*math.sin(E_n)
    epsilon = 0.1
    while (abs(E_n - E_nplus1) > epsilon):
        E_nplus1 = direct_iteration(E_nplus1,M,e)
    return E_nplus1

# part (d)
def newton_solver(E_n,M,e):
    E_nplus1 = (E_n - e*math.sin(E_n) - M)/(1-e*math.cos(E_n))
    epsilon = 0.1
    while (abs(E_n - E_nplus1) > epsilon):
        E_nplus1 = newton_solver(E_nplus1,M,e)
    return E_nplus1

res = newton_solver(.9, .9,.1)
print(res)

###################### QUESTION 4 ######################

def hessian_gradient(f,x):
    pass

def newton_optimization(d,x):
    return d
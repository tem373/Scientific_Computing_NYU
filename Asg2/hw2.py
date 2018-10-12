#!/usr/local/bin/python3
import math
import numpy as np

import f


###################### QUESTION 1 ######################
def func(x):
    return math.exp(x)
    

def f_bar_k(h, k):
    
    return (1./h) * (exp((h/2.)) - exp((-h/2.)))


###################### QUESTION 3 ######################
def PanelIntegratorRect(n,a,b,f):
    #import f
    # Define the intervals
    x = a
    h = float(b-a)/n
    I = 0.0
    # Iterate through the panels
    for i in range(0, n):
        I += h*f(x)
        x += h
    return I


def PanelIntegratorTrap(n,a,b,f):
    #import f
    # Define the intervals
    x = a
    h = float(b-a)/n
    I = 0.0
    # Iterate through the panels
    for i in range(0, n):
        I += (h/2.)*(f(x) + f(x+h))
        x += h
    return I


def PanelIntegratorSimp(n,a,b,f):
    #import f
    # Define the intervals
    x = a
    h = float(b-a)/n
    I = 0.0
    # Iterate through the panels
    for i in range(0, n):
        I += (h/6.)*(f(x) + 4.*f(x + (h/2.)) + f(x+h))
        x += h
    return I


# Run experiment
true_value = 1. - math.cos(2.)
n = 100
# Rectangle
print("Rectangle Method")
for i in range(1, n+1):
    Integral = PanelIntegratorRect(i, 0, 2, f.f)
    rel_err = abs((Integral - true_value)/true_value)
    print("N: " + str(i) + " Relative Error: "  +str(rel_err))
    
# Trapezoid
print("Trapezoid Method")
for i in range(1, n+1):
    Integral = PanelIntegratorTrap(i, 0, 2, f.f)
    rel_err = abs((Integral - true_value)/true_value)
    print("N: " + str(i) + " Relative Error: "  +str(rel_err))

# Simpson
print("Simpson Method")
for i in range(1, n+1):
    Integral = PanelIntegratorSimp(i, 0, 2, f.f)
    rel_err = abs((Integral - true_value)/true_value)
    print("N: " + str(i) + " Relative Error: "  +str(rel_err))

# NOTES: Simpson's method is far more accurate (1e-11) than either the Trapezoid
# Method or the Rectangle Method, as predicted. All of the methods appear to
# have the claimed order of accuracy


###################### QUESTION 4 ######################
# Rectangle adaptive method
def adaptiveRect(n,a,b,epsilon, f):
    Ah = PanelIntegratorRect(n,a,b,f)
    Ah2 = PanelIntegratorRect(2*n,a,b,f)
    while (abs(Ah2-Ah) > 3*epsilon):
        n=n*2
        Ah = Ah2
        Ah2 = PanelIntegratorRect(2*n,a,b,f)
    return Ah2


# Simpson adaptive method
def adaptiveSimp(n,a,b, epsilon, f):
    Ah = PanelIntegratorSimp(n,a,b,f)
    Ah2 = PanelIntegratorSimp(2*n,a,b,f)
    while (abs(Ah2-Ah) > 3*epsilon):
        n=n*2
        Ah = Ah2
        Ah2 = PanelIntegratorSimp(2*n,a,b,f)
    return Ah2


# Run the functions
true_value = -0.0087295     # Wolfam-Alpha (k=100, r=100)
n = 100

print("Rectangle Method Adaptive Error")
for i in range(1, n+1):
    Integral = adaptiveRect(i, 0, 2, 0.001, f.f_harder)
    rel_err = abs((Integral - true_value)/true_value)
    print("N: " + str(i) + " Relative Error: "  +str(rel_err))

print("Simpson Method Adaptive Error")
for i in range(1, n+1):
    Integral = adaptiveSimp(i, 0, 2, 0.001, f.f_harder)
    rel_err = abs((Integral - true_value)/true_value)
    print("N: " + str(i) + " Relative Error: "  +str(rel_err))

# These methods appear to be able to estimate the integral to a reasonable 
# degree of accuracy, particularly as N->large. It should be noted that the
# Simpson method is far more accurate than the rectangle method, which is
# again unsurprising

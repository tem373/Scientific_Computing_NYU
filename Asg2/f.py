import math

def f(x):
    """ Problem 3"""
    return math.sin(x)


def f_harder(x):
    """ Problem 4"""
    k = 100
    r = 100
    f = (math.cos(k*x)) / (math.exp(((x ** 2)/(r ** 2))))
    return f


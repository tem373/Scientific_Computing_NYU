import cmath
import math
import numpy as np

def DFT(f: np.array):
    f = np.asarray(f)
    N = f.shape[0]
    fklist = []
    for k in range(N):
        fk = 0.0
        for n in range(N):
            fk += f[n] * cmath.exp(-1j * 2.0 * cmath.pi * k * n / N)
        fklist.append(fk/N)
    fhat = np.asarray(fklist)
    return fhat


def iDFT(fhat: np.array):
    fhat = np.asarray(fhat)
    N = fhat.shape[0]
    fnlist = []
    for n in range(N):
        fn = 0.0
        for k in range(N):
            fn += fhat[k] * cmath.exp(1j * 2.0 * cmath.pi * k * n / N)
        fnlist.append(fn)
    f = np.asarray(fnlist)
    return f

def tPOLY_INT(coeff, points, L):
    """ Trig polynomial interpolation"""
    coeff = np.asarray(coeff)
    N = coeff.shape[0]
    #assert( N % 2 == 1)     # check for odd number of points
    pxlist = []
    for x in points:
        px = 0.0
        for n in range(N):
            px = coeff[n] * cmath.exp(1j * 2.0 * cmath.pi * x * n / L)
        pxlist.append(px)
    p = np.asarray(pxlist)
    return p

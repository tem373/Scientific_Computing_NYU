import cmath
import math
import numpy as np
import DirectDFT as SFTW
import matplotlib.pyplot as plt
import scipy.fftpack as fftpack

################################## Question 1 ##################################
# Part (b)
fj = [np.sin(i*(np.pi/2)) for i in range(10)]
fjsq = [i ** 2 for i in fj]
fjsum = sum(fjsq)

fk = SFTW.DFT(fj)
fksq = [abs(i) ** 2 for i in fk]
fksum = sum(fksq)
# Here, we have that C = n^p and p=2 (it can be shown numerically)
# where we have n=10 in this example, C=10^2 = 100, so to show equality:
p = 1
C = 10 ** p
print("Question 1b")
print("Sum of fjs squared")
print(fjsum)
print("Sum of C * abs(f_hat) squared")
print(np.round((C * fksum), decimals=3))
print("\n")

# Part (c)
print("Question 1c")
f = [np.sin(i*(np.pi/2)) for i in range(10)]
g = [np.sin(i*(np.pi/2)) for i in range(10)]
g.pop(0)
g.append(f[-1])
fhat = SFTW.DFT(f)
ghat = SFTW.DFT(g)
closeness = []
for x in range(len(fhat)):
    closeness.append(abs(fhat[x] - ghat[x]))
print("Distances between DFT transforms")
print(closeness)
print("\n")

# Part (d)
print("Question 1d")
# Unaltered function f
f = [np.sin(i*(np.pi/2)) for i in range(10)]
print("Original Function")
print(np.round_(f, decimals=3))

# Function fhat after DFT
fhat = SFTW.DFT(f)
print("Fourier")
print(np.round_(fhat, decimals=3))

# Function f after iDFT(fhat)
f_rev = SFTW.iDFT(fhat)
print("Fourier Inverse")
print(np.round_(f_rev, decimals=3))     # Note the values are the same
print("\n")

# Part (e)
print("Question 1e")
fhat = SFTW.DFT(f)
print("F")
print(np.round_(fhat, decimals=3))

fhat2 = SFTW.DFT(fhat)
fhat3 = SFTW.DFT(fhat2)
fhat4 = SFTW.DFT(fhat3)
print("F4")
print(np.round_(fhat4, decimals=3))     # Note that this is equal to f * C^2
print("\n")                             # since the transformation was applied twice

################################## Question 2 ##################################
# Interval constants - Range = [-3,3] take n=100 samples
L = 6
n = 10
points = np.linspace((-L/2),(L/2),n, endpoint=False)
# Function setup
g = [(1 / (math.sqrt(2*np.pi))) * np.exp(-0.5 * (x ** 2)) for x in points]
h = [(0.5 * np.exp(-abs(x))) for x in points]
# DFT
ghat = np.fft.fft(g)
hhat = np.fft.fft(h)
g_hat = np.array([abs(x) for x in ghat])
h_hat = np.array([abs(x) for x in hhat])

# Plot
plt.xlabel("Input range (x)")
plt.ylabel("DFT coefficients")
plt.title("Plot of |ghat_k| and |hhat_k| in centered interval")
plt.plot(points, g_hat, points, h_hat)   #g=blue, h=orange
plt.show()
# NOTE: The plot shows that the coefficients of the DFT of g converge faster to
# zero than those of the DFT of h

################################## Question 3 ##################################
L = 6
n = 100
xvals = np.linspace((-L/2),(L/2),n, endpoint=False)

g = [(1 / (math.sqrt(2*np.pi))) * np.exp(-0.5 * (x ** 2)) for x in xvals]
h = [(0.5 * np.exp(-abs(x))) for x in xvals]

ghat = np.array(np.fft.fft(g))
hhat = np.array(np.fft.fft(h))

p_g = SFTW.tPOLY_INT(ghat, xvals, L)
p_h = SFTW.tPOLY_INT(hhat, xvals, L)

# Plot g
plt.xlabel("Input range (x)")
plt.ylabel("")
plt.title("G function and polynomial interpolation")
plt.plot(xvals, ghat, xvals, p_g)
plt.show()

# Plot h
plt.xlabel("Input range (x)")
plt.ylabel("")
plt.title("H function and polynomial interpolation")
plt.plot(xvals, hhat, xvals, p_h)
plt.show()

################################## Question 4 ##################################
L = 6
n = 50
xvals = np.linspace((-L/2),(L/2),n, endpoint=False)

g = [(1 / (math.sqrt(2*np.pi))) * np.exp(-0.5 * (x ** 2)) for x in xvals]
h = [(0.5 * np.exp(-abs(x))) for x in xvals]

ghat = np.array(np.fft.fft(g))
hhat = np.array(np.fft.fft(h))

p_g = SFTW.tPOLY_INT(ghat, xvals, L)
p_h = SFTW.tPOLY_INT(hhat, xvals, L)

# Part (a)
gdir = fftpack.diff(ghat, order=1)
hdir = fftpack.diff(hhat, order=1)
pgdir = fftpack.diff(p_g, order=1)
phdir = fftpack.diff(p_h, order=1)

gerr = []
herr = []
for x in range(len(gdir)):
    gerr.append(gdir[x] - pgdir[x])
    herr.append(hdir[x] - phdir[x])
gerr = np.asarray(gerr)
herr = np.asarray(herr)

# Plot g and h
plt.xlabel("Input Range (x)")
plt.ylabel("Error")
plt.title("Derivative Error of Functions g and h")
plt.plot(xvals, gerr, xvals, herr)
plt.show()

# Part (b)
Dkglist = [abs(x) for x in gerr]
Dkhlist = [abs(x) for x in herr]
# Plot g and h
plt.xlabel("Input Range (x)")
plt.ylabel("Error: Dk = max (abs(f'(x) - p'(x)))")
plt.title("Derivative Error of Functions g and h")
plt.plot(xvals, Dkglist, xvals, Dkhlist)
plt.show()

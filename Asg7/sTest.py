import numpy as np
import matplotlib.pyplot as plt
import math

# QUESTION 2 HOMEWORK 7

def sFit(xk, fk):
    # Create 4dx4d matrix
    Matrix = np.zeros((4*len(xk), 4*len(xk)), dtype=float)


    qk = np.linalg.solve(Matrix, fk)    # TODO: dimension mismatch
    K = np.linalg.cond(Matrix)
    return (K, qk)


def sEval(xk, qk, x):
    sum = 0.0
    for i in range(0,len(qk)):
        if (xk[i] <= x <= xk[i+1]):
            for j in [0,1,2,3]:
                sum += qk[i+j] * ((x - xk[i]) ** j)
    return sum



#def sTest():
#    pass


def sPlot(K_lst, d_lst):
    Karray = np.asarray(K_lst)   # Condition Numbers
    darray = np.asarray(d_lst)   # d

    # Take log of y-axis
    Karray = np.log(Karray)

    plt.xlabel("Nodes (d)")
    plt.ylabel("Cubic Spline Matrix Condition Number (K)")
    plt.title("Growth of Spline Matrix Condition Number vs Nodes")
    plt.plot(darray, Karray)
    plt.show()


def F(x):
    """ Problem function for Question 3"""
    return math.exp(-0.5 * (x ** 2))


def main():
    c = 200
    interpolpts = np.linspace(-2, 2, c, endpoint=True)

    # Lists for K-analysis
    K_lst = []
    d_lst = []

    # Create the xk and fk of length d
    dmax = 20
    for d in range(1, dmax):
        xk = np.linspace(-2, 2, d, endpoint=True)
        fk = np.asarray([F(x) for x in xk])

        # Fit
        (K, qk) = sFit(xk, fk)

        # Append for condition number analysis
        K_lst.append(K)
        d_lst.append(d)

        # Part (c): Plot the actual interpolating function with nodes
        sx_lst = []
        for iter, x in enumerate(interpolpts):
            sx = sEval(xk, qk, x)
            sx_lst.append(sx)
        sxarray = np.asarray(sx_lst)

        # Plotting
        if (d == dmax - 1):
            plt.xlabel("X values")
            plt.ylabel("F(x) and r(x)")
            plt.title("Radial Basis Function Interpolation of Function F(x)")
            plt.plot(interpolpts, sxarray, xk, fk)  # Interpolated polynomial, then original function
            plt.show()

        # Plot K-analysis (x-axis is d, lines are L)
    sPlot(K_lst, d_lst)


if __name__ == '__main__':
    main()
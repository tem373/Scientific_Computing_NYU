import numpy as np
import matplotlib.pyplot as plt
import math

# QUESTION 1 HOMEWORK 7

def pFit(xk, fk):
    # Vandermonde
    V = np.vander(xk, len(xk))
    K = np.linalg.cond(V)           # Condition Number
    pk = np.linalg.solve(V, fk)     # Coefficients

    return (K, pk)


def pEval(pk, x):
    # Horners rule
    px = pk[-1]
    i = len(pk) - 2
    while i >= 0:
        px = px * x + pk[i]
        i -= 1

    return px


#def pTest(): TODO: make this
#    pass


def pPlot(K_lst, d_lst):
    Karray = np.asarray(K_lst)   # Condition Numbers
    darray = np.asarray(d_lst)   # d

    # Take log of y-axis
    Karray = np.log(Karray)

    plt.xlabel("Nodes (d)")
    plt.ylabel("Vandermonde Matrix Condition Number (K)")
    plt.title("Growth of Vandermonde Condition Number vs Nodes")
    plt.plot(darray, Karray)
    plt.show()


def F(x):
    """ Problem function for Question 1"""
    return math.exp(-0.5 * (x ** 2))


def main():
    # Create the plotting xvals for the interpolating polynomial
    c = 200
    interpolpts = np.linspace(-2, 2, c, endpoint=True)      # Interval is [-2,2]

    # Lists for Condition Number analysis
    K_lst = []
    d_lst = []

    # Create the xk and fk of length d
    dmax = 20
    for d in range(1,dmax):
        xk = np.linspace(-2, 2, d, endpoint=True)
        fk = np.asarray([F(x) for x in xk])

        # Fit
        (K, pk) = pFit(xk, fk)

        # Append for the Condition Number analysis
        K_lst.append(K)
        d_lst.append(d)

        # Part (c): Plot the actual interpolating polynomial with nodes
        px_lst = []
        for iter, x in enumerate(interpolpts):
            px = pEval(pk, x)
            px_lst.append(px)
        pxarray = np.asarray(px_lst)

        # Plotting
        if(d==5):
            plt.xlabel("X values")
            plt.ylabel("F(x) and P(x)")
            plt.title("Polynomial Interpolation of Function F(x)")
            plt.plot(interpolpts, pxarray, xk, fk)      # Interpolated polynomial, then original function
            plt.show()

    # Part (b): Plot K-analysis
    pPlot(K_lst, d_lst)

    # NOTES: The error inside the range [-1,1] is relatively modest, however the error outside of that range
    # increases massively, as the plot shows.
    # Increasing d leads to an increase in the error and an increase in the condition number


if __name__ == '__main__':
    main()

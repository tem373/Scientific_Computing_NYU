import numpy as np
import matplotlib.pyplot as plt
import math

# QUESTION 3 HOMEWORK 7

def rFit(xk, fk, L):
    # Create radial basis matrix (d+1 by d+1)
    RBM = [[theta(x-x_k, L) for x_k in xk] for x in xk]
    wk = np.linalg.solve(RBM, fk)
    K = np.linalg.cond(RBM)
    return (K, wk)


def rEval(xk, wk, x, L):
    sum = 0.0
    for k in range(0,len(xk)):
        sum += wk[k] * theta(x-xk[k], L)
    return sum


#def rTest():
#    pass


def rPlot(K_lst, d_lst):
    Karray = np.asarray(K_lst)   # Condition Numbers
    darray = np.asarray(d_lst)   # d

    # Take log of y-axis
    Karray = np.log(Karray)

    plt.xlabel("Nodes (d)")
    plt.ylabel("Radial Basis Matrix Condition Number (K)")
    plt.title("Growth of RBM Condition Number vs Nodes")
    plt.plot(darray, Karray)
    plt.show()


def F(x):
    """ Problem function for Question 2"""
    return math.exp(-0.5 * (x ** 2))


def theta(x, L):
    return math.exp(-0.5 * ((x ** 2)/(L ** 2)))


def main():
    c = 200
    interpolpts = np.linspace(-2, 2, c, endpoint=True)

    # Lists for K-analysis
    K_lst = []
    d_lst = []
    #L_lst = [1,2,3,4]
    L = 2

    # Create the xk and fk of length d
    dmax = 20
    for d in range(1, dmax):
        xk = np.linspace(-2, 2, d, endpoint=True)
        fk = np.asarray([F(x) for x in xk])

        # Fit
        (K, wk) = rFit(xk, fk, L)
        # Once you have generated the vector of d+1 <w> coefficients you can pass those into rEval()
        # and evaluate any point

        # Append for condition number analysis
        K_lst.append(K)
        d_lst.append(d)

        # Part (c): Plot the actual interpolating function with nodes
        rx_lst = []
        for iter, x in enumerate(interpolpts):
            rx = rEval(xk, wk, x, L)
            rx_lst.append(rx)
        rxarray = np.asarray(rx_lst)

        # Plotting
        if (d == dmax-1):
            plt.xlabel("X values")
            plt.ylabel("F(x) and r(x)")
            plt.title("Radial Basis Function Interpolation of Function F(x)")
            plt.plot(interpolpts, rxarray, xk, fk)  # Interpolated polynomial, then original function
            plt.show()


    # Plot K-analysis (x-axis is d, lines are L)
    rPlot(K_lst, d_lst)

    # NOTES: Larger L leads to a "noisier" interpolation - smaller L leads to better results
    # K grows faster the larger L gets
    # However, too small L leads to catastrophic failure - you need a huge number of radial basis functions


if __name__ == '__main__':
    main()
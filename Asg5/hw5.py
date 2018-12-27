import numpy as np


###################### QUESTION 5 ######################
def mCh(H):
    """ Takes a numpy matrix as input and performs the MODIFIED Cholesky
        Decomposition on this matrix. This operation can take place whether
        the matrix in question is positive definite or not."""
    # Lower triangular factor
    L = [[0.0] * len(H) for i in range(len(H))]
    p = False               # True if positive definite else false

    # Determine value of p
    if (np.all(np.linalg.eigvals(H)) > 0):
        p = True
    else:
        p = False

    # Modified Cholesky Decomposition
    for i, (Hi, Li) in enumerate(H,L):
        for j, Lj in enumerate(L[:i+1]):
            s = sum(Li[k] * Lj[k] for k in range(j))
            Li[j] = sqrt(Ai[i] - s) if (i == j) else (1.0/Lj[j]) * (Ai[j] - s)

    return (L,p)

mat = [[25,15,-5],
       [15,18,0],
       [-5,0,11]]

print(mCh(mat))
import numpy as np
import time
import math
import matplotlib.pyplot as plt

###################### QUESTION 3 ######################
A_test = [[0,1],[-1,0]]


###################### QUESTION 4 ######################


###################### QUESTION 5 ######################


###################### QUESTION 6 ######################
def compare_matrix_methods(n):
    # Initialize matrices
    A = np.random.rand(n,n)
    B = np.random.rand(n,n)
    C = np.zeros((n,n))

    # Scalar Loop
    scalar_start = time.time()
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i,j] += A[i,k] * B[k,j]
    scalar_end = time.time()
    scalar_diff = scalar_end - scalar_start

    # Hand-Coded Vector Loop
    vector_start = time.time()
    for i in range(n):
        for j in range(n):
            C[i,j] = np.sum(A[i,:] * B[:,j])
    vector_end = time.time()
    vector_diff = vector_end - vector_start

    # Numpy linalg routine
    linalg_start = time.time()
    #C = np.linalg.matrixMultiply(A,B)
    C = np.matmul(A,B)
    linalg_end = time.time()
    linalg_diff = linalg_end - linalg_start

    # Output time comparison
    return (scalar_diff, vector_diff, linalg_diff)

# Run the simulations
max_n = 50
scalar_list = []
vector_list = []
linalg_list = []
x_range = range(1, max_n+1)
x_axis = np.array(x_range)
for n in range(1, max_n+1):
    (scalar, vector, linalg) = compare_matrix_methods(n)
    scalar_list.append(scalar)
    vector_list.append(vector)
    linalg_list.append(linalg)

scalar_np = np.array(scalar_list)
vector_np = np.array(vector_list)
linalg_np = np.array(linalg_list)

#print(x_axis)
#print(scalar_list)

# Graph using matplotlib
plt.plot(x_axis, scalar_np, 'r--', x_axis, vector_np, 'b--', 
         x_axis, linalg_np, 'g--')
plt.show()


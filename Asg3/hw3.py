import numpy as np
import time
import math
import matplotlib.pyplot as plt

###################### QUESTION 3 ######################
# Problem e
A_test = [[0,1],[-1,0]]

def calc_mat_exp(A,t):
    w,v = np.linalg.eig(A)
    Right = v
    Diag = np.diag(w)
    Left = np.linalg.inv(v)
    
    s = 0
    for i in range(0,100):
        s = np.add(s, (1./(math.factorial(i))*((t**i)*
                          np.linalg.matrix_power(Diag,i))), casting = "unsafe")
    result = np.matmul(np.matmul(Left,Diag),Right)
    return result

res = calc_mat_exp(A_test, 10)
print(res)              # Prints [[0.+0.j 0.+1.j], [0.+1.j 0.+0.j]]

###################### QUESTION 4 ######################
# Problem a
def hornersrule(A,t,k):
    I = np.identity(len(A))     # Create I
    p = np.identity(len(A))
    for i in range(k, -1,-1)[:-1]:
        p = np.matmul(np.add(I, t*(1/np.factorial(k))),p)
    return p

# Problem b
def b_n_method(B,n):
    B_next = B
    ops = 0             # Counter for operations
    i = 1
    exp = 2 ** i
    while(exp <= n):
        B_next = np.matmul(B,B)
        B = B_next
        ops += 1
        i += 1
        exp = 2 ** i
    print(ops)          # Prints 7
    return B_next
    
# Problem c
res1 = b_n_method(A_test, 128)
res2 = np.linalg.matrix_power(A_test, 128)
print(res1)             # Prints [[1 0], [0 1]]
print(res2)             # Prints [[1 0], [0 1]]

# NOTES: the number of operations (ops) here is 7 which is lg(128) = 7 so this
# algorithm runs on O(lg(n)). We can also see that it produces the correct 
# answer by verifying with the matrix_power() function, which agrees.

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

# Graph using matplotlib
plt.xlabel('dimension of the matrix (n)')
plt.ylabel('Runtime (ms)')
plt.title('Comparison of scalar, vector and linalg matrix multiplication')
plt.plot(x_axis, scalar_np, 'r--', x_axis, vector_np, 'b--', 
         x_axis, linalg_np, 'g--')
plt.show()


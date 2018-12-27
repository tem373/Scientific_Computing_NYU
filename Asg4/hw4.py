import math
import numpy as np
import matplotlib.pyplot as plt

###################### QUESTION 2 ######################
# part (b)
# Solving for p
def convergence_analysis(true, guesses):
    err = []
    for x in guesses:
        err.append(abs(x - true))
    rns = []
    for n in range(1,len(err)-1):
        rn = math.log(err[n+1]/err[n])/math.log(err[n]/err[n-1])
        rns.append(rn)
        print("{0:.4f}".format(rn))
    return rns

def secant_method(E_nminus1,E_n,M,e):
    f_nminus1 = (M - E_nminus1 + e*math.sin(E_nminus1))
    f_n = (M - E_n + e*math.sin(E_n))
    E_nplus1 = E_n - f_n/((f_n - f_nminus1)/(E_n-E_nminus1))
    epsilon = 0.00001
    guesses = []
    while (abs(E_n - E_nplus1)/abs(E_n) > epsilon):
        E_nplus1, E_n = secant_method(E_n,E_nplus1,M,e)
        guesses.append(E_nplus1)
        conv = convergence_analysis(9.811447, guesses)
    return (E_nplus1, E_n)

# part (c)
def direct_iteration(E_n,M,e):
    E_nplus1 = M + e*math.sin(E_n)
    epsilon = 0.00001
    guesses = []
    while (abs(E_n - E_nplus1)/abs(E_n) > epsilon):
        E_n = E_nplus1
        E_nplus1 = M + e*math.sin(E_n)
        guesses.append(E_nplus1)
        conv = convergence_analysis(9.811447, guesses)
    return E_nplus1

# part (d)
def newton_solver(E_n,M,e):
    E_nplus1 = E_n - (1/(-1+e*math.cos(E_n)))*(M - E_n + e*math.sin(E_n))
    epsilon = 0.00001
    guesses = []
    while (abs(E_n - E_nplus1)/abs(E_n) > epsilon):
        E_n = E_nplus1
        E_nplus1 = E_n - (1/(-1+e*math.cos(E_n)))*(M - E_n + e*math.sin(E_n))
        guesses.append(E_nplus1)
        conv = convergence_analysis(9.811447, guesses)
    return E_nplus1

# Testing Secant
print("Secant Method")
print("p-value")
res = secant_method(50000,10000,10,.5)
print("Answer: " + str(res[0]))   # Converges after 5 iterations

# Testing Direct
print("Direct Iteration")
print("p-value")
res = direct_iteration(500, 10,.5)
print("Answer: " + str(res))      # Converges after 20 iterations

print("Newton Method")
print("p-value")
res = newton_solver(-10, 10,.5)
print("Answer: " + str(res))      # Converges after 3 iterations

# NOTE: The secant method here converges faster than the direct method but 
# slower than the newton method


###################### QUESTION 4 ######################

def hessian_gradient(lam, x):
    gradient = []
    g_0 = 4*x[0] - 2*x[1] + lam*math.exp(x[0])
    gradient.append(g_0)
    for k in range(1,len(x)-1):
        g_x = 4*x[k] - 2*x[k-1] + 2*x[k+1] + lam*math.exp(x[k])
        gradient.append(g_x)

    g_d = 4*x[len(x)-1] - 2*x[len(x)-2] + lam*math.exp(x[len(x)-1])
    gradient.append(g_d)
    gradient_array = np.array(gradient)

    hessian = np.zeros((len(x),len(x)))
    for i in range(0,len(x)):
        for j in range(0,len(x)):
            if (i == j):
                hessian[i,j] = 4 + lam*math.exp(x[i])
            elif (abs(i-j) == 1):
                hessian[i,j] = -2
            else:
                hessian[i,j] = 0
    
    return (gradient_array, hessian)


def multi_newton_optimization(lamb, x_0):
    (grad, hess) = hessian_gradient(lamb,x_0)
    x_1 = np.add(x_0, np.matmul(-np.linalg.inv(hess), grad))
    epsilon = 0.0001
    while (np.linalg.norm(x_1 - x_0)/np.linalg.norm(x_0) > epsilon):
        x_0 = x_1
        x_1 = np.add(x_0, np.matmul(-np.linalg.inv(hess), grad))
        
    return x_1

dimensions = [2,3,4,5,6]
lambdas = [.1,.2,.3,.4,.5,.6,.7,.8,.9,1]
true_answers = [4714.988016952462,
                7870.83886416326,
                11060.674193730545,
                14080.369090542232,
                16858.541754835598]

ans_ls = [[],[],[],[],[]]

for dim in dimensions:
    x = np.zeros(dim)
    true = true_answers[dim-2]
    for lamb in lambdas:
        res = np.linalg.norm(multi_newton_optimization(lamb,x))
        rel_err = abs((res - true)/true)
        ans_ls[dim-2].append(rel_err)

l1 = np.array(ans_ls[0])
l2 = np.array(ans_ls[1])
l3 = np.array(ans_ls[2])
l4 = np.array(ans_ls[3])
l5 = np.array(ans_ls[4])
   
# Graph using Matplotlib
plt.xlabel('Lambda value')
plt.ylabel('Relative Error')
plt.title('Newton Optimization Algorithm vs. Dimensions, Lambda Values')
plt.plot(lambdas, l1, lambdas, l2, lambdas, l3, lambdas, l4, lambdas, l5)

plt.show()

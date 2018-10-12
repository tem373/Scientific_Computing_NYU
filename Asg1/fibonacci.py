#!/usr/local/bin/python3
import math

def fib(f0, f1, L):
    
    n_min1 = f0
    n = f1
     
    #print("F0: " + str(f0))
    #print("F1: " + str(f1))

    # Ascend the sequence until terminated by L
    for i in range(0, L):
        n_plus1 = n_min1 + n
        n_min1 = n
        n = n_plus1

    # Descend the sequence now
    n = n_min1
    n_min1 = n_plus1
    for j in range(0, L):
        n_plus1 = n_min1 - n
        n_min1 = n
        n = n_plus1
       
    #print("F1: " + str(n)) 
    #print("F0: " + str(n_plus1))
     
    # Calculating and printing the error
    fhat0 = n_plus1
    error = (fhat0 - f0) / f0
    print("Error: " + str(error))


# PART A
for i in range(1, 100):
    fib(math.sqrt(2), math.exp(1), 10)

# PART B
fib(1., 1., 76)
fib(1., 1., 77)    # THE fibonacci sequence
# 76 produces no error, however 77 produces an error

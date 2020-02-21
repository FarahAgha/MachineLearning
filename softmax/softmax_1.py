# formula for the Softmax function in Python.

import numpy as np

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    expL = np.exp(L)
    print("expL" , expL)
    sumExpL = sum(expL)
    print("Sum expL", sumExpL)
    result = []
    for i in expL:
        result.append(i*1.0/sumExpL)
    return result

l = [3,2,1,0]    
softmax(l)
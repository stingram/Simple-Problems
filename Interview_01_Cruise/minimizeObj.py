from typing import Callable, List

import numpy as np

def obj(x1, x2):
    return x1 * x2 + 2 * x1 * x1 + 3 * x2 * x2

# given objective function - write an optimizer to find local minima

# xy +2x^2 + 3y^2

# Two values of a feature vector

# One method, for example using SGD, we want to mimimize this function, 
# we could iteratively take the derivative of the function and "walk down the function" to a minimum
# we seek to find a place where the slope is the 2d space is zero
# x1 is data, and x2 is theta, one which we'd want to find to set the minimum

# Need partial derivatives for each step
# for x1
# d(obj)/d(x1) = 2*x2 + 2*x1 

# for x2
# d(obj/d(x2)) = 2*x1 + 6*x2 

# limited number of operations, +, *


class Solution(object):
    def optimize(func: Callable, x1, x2):
        
        # intialize x1, x2
        x1 = 5
        x2 = 5
        
        # learning rate
        alpha = 0.01
        
        
        # N iterations
        N = 1000
        
        
        # stop after N iterations or when the delta x1 and x2 is < 0.0001
        iter = 0
        tolerance = 0.0001
        
        x1_delta = float.inf
        x2_delta = float.inf
        
        while(iter < N and x2_delta > tolerance and x2_delta > tolerance):        
            
            # Gradient
            x1_partial = 2*x2 + 2*x1
            x2_partial = 2*x1 + 6*x2
        
        
            # compute new X
            new_x1 = x1 - x1_partial*alpha
            new_x2 = x2 - x2_partial*alpha
            
            # compute x deltas 
            x1_delta = np.abs(new_x1 - x1)
            x2_delta = np.abs(new_x2 - x2)
            
            
            # Update x
            x1 = new_x1
            x2 = new_x2
        
        return x1, x2
        
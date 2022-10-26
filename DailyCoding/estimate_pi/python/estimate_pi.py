# The area of a circle is defined as πr^2. Estimate π to 3 decimal places using a Monte Carlo method.

# Hint: The basic equation of a circle is x2 + y2 = r2.

# We know we can enclose a circle inside a square, we know that the equation for the area of square
# that encloses a circl whose radius = 1 is just 4*r**2. Since the area of the circle is pi*r**2, the
# ratio of these areas is A_square/A_circle = (4*r**2)/(pi*r**2), which is just 4/pi. If we decide to 
# randomly generate N x,y points from a uniform distribution from [-1,1], then the ratio of points inside
# the circle to the total number of points is also 4/pi. This means that given enough points, 
# 4/pi ~= N/N_points_in_circle. Now, we just need to generate enough points and keep track
# of when the points lie inside the unit circle, in this case, when our randomly generated points satisfy
# this conidition: x**2 + y**2 <= 1, we add that to our count of N_points_in_circle. Once we run the experiment
# N times, we can return our estimate of pi as 4*(N_points_in_circle/N).


import random

def estimate_pi(N: int = 1000):
    
    num_in_circle = 0
    for _ in range(N):
        x = random.uniform(-1,1)
        y = random.uniform(-1,1)
        if x**2 + y**2 < 1:
            num_in_circle += 1
            
    return 4*(num_in_circle/N) 

N = 100000
print(f"Pi estimate: {estimate_pi(N)}.")
import math

def perfect_squares(n: int):
    D = [0]*(n+1)
    
    for k in range(1, n+1):
        minD = math.inf
        for i in range(1, int(math.sqrt(k))+1):
            tmin = 1+D[k-i**2]
            if tmin < minD:
                minD = tmin
        D[k] = minD
    return D[n]



print(perfect_squares(99287))
    
    
    
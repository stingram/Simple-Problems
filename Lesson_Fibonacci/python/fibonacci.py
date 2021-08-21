# f(n) = f(n-1) + f(n-2)




def fib(n):
    if n==0:
        return 0
    if n==1:
        return 1
    n_2 = 0
    n_1 = 1
    res = 0
    for i in range(2,n+1):
        # update n
        res = n_1 + n_2
        
        # update n_2
        n_2 = n_1
        
        # update n_1
        n_1 = res
        
    return res


print(fib(10))
# 55
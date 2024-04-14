#
# Complete the 'minimalHeaviestSetA' function below.
#
# The function is expected to return an INTEGER_ARRAY.
# The function accepts INTEGER_ARRAY arr as parameter.
#

def minimalHeaviestSetA(arr):
    A = []
    arr.sort(reverse=True)
    L=0
    R=len(arr)
    A_curr=0
    B_curr=0
    while L<R:
        A_curr += arr[L]
        A.append(arr[L])
        while B_curr < A_curr and R > L:
            R -= 1
            B_curr += arr[R]
        # We broke the constraint so we need
        # to make it valid before starting the
        # loop again
        B_curr -= arr[R]
        R += 1
        L+=1
        
    return A[::-1]
def partition(arr, L, R):
    i = L - 1
    p = arr[R]
    for j in range(L,R):
        if arr[j] <= p:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i+1], arr[R] = arr[R], arr[i+1]
    return i+1

def quick_select(arr, k):
    
    # Kth LARGEST
    k = len(arr) - k
    
    L = 0
    R = len(arr) - 1
    
    while L <= R:
        p = partition(arr, L,R)
        if p == k:
            return arr[k]
        elif p > k:
            R = p - 1
        else:
            L = p + 1
    return -1

print(quick_select([8,7,2,3,4,1,5,6,9,0], 3))
def bst(arr):
    L=0
    R = len(arr)
    while L <= R:
        m = L+(R-1)//2
        if arr[m] == m:
            return m
        elif arr[m] > m:
            R = m - 1
        elif arr[m] < m:
            L = m + 1

    return -1


print(bst([-5, 1, 3, 4]))
# 1

print(bst([-5, 1, 3, 4]))
# 1
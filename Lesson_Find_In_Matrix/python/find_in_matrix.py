def find_in_matrix(matrix, val):
    if len(matrix) == 0:
        return False

    r = len(matrix)
    c = len(matrix[0])

    L = 0
    R = r*c-1
    
    while L < R:
        m = (R+(L-1))//2
        # convert m to r,c
        ri = m // c
        ci = m % c
        print(f"checking val: {matrix[ri][ci]} at {ri},{ci}")
        if matrix[ri][ci] == val:
            return True

        elif matrix[ri][ci] < val:
            L = m + 1
        elif matrix[ri][ci] > val:
            R = m
    return False


mat = [
    [1, 3, 5, 8],
    [10, 11, 15, 16],
    [24, 27, 30, 31],
]

print(find_in_matrix(mat, 4))
# False

print(find_in_matrix(mat, 10))
# True

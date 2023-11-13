from typing import List

def first_not_smaller(arr: List[int], target: int) -> int:
    # WRITE YOUR BRILLIANT CODE HERE
    L = 0
    R = len(arr)
    first = -1
    while L<=R:
        mid = L + (R-L)//2
        if(arr[mid] >= target):
            first = mid
            R = mid - 1
        else:
            L = mid + 1
    return first

arr = [1, 2, 2, 2, 2, 2, 2, 3, 5, 8, 8, 10]
target = 2
print(f"First element >= {target} is at index: {first_not_smaller(arr,target)}.")
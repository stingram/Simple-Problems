from typing import List

def binary_search(arr: List[int], target: int) -> int:
    # WRITE YOUR BRILLIANT CODE HERE
    left = 0
    right = len(arr) - 1
    while left<=right:
        mid = left + (right-left)//2
        if(arr[mid] == target):
            return mid
        elif(arr[mid] < target):
            left = mid + 1
        else:
            right = mid - 1
    return -1

arr = [1,3,5,7,8]
target = 9
print(f"Index of {target} is: {binary_search(arr,target)}.")
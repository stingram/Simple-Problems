from typing import List

def find_first_occurrence(arr: List[int], target: int) -> int:
    first = -1
    L = 0
    R = len(arr) - 1
    while L<=R:
        mid = L + (R-L)//2
        if(arr[mid] == target):
            first = mid
            R = mid - 1
        elif(arr[mid]<target):
            L = mid + 1
        else:
            R = mid - 1
    return first

target = 1
arr = [4, 6, 7, 7, 7, 20]
print(f"First occurrence of {target} is at index: {find_first_occurrence(arr,target)}.")
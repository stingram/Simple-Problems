from typing import List

def find_boundary(arr: List[bool]) -> int:
    # WRITE YOUR BRILLIANT CODE HERE
    L = 0
    R = len(arr)
    first = -1
    while(L<=R):
        mid = L + (R-L) // 2
        if(arr[mid] == True):
             R = mid - 1
             first = mid
        elif(arr[mid] == False):
            L = mid + 1
    
    return first


arr = [True,True,True,True,True]
print(f"First true is at index: {find_boundary(arr)}.")
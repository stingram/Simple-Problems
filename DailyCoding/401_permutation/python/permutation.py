# O(1) - Space
# O(N) - Time

from typing import List, Tuple

class Solution():
    def __init__(self):
        pass
    
    # need to process a cycle
    # need to move to next cycle
    
    # O(N)
    def _cycle(self, arr: List[int], P: List[int], start_loc: int, new_loc: int):

        old_loc = start_loc 
        while(new_loc!=-1):
            temp = arr[new_loc]
            arr[new_loc] = arr[start_loc]
            arr[start_loc] = temp
            
            P[old_loc] = -1
            old_loc = new_loc
            new_loc = P[new_loc]
    
    # O(N)   
    def _next_cycle(self, P: List[int]) -> Tuple[int, int]:
        for start_loc, new_loc in enumerate(P):
            if new_loc != -1:
                return start_loc, new_loc
        return -1,-1      
        
    # O(N), sequentials calls two functions that are each O(N), so O(2N) -> O(N)
    def permute(self, arr: List[str], P: List[int]) -> List[int]:
        # P - list of indices
        while(1):
            start_loc, new_loc = self._next_cycle(P)
            if start_loc != -1:
                self._cycle(arr, P, start_loc, new_loc)
            else:
                break
        return arr
    
    
P: List[int] = [2, 1, 0]
arr: List[str] = ["a", "b", "c"]
print(f"Permutation of array {arr} is : {Solution().permute(arr,P)}.")

P: List[int] = [4, 0, 1, 2, 3, 5, 6]
arr: List[str] = ["a", "b", "c", "d", "e", "f", "g"]
print(f"Permutation of array {arr} is : {Solution().permute(arr,P)}.")

P: List[int] = [4, 0, 1, 2, 3, 6, 5]
arr: List[str] = ["a", "b", "c", "d", "e", "f", "g"]
print(f"Permutation of array {arr} is : {Solution().permute(arr,P)}.")
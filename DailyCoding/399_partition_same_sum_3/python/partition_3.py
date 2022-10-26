# O(N) in space and time for solution


from typing import List, Optional

class Solution():
    def __init__(self):
        pass
    
    def partition(self, in_list: List[int])-> Optional[List[List[int]]]:
        if len(in_list) < 3:
            return None
        # Get sum
        S = sum(in_list)
        
        # check if divisivle by three
        if S % 3 != 0:
            return None
        
        sum_needed = S / 3
        # go through array subarray sum == S/3 is found 
        temp_sum = 0
        start_ind = 0
        FOUND_ALL_SEGMENTS = False
        ret_list = []
        for i, val in enumerate(in_list):
            temp_sum += val
            # if we get sum needed, then we add to result, reset temp_sum, and advance start_ind
            if temp_sum == sum_needed:
                ret_list.append(in_list[start_ind:i+1])
                temp_sum = 0
                start_ind = i+1
                
                # Once we get 2nd to last segment we ared done
                if FOUND_ALL_SEGMENTS:
                    ret_list.append(in_list[start_ind:])
                    return ret_list
                else:
                    FOUND_ALL_SEGMENTS = True
                    

in_list = [3, 5, 8, 0, 8]

print(f"Segments: {Solution().partition(in_list)}.")

                
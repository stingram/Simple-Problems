# Time complexity: O(N^2)
#Spacecomplexity: O(N) 
from typing import Dict, List


class Solution:
    def __init__(self):
        self.strobe_digits: Dict[int, int] = {'1':'1', '6':'9', '9':'6', '8':'8', '0':'0'}
    
    # O(N) in space and time
    def _is_strobe(self, i: int) -> bool:
        i_list = [x for x in str(i)]
        # print(f"i_list: {i_list}.")
        for start, end in enumerate(range(len(i_list)-1,-1,-1)):
            # print(f"start: {start}, end: {end}")
            if start > end:
                return True
            if i_list[start] not in self.strobe_digits or self.strobe_digits[i_list[start]] != i_list[end]:
                return False
        return True
    
    # O(N) in time
    def find_strobogrammatic_numbers(self, N: int) -> List[int]:
        strobes = []
        for i in range(pow(10,N-1),pow(10,N),1):
            if self._is_strobe(i):
                strobes.append(i)  
        return strobes

N = 4
print(f"All the strobogrammatic numbers with {N} digits are: {Solution().find_strobogrammatic_numbers(N)}.")
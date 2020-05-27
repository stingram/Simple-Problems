from typing import List

class Solution(object):
    def count_sort(self, a: List[int]) -> List[int]:
        counts = {}
        for element in a:
            if element in counts:
                counts[element] += 1
            else:
                counts[element] = 1
        
        offset = 0
        for i in range(3):
            if counts.get(i):
                for j in range(counts[i]):
                    a[j+offset] = i
                offset += counts[i]
        return a
    
    def v2_sort(self, a: List[int]) -> List[int]:
        p0 = 0
        p1 = 0
        p2 = len(a) - 1
        while p1 <= p2:
            # If we get a zero at p1, we swap with what's at p0
            # Then we increment our p0 and p1 pointers
            if a[p1] == 0:
                a[p0], a[p1] = a[p1], a[p0]
                p0 += 1
                p1 += 1
            # If we get a one at p1, we just advance p1
            elif a[p1] == 1:
                p1 += 1
            # If we get a two at p1, we swap with what's at p2
            # Then we decrement p2, but we leave p1 as is
            else:
                a[p0], a[p2] = a[p2], a[p0]
                p2 -= 1
        return a
    
    
a = [2,2,1,1,1,1]
print(Solution().count_sort(a))
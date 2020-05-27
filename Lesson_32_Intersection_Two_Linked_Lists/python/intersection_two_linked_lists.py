from collections import OrderedDict
from typing import List

class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None
    
class Solution(object):
    def find_intersection(self, a: ListNode, b: ListNode) -> ListNode:
        c = []
        visited = {} # OrderedDict(int)
        i = 0
        while(a):
            visited[i] = a
            a = a.next
            i += 1
        
        while(b):
            if b in visited.values():
                return b
            b = b.next
        
        return None
    
    def length(self, head: ListNode):
        if not head:
            return 0
        return 1 + self.length(head.next)
    
    def find_intersection_v2(self, a: ListNode, b: ListNode) -> ListNode:
        lenA, lenB = self.length(a), self.length(b)
        currA, currB = a, b

        # Advance longer path pointer
        if lenA > lenB:
            for _ in range(lenA-lenB):
                currA = currA.next
        else:
            for _ in range(lenB-lenA):
                currB = currB.next

        while(currA != currB):
            currA = currA.next
            currB = currB.next
            
        return currA


a = ListNode(10)
a.next = ListNode(8)
a.next.next = ListNode(7)
b = ListNode(10)
b.next = a.next
intersection = Solution().find_intersection_v2(a,b)

while(intersection):
    print(intersection.val)
    intersection = intersection.next

# Definition for singly-linked list.
class ListNode:
  def __init__(self, x):
    self.val = x
    self.next = None

  def __str__(self):
    result = str(self.val)
    if self.next:
      result += str(self.next)
    return result

class Solution(object):
    def reverseListRecursive(self, head: ListNode) -> ListNode:
        if head is None or head.next is None:
            return head
        p = self.reverseListRecursive(head.next)
        head.next.next = head
        head.next = None
        return p
    
    def reverseListIterative(self, head: ListNode) -> ListNode:
        prev = None
        curr = head
        while (curr != None):
            temp = curr.next
            curr.next = prev
            
            # shift pointers
            prev = curr
            curr = temp
            
        return prev
    
node = ListNode(1)
node.next = ListNode(2)
node.next.next = ListNode(3)

print(Solution().reverseListIterative(node))
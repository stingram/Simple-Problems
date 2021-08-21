from typing import Optional

class Node(object):
  def __init__(self, x):
    self.val = x
    self.next = None
    
class Solution:
    def addTwoNumbers(self, l1, l2):
        # return self.addTwoNumbersRecursive(l1, l2, 0)
        return self.addTwoNumbersIterative(l1, l2)
        
    def addTwoNumbersRecursive(self, l1, l2, c):
        val = l1.val + l2.val + c
        c = val // 10  # carry bit
        
        # Make Node for solution - single digit
        ret = (val % 10)
        
        # still have to add because one or both lists still have elements
        if l1.next != None or l2.next != None:
            if not l1.next:
                l1.next = Node(0)  # need a zero
            if not l2.next:
                l2.next = Node(0) # need a zero
            ret.next = self.addTwoNumbersRecursive(l1.next, l2.next, c)
        # both next were none so all we hace is c
        elif c:
            ret.next = Node(c)
        return ret
    
    def addTwoNumbersIterative(self, l1, l2):
        a = l1
        b = l2
        c = 0
        ret = current = None
        
        while a or b:
            
            # compute value
            val = a.val + b.val + c
            
            # compute carry digit
            c = val // 10
            
            # compute digit and set return and current node
            if not current:
                ret = current = Node(val % 10)  # first time in loop, we set ret and current
            else:
                current.next = Node(val % 10)  # not first time in loop, we set current.next
                current = current.next
            
            # checking next values
            if a.next or b.next:
                if not a.next:
                    a.next = Node(0)
                if not b.next:
                    b.next = Node(0)
            # update current.next with carry bit since a and b were both none, we'll
            # be done after this
            elif c:
                current.next = Node(c)
            
            # Set a and b for next iteration
            a = a.next
            b = b.next
        
        # Return start of list
        return ret

    def addTwoNumbersPractice(self, l1: Node, l2: Node):
        a = l1
        b = l2
        ret = current = None
        c = 0
        while a or b:
            val = a.val + b.val + c
            c = val // 10
            if not current:
                ret = current = Node(val % 10)
            else:
                current.next = Node(val % 10)
                current = current.next
            if a.next or b.next:
                if not a.next:
                    a.next = Node(0)
                if not b.next:
                    b.next = Node(0)
            elif c:
                current.next = Node(c)
            a = a.next
            b = b.next
        return ret

l1 = Node(5)
l1.next = Node(5)
l1.next.next = Node(5)

l2 = Node(5)
l2.next = Node(5)
l2.next.next = Node(5)

result = Solution().addTwoNumbers(l1, l2)
printres = ""
while result is not None:
    printres += str(result.val)
    result = result.next

print(printres[::-1])
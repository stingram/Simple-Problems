from collections import deque

class Node(object):
  def __init__(self, val, children):
    self.val = val
    self.children = children
    
class Solution:
    def level_by_level(self, root: Node) -> str:
        q = deque()
        q.append(root)
        result = str(root.val) +"\n"
        while q:
            # We use this num to know how many items in the queue we
            # must process before making a new line
            num = len(q)
            while num > 0:
                s = q.popleft()
                # print(s.val)
                for n in s.children:
                    q.append(n)
                    result += str(n.val)
                num -= 1
            result += "\n"
            
        return result
    
    
tree = Node('a', [])
tree.children = [Node('b', []), Node('c', [])]
tree.children[0].children = [Node('g', [])]
tree.children[1].children = [Node('d', []), Node('e', []), Node('f', [])]

print(Solution().level_by_level(tree))
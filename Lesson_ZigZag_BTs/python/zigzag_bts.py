class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right 


def zz_helper(level, node, res):
    if node == None:
        return ""
    
    res += str(node.val)
    if level % 2 == 0: # going left
        print("go left")
        res += zz_helper(level+1, node.right,"")
        res += zz_helper(level+1, node.left, "")
    else:
        print("go right")
        if node.left:
            res += zz_helper(level+1, node.left,"")
        if node.right:
            res += zz_helper(level+1, node.right,"")
    
    return res


def zz(node):
    return zz_helper(0, node, "")



def zz_v2(node):
    
    # initialize  data structures and result
    C = [node]
    N = []
    res = ""
    level = 0
    
    while(len(C)>0):
        
        print("level: {l}".format(l=level))
        
        # push all children in C stack to N stack
        if level % 2 == 0: # go left
            while(len(C)>0):
                node = C.pop()
                if node.right:
                    N.append(node.right)
                    print(type(N))
                if node.left:
                    N.append(node.left)
        
                # add all values in C Stack to res        
                res += str(node.val)
        else:
            while(len(C)>0):
                node = C.pop()
                if node.left:
                    N.append(node.left)
                if node.right:
                    N.append(node.right)
                
                res += str(node.val)
        
        # done with this level
        # set C = N, empty N
        C = N
        N = []
        level += 1
            
    return res



n7 = Node(7)
n6 = Node(6)
n5 = Node(5)
n4 = Node(4)
n3 = Node(3, n6, n7)
n2 = Node(2, n4, n5)
n1 = Node(1, n2, n3)

print(zz_v2(n1))
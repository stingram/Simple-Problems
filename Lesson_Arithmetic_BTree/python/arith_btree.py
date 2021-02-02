class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right



def arith_btee(node: Node):
    if node is None:
        return 0

    operators = {'+': lambda a,b : a+b,
                 '-': lambda a,b : a-b,
                 '*': lambda a,b : a*b,
                 '/': lambda a,b : a/b}

    if node.val in operators:
        return operators[node.val](arith_btee(node.left), arith_btee(node.right))
    else:
        return node.val




node = Node('*')
node.left = Node('+')
node.right = Node('+')
node.left.left = Node(3)
node.left.right = Node(2)
node.right.left = Node(4)
node.right.right = Node(5)
# (3+2)*(4+5) = 45
print(arith_btee(node))
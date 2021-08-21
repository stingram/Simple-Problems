class Node(object):
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

    def __repr__(self):
        result = str(self.val)
        if self.left:
            result = result + str(self.left)
        if self.right:
            result = result + str(self.right)
        return result

def gen_helper(available_values):
    
    # return list with None
    if len(available_values) == 0:
        return [None]
    
    # return list with one node
    if len(available_values) == 1:
        return [Node(available_values[0])]
    
    # build all available bsts
    bsts = []
    for n in available_values:
        
        # build all possible left and right bsts for the node
        lefts = gen_helper(range(available_values[0], n)) # only put smaller values into left subtree
        rights = gen_helper(range(n+1, available_values[-1]+1)) # only put larger values into right subtree
    
        # build all the tree combinations from the lists of lefts and rights
        for left in lefts:
            for right in rights:
                n = Node(n)
                n.left = left
                n.right = right
                
                # add tree to valid list of bsts
                bsts.append(n)
    # return bsts
    return bsts
        

def generate_bsts(N):
    trees = []
    if N > 1:
        trees = gen_helper(range(1,N+1))
        return trees
    return Node(1)

N = 3

print(generate_bsts(N))
print(len(generate_bsts(N)))
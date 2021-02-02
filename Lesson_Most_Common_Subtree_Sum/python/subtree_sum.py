from collections import defaultdict

class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# def subtree_sums(node: Node):
#     if node is None:
#         return [], 0

#     res = []
#     lres, l = subtree_sums(node.left)
#     rres, r = subtree_sums(node.right)
#     res.extend(lres)
#     res.extend(rres)
#     res.append(l+r+node.val)

#     return res, l+r+node.val

def subtree_sums(node: Node, counter):
    if node is None:
        return 0

    total = subtree_sums(node.left, counter) + subtree_sums(node.right,counter) + node.val
    counter[total] +=1

    return total


def most_freq_subtree_sum(node: Node):
    s_dict = defaultdict(int)
    subtree_sums(node, s_dict)
    
    # build dictionary of values
    # s_dict = {}
    # for s in sums:
    #     if s not in s_dict:
    #         s_dict[s] = 1 
    #     else:
    #         s_dict[s] += 1

    # find mode
    count = 0
    most_freq = 0
    for k, v in s_dict.items():
        if v > count:
            count = v
            most_freq = k

    return most_freq


root = Node(3, Node(1), Node(-3))
#print(most_freq_subtree_sum(root))
# 1

print(most_freq_subtree_sum(root))
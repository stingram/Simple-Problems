# You are given a binary tree in a peculiar string representation.
# Each node is written in the form (lr), where l corresponds to
# the left child and r corresponds to the right child.

# If either l or r is null, it will be represented as a zero.
# Otherwise, it will be represented by a new (lr) pair.

# Here are a few examples:

# A root node with no children: (00)
# A root node with two children: ((00)(00))
# An unbalanced tree with three consecutive left children: ((((00)0)0)0)
# Given this representation, determine the depth of the tree.


def count_height(encoded_bt: str) -> int:
    max_depth = 0
    curr = 0
    for c in encoded_bt:
        if c == "(":
            curr += 1
            if curr > max_depth:
                max_depth = curr
        elif c == ")":
            curr -= 1
    return max_depth

tests = ["(00)",
         "((00)(00))",
         "((((00)0)0)0)"]
for test in tests:
    print(f"depth: {count_height(test)}")
from typing import Dict

class Trie:
    def __init__(self,level=0):
        self.end = False
        self.level = level
        self.children: Dict[str,Trie] = {}

    def __repr__(self):
        indent = '  ' * self.level
        children_repr = ', '.join(f"{char}: {child}" for char, child in self.children.items())
        return f"{'END ' if self.end else ''}Level {self.level} {{ {children_repr} }}"


    def add(self,char):
        if self.level > 2:
            return
        node = self
        for _,child in node.children.items():
            child.add(char)
        
        if char not in node.children:
            self.children[char] = Trie(self.level+1)



# vals = ["a", "b", "c", "a", "a"]
vals = ["a", "a", "a", "b"]
trie = Trie()
for val in vals:
    trie.add(val)

print(f"{trie}")
# abc
# bca
# caa
# 
class Node:
    def __init__(self):
        self.count = 0
        self.children = {}
        
class Trie:
    def __init__(self):
        self.root = Node()
        
    def insert(self, word):
        node = self.root
        for c in word:
            if c not in node.children:
                node.children[c] = Node()    
            node = node.children[c]
            node.count += 1
            
    def unique_prefix(self, word):
        node = self.root
        prefix = ''
        for c in word:
            if node.count == 1:
                return prefix
            else:
                prefix += c
                node = node.children[c]
        return prefix
        
        

def shortest_unique_prefix(words):
    
    trie = Trie()
    
    for word in words:
        trie.insert(word)
        
    res = []
    for word in words:
        res.append(trie.unique_prefix(word))
    
    return res
        
        

words = []
print(shortest_unique_prefix(['jon', 'john', 'jack', 'techlead']))
class Node:
    def __init__(self, isWord, children):
        self.isWord = isWord
        # children is a dictionary, example:
        # {'a': Node(), 'b':Node(), etc.}
        self.children = children
        
class Solution(object):
    def build(self, words):
        # make root node
        trie = Node(False, {})
        for word in words:
            # set current back to root
            current = trie
            # add all characters to trie
            for char in word:
                if char not in current.children:
                    # make new node
                    current.children[char] = Node(False, {})
                # go down trie
                current = current.children[char]
            # we finished adding word so current node
            # isWord flag can be set to True
            current.isWord = True
        self.trie = trie
    
    def autocomplete(self, word):
        current = self.trie
        # go down trie for prefix
        for char in word:
            if char not in current.children:
                return [] # couldn't even find prefix
            current = current.children[char]
        # NOW WE CAN DO DFS
        words = []
        self._dfs_iterative(current, word, words)
        return words
    
    def _dfs(self, node, prefix, words):
        if node.isWord:
            words.append(prefix)
        for char in node.children:
            # call dfs
            self._dfs(node.children[char], prefix+char, words)
    
    def _dfs_iterative(self, node, prefix, words):
        # for iterative dfs, we need to keep track of 
        # node and prefix
        stack = [(node, prefix)]
        while stack:
            # pop from stack
            (node, prefix) = stack.pop()
            
            # process node, in this case,
            # we check if the current node is a word
            # if it is, we can append prefix to list
            # of words
            if node.isWord:
                words.append(prefix)
                
            # push children to stack
            for char in node.children:
                stack.append((node.children[char], prefix+char))
            
    
    
s = Solution()
s.build(['dog', 'dark', 'cat', 'door', 'dodge'])
print(s.autocomplete('do'))
# ['dog', 'door', 'dodge']
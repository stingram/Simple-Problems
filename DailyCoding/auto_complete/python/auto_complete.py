# Implement an autocomplete system. That is, given a query string s and 
# a set of all possible query strings, return all strings in the set that have s as a prefix.

# For example, given the query string de and 
# the set of strings [dog, deer, deal], return [deer, deal].

# Hint: Try preprocessing the dictionary into a more
# efficient data structure to speed up queries.

from typing import Dict, List

class Node(object):
    def __init__(self, is_word: bool, children: Dict[str, 'Node']):
        self.is_word = is_word
        self.children = children

class Trie:
    def __init__(self):
        self.root = Node(False,{})
        
    def build_trie(self, words: List[str]):
        for word in words:
            
            # reset to top when we want to add a word
            curr_node = self.root
            
            # start looping over word
            for char in word:
                # find where this character fits
                if char not in curr_node.children:
                    curr_node.children[char] = Node(False, {})
                curr_node = curr_node.children[char]
            # Finished word, so set is_word to true,
            # This is flag in Node for last character
            curr_node.is_word = True
        
    def _dfs_traversal(self, node: Node, prefix: str, words: List[str]):
        stack = [(node, prefix)]
        
        while stack:
            (curr_node, prefix) = stack.pop()
            
            if curr_node.is_word:
                words.append(prefix)
            
            for char in curr_node.children:
                stack.append((curr_node.children[char],prefix+char))
        
    def find_words(self, prefix: str) -> List[str]:
        words = []
        
        curr_node = self.root
        
        # get to position in trie
        for char in prefix:
            if char in curr_node.children:
                curr_node = curr_node.children[char]
            # Couldn't even find the prefix
            else:
                return words
                
        # from this node do dfs for all words in this sub-trie
        self._dfs_traversal(curr_node, prefix, words)
        
        return words
        
words = ['dog', 'deer', 'deal']
trie = Trie()
trie.build_trie(words)
print(f"Words starting with 'de': {trie.find_words('de')}.")
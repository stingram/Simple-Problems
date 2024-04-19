from typing import List, Dict

class Trie:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

    def add_word(self, word):
        node = self
        for char in word:
            if char not in node.children:
                node.children[char] = Trie()
            node = node.children[char]
        # this will update "last" node 
        node.is_end_of_word = True

class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        res = 0
        trie = Trie()
        for bad_word in set(forbidden):
            trie.add_word(bad_word)

        n = len(word)
        right = n - 1
        for left in range(n-1,-1,-1):
            current_node = trie
            for k in range(left,min(left+10,right+1)):
                c = word[k]
                if c not in current_node.children:
                    break
                current_node = current_node.children[c]
                if current_node.is_end_of_word:
                    right = k - 1
                    break
            res = max(res,right-left+1)
        return res

s = "cbaaaabc"
bad_words = ["aaa","cb"]
print(f"{Solution().longestValidSubstring(s, bad_words)}") # 4

s = "leetcode"
bad_words = ["de","le", "e"]
print(f"{Solution().longestValidSubstring(s, bad_words)}") # 4

s = "a"
bad_words = ["n"]
print(f"{Solution().longestValidSubstring(s, bad_words)}") # 1

s = "acbc"
bad_words = ["cbc","acb","acb","acbc",]
print(f"{Solution().longestValidSubstring(s, bad_words)}") # 1
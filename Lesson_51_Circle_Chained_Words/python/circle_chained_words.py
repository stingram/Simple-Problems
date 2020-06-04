import collections
from typing import List
class Solution:
    def circle(self, words: List[str]) -> bool:
        start = {}
        end = {}
        pairs = {}
        
        for word in words:
            if word[0] not in pairs:
                pairs[word[0]] = (1,0)
            else:
                pairs[word[0]] = (pairs[word[0]][0] + 1, pairs[word[0]][1])
                
            if word[-1] not in pairs:
                pairs[word[-1]] = (0,1)
            else:
                pairs[word[-1]] = (pairs[word[-1]][0], pairs[word[-1]][1] + 1)
                
        for k,v in pairs.items():
            if v[0] != v[1]:
                return False
        return True
    
    def is_cycle_dfs(self, symbol, current_word, start_word, length, visited):
        if length == 1:
            return start_word[0] == current_word[-1]

        visited.add(current_word)
        for neighbor in symbol[current_word[-1]]:
            if (neighbor not in visited and
                self.is_cycle_dfs(symbol, neighbor, start_word, length - 1, visited)):
                return True
        visited.remove(current_word)
        return False

    def chainedWords(self, words):
        symbol = collections.defaultdict(list)
        for word in words:
            symbol[word[0]].append(word)

        return self.is_cycle_dfs(symbol, words[0], words[0], len(words), set())
            
words = ["eggs","apple","snack","tuna", "karot","eggs","apple","snack","tuna", "karot"]
print(Solution().circle(words))

words = ['apple', 'eager', 'ege', 'raa']
print(Solution().circle(words))

words = ['apple', 'eager', 'ege', 'raa']
print(Solution().chainedWords(words))


from typing import List, Dict
import collections

class Solution(object):
    
    def _is_anagram(self, s1: str, s2: str) -> bool:
        s1 = sorted(list(s1))
        s2 = sorted(list(s2))
        for i in range(len(s1)):
            if s1[i] != s2[i]:
                return False
        return True

    def group_anagrams(self, words: List[str]) -> List[str]:
        anagrams = {}
                
        # compare words to each other
        for i in range(len(words)):
            for j in range(i+1,len(words)):
                if i != j:
                    is_anagram = self._is_anagram(words[i], words[j])
                    if is_anagram:
                        if words[i] not in anagrams:
                            anagrams[words[i]]=[words[i]]
                        anagrams[words[i]].append(words[j])
                        
        return [list(set(v)) for k,v in anagrams.items()]
    
    def group_anagrams_v2(self, words: List[str]) -> List[str]:
        groups = collections.defaultdict(list)
        for word in words:
            groups["".join(sorted(word))].append(word)
        return groups.values()
    
    
    def _hash_key(self, s: str):
        return "".join(sorted(str))
    
    def _hash_key_v2(self, s: str):
        arr = [0]* 26
        for char in s:
            arr[ord(char)- ord('a')] = 1
        return tuple(arr)
    
    def group_anagrams_v3(self, words: List[str]) -> List[str]:
        groups = collections.defaultdict(list)
        for word in words:
            hash_key = self._hash_key_v2(word)
            groups[hash_key].append(word)
        return groups.values()
    
    
test = ["abc", "bcd", "cba", "cbd", "efg"]
print(Solution().group_anagrams(test))
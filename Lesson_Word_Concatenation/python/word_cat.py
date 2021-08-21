# Time Complexity
# Space Complexity

# Set in python is implemented as a had table, so lookups are O(1),
# unless the load factor is too high, then with collisions, look-up time can be O(n)

 
class Solution(object):
    def find_concatenated_words(self, words):
        words_dict = set(words)
        cache = {}
        return [word for word in words if self._can_form(word, words_dict, cache)]
    
    def _can_form(self, word, word_dict, cache): 
        if word in cache:
            return cache[word]
        for index in range(1, len(word)):
            prefix = word[:index]
            suffix = word[index:]
            
            if prefix in word_dict:
                if suffix in word_dict or self._can_form(suffix, word_dict, cache):
                    cache[word] = True
                    return True
        cache[word]=False
        return False
            
     
words = ["cat", "cats", "dog", "catsdog"]
print(Solution().find_concatenated_words(words))
# ["catsdog"]
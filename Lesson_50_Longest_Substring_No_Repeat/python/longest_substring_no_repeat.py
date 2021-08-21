class Solution:
    def longest_substring_v2(self, input: str) -> str:
        letters= {}
        tail = -1
        result = 0
        for i in range(len(input)):
            if input[i] in letters:
                tail = max(tail, letters[input[i]])
            letters[input[i]] = i
            result = max(result, i-tail)
        return result
    
string_in = "abcabcbb"
print(Solution().longest_substring_v2(string_in))

string_in = "pwwkewxyzawlmnopq"
print(Solution().longest_substring_v2(string_in))
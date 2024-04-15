# Given a string, find the length of the longest substring
# in it with no more than K distinct characters.

def longest_substring(s: str, k: int) -> int:
    window = {}
    l=0
    longest = 0
    distinct_count = 0 
    for r in range(len(s)):
        
        # add this character to our window
        if s[r] not in window:
            window[s[r]] = 0
            distinct_count += 1
        window[s[r]] += 1
        
        # advance l such that window is valid
        while distinct_count > k:
            window[s[l]] -= 1
            if window[s[l]] == 0:
                del window[s[l]]
                distinct_count -= 1    
            l += 1
        
        # update window_length and longest
        window_length = r-l+1
        if window_length > longest:
            longest = window_length
    
    return longest

s = "abcdeffg"
k = 3

print(f"{longest_substring(s,k)}")
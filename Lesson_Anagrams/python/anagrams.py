from collections import defaultdict

def find_anagrams(s1, s2):
    inds = []
    char_map = defaultdict(int)
    
    # build character map, we use another loop
    # to remove/add characters from this char_map
    for c in s2:
        char_map[c] += 1
        
    # loop through and once we have char_map empty,
    # then we have an anagram of s2
    for i in range(len(s1)):
        
        # We need to add back characters once our index
        # is larger than the size of s2
        if i >= len(s2):
            c_old = s1[i-len(s2)]
            char_map[c_old] += 1
            
            if char_map[c_old] == 0:
                del char_map[c_old]
        
        # get current character
        c = s1[i]
        
        # subtract from our character_map
        char_map[c] -= 1
        
        # if we have zero count, we remove it
        if char_map[c] == 0:
            del char_map[c]
            
        # if have an empty char_map, then we know we found an
        # anagram
        if i+1 >= len(s2) and len(char_map) == 0:
            inds.append(i-len(s2)+1)
    
    return inds


print(find_anagrams('acdbacdacb', 'abc'))
# [3, 7]
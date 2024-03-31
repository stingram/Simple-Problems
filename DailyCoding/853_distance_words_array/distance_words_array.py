# Find an efficient algorithm to find the smallest distance
# (measured in number of words) between any two given words in a string.
# assume both words are guaranteed to exist in the string

# For example, given words "hello", and "world" and a text content
# of "dog cat hello cat dog dog hello cat world", return 1 because
# there's only one word "cat" in between the two words.


from math import inf

def find_distance(words: str, w1: str, w2: int) -> int:
    # create list of all words in the string
    words = words.split(" ")
    
    # create a list of indices for each target word
    w1s = [i for i, word in enumerate(words) if word == w1]
    w2s = [i for i, word in enumerate(words) if word == w2]
    
    w1_ind, w2_ind = 0,0
    min_dist = inf
    # linear search to find closest word distance
    while w1_ind < len(w1s) and w2_ind < len(w2s):
        p1 = w1s[w1_ind]
        p2 = w2s[w2_ind]
        dist = abs(p1-p2)
        if dist == 1:
            return 0
        if dist < min_dist:
            min_dist = dist
        # advance smaller of the two
        if p1<p2:
            w1_ind += 1
        else:
            w2_ind += 1
    
    # got to end of list2 before the list1
    while w1_ind < len(w1s):
        p1 = w1s[w1_ind]
        p2 = w2s[-1]
        dist = abs(p1-p2)
        if dist == 1:
            return 0
        if dist < min_dist:
            min_dist = dist
        w1_ind += 1

    # got to end of list1 before the list2
    while w2_ind < len(w2s):
        p1 = w1s[-1]
        p2 = w2s[w2_ind]
        dist = abs(p1-p2)
        if dist == 1:
            return 0
        if dist < min_dist:
            min_dist = dist
        w2_ind += 1
    
    return min_dist-1

s = "dog cat hello cat dog dog hello cat world"
w1 = "hello"
w2 = "world"

# should return 1 since only 1 word (cat) is between hello and world
print(f"{find_distance(s,w1,w2)}") 
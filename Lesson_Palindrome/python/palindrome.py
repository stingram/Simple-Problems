from collections import Counter
def find_palindrome(word):
    c = Counter()
    res = ""
    # build count of all letters
    for char in word:
        c[char]+= 1
        
    print(c)
    # start building output and stop if unable to complete
    num_odd = 0
    for char, count in c.items():
    
        if count % 2 != 0:
            num_odd+=1
        # can only have one character that occurs an odd number of times
        if num_odd > 1:
            return False
        
        # now we start building result based on char counts
        #so for even count, we place count/2 elements into result
        # one at the beginning and one at the end
        if count % 2 == 0:
            while count > 0:
                res = char + res + char
                count -=2 
        # for an odd count, then we place count//2 + 1elements into result
        # one at the beginning, one at the end, one in the middle 
        else:
            while count > 1:
                res = char + res + char
                count -= 2
            l = int(len(res)/2)
            res = res[:l] + char + res[l:]
        
    return res
        
        
        
        
        
        
print(find_palindrome('foxfo'))
# foxof
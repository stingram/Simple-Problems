lettersMaps = {
    1: [],
    2: ['a', 'b', 'c'],
    3: ['d', 'e', 'f'],
    4: ['g', 'h', 'i'],
    5: ['j', 'k', 'l'],
    6: ['m', 'n', 'o'],
    7: ['p', 'q', 'r', 's'],
    8: ['t', 'u', 'v'],
    9: ['w', 'x', 'y', 'z'],
    0: []
}

validWords = ['dog', 'fish', 'cat', 'fog']

def make_words_helper(digits, letters):
    
    # if there are no more digits, then we can use what's in letters
    # to look up if we have a valid word from those letter and return
    # the word if we do
    if not digits:
        word = ''.join(letters)
        if word in validWords:
            return [word]
        return []
    
    # create results list
    results = []
    
    # get all characters in map from the first digit
    chars = lettersMaps[digits[0]]
    
    # for all the characters for this first digit
    for char in chars:
        # recursively call helper with all digits except the first one
        # and append the current character to running letters list
        results += make_words_helper(digits[1:], letters+[char])
    
    # return results
    return results
    

def make_words(phone):
    digits = []
    
    # turn string into list of integers
    for digit in phone:
        digits.append(int(digit))
        
    # using all integers from inpput, call recursive function
    return make_words_helper(digits, [])



print(make_words('364'))
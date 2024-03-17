# Soundex maps every name to a string consisting of 
# one letter and three numbers, like M460.

# One version of the algorithm is as follows:

# Remove consecutive consonants with the same sound (for example, change ck -> c).
# Keep the first letter. The remaining steps only apply to the rest of the string.
# Remove all vowels, including y, w, and h.
# Replace all consonants with the following digits:
# b, f, p, v → 1
# c, g, j, k, q, s, x, z → 2
# d, t → 3
# l → 4
# m, n → 5
# r → 6
# If you don't have three numbers yet, append zeros until you do. Keep the first three numbers.
letters_to_delete = set(["a","e", "i", "o", "u", "y","w","h"])

same_sound_map = {"c":"k", "k": "c"}

letter_to_number_map = {"b":1,
                        "f":1,
                        "p":1,
                        "v":1,
                        "c":2,
                        "g":2,
                        "j":2,
                        "k":2,
                        "q":2,
                        "s":2,
                        "x":2,
                        "z":2,
                        "d":3,
                        "t":3,
                        "l":4,
                        "m":5,
                        "n":5,
                        "r":6,}


def soundex(s: str) -> str:
    res = s[0]
    i = 0
    while i < len(s):
        if i < len(s):
            if s[i] in same_sound_map and s[i+1] in same_sound_map:
                s = s[0:i+1] + s[i+2:]
                continue
        if s[i] in letters_to_delete:
            s = s[0:i] + s[i+1:]
            continue
        if s[i] in letter_to_number_map:


    return res
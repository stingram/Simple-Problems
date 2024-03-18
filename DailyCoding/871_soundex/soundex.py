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
    s = s.lower()
    while i < len(s):
        print(f"i: {i},s: {s}, r: {res}")
        if i < len(s) -1 and s[i] in letter_to_number_map and s[i+1] in letter_to_number_map and letter_to_number_map[s[i]] == letter_to_number_map[s[i+1]]:
                s = s[0:i+1] + s[i+2:]
                continue
        elif s[i] in letters_to_delete and i != 0:
            s = s[0:i] + s[i+1:]
            continue
        elif s[i] in letter_to_number_map and i != 0:
            res += str(letter_to_number_map[s[i]])
            i += 1
        elif i == 0:
             i += 1
        else:
             print(f"ERROR")
    while len(res) < 4:
        res += "0"

    return res

# Jackson and Jaxen both map to J250
test = "Jackson"
print(f"{soundex(test)}")

test = "Jaxen"
print(f"{soundex(test)}")
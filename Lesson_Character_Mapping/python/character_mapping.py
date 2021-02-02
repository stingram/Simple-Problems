


def counts(s1):
    counts = {}
    for c in s1:
        if c not in counts:
            counts[c] = 1
        else:
            counts[c] += 1
    return counts 
    
def cmp_dict(counts):
    cdict = {}    
    for k,v in counts.items():
        if v not in cdict:
            cdict[v] = 1
        else:
            cdict[v] += 1
    return cdict

    # if for each key, the lengths are the same for s1 and s2, then mapping is possible
def compare_dictionaries(cdict1, cdict2):
    for k,v in cdict1.items():
        if k not in cdict2:
            return False
        if v != cdict2[k]:
            return False
    return True

def character_mapping(s1, s2):


    cd1 = counts(s1)
    cd2 = counts(s2)
    
    # build dictionary mapping ints to list of letters
    # where the int is number of occurences
    cdict1 = cmp_dict(cd1)
    cdict2 = cmp_dict(cd2)

    return compare_dictionaries(cdict1, cdict2)
    
    
def character_mapping_v2(s1, s2):
    
    if len(s1) != len(s2):
        return False
    
    chars = {}
    for i in range(len(s1)):
        if s1[i] not in chars:
            chars[s1[i]] = s2[i]
        else:
            if chars[s1[i]] != s2[i]:
                return False
    return True

    
s1 = "aaabb"
s2 = "cdcdd"

print(character_mapping_v2(s1,s2))
    
    
    
    
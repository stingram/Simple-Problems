from typing import List
from collections import Counter



def get_available_characters(str1: List[str], str2: List[str]) -> List[str]:
    res = list((Counter(str1) - Counter(str2)).elements())
    return res


def helper(in_str1: List[str], in_str2: List[str], N: int) ->List[int]:
    if N == 0:
        return [""]
    in_str2 = helper(in_str1, in_str2, N-1)
    rfull = []
    for element in in_str2:
        lin_str = []
        lin_str[:0] = element
        branches = get_available_characters(in_str1, lin_str)
        print(f"N: {N}, branches: {branches}.")
        res = []
        for b in branches:
            res.append(element+b)
        rfull += res
    print(f"N: {N}, Res: {rfull}.")
    return rfull


def string_perm(in_str: str)->List[str]:
    N = len(in_str)
    lin_str = []
    lin_str[:0] = in_str
    print(f"In string: {lin_str}.")
    res = helper(lin_str, "", N)  
    return res

str1 = ['a', 'b', 'c']
str2 = ['']

print(f"Difference: {get_available_characters(str1,str2)}.")

in_str = "abc"
print(f"Perms for {in_str}: {string_perm(in_str)}.")
# build list of perumutations for 0s and 1s
from typing import List

def helper(N: int) -> List[str]:
    if N == 0:
        return [""]
    new_arr = helper(N-1)
    r, l = [],[]
    for a in new_arr:
        r.append("1"+a)
        l.append("0"+a)
    return l + r


def res(N: int)->List[str]:
    return helper(N)


N = 4

print(f"Permutations: {res(N)}.")

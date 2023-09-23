from typing import List

def find_combs(s: str) ->List[str]:
    if len(s) == 0:
        return [""]
    else:
        combs_without_first = find_combs(s[1:])
        combs_with_first = [s[0]+comb for comb in combs_without_first]
        return combs_with_first + combs_without_first
    
s = 'abc'
print(f"perms: {find_combs(s)}")
from typing import List

def string_perm(in_string: str) -> List[str]:
    if len(in_string) == 0:
        return [""]
    else:
        combs_without_first = string_perm(in_string[1:])
        combs_with_first = [in_string[0] + comb for comb in combs_without_first]
        return combs_with_first + combs_without_first


s = "abcde"
print(f"perms of {s}: {string_perm(s)}.")
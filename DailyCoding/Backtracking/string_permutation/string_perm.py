# given a strign ike abc, ginf al permutation like 
# # ["abc", "acb", "bac", "bca", "cba", "cab"]
# assuming each letter is unique


from typing import List


def is_valid_state(state: str, s:str) -> bool:
    return len(state) == len(s)


def get_candidates(state:str, s: str) -> List[str]:
    candidates = list(s)
    for c in s:
        if c in state:
            candidates.remove(c)
    return candidates

def search(state:str, solutions:List[str], s:str):
    
    if is_valid_state(state, s):
        solutions.add(state)
    
    for candidate in get_candidates(state,s):
        state += candidate
        search(state,solutions,s)
        state = state[:-1]

def find_perms(s: str) -> List[str]:
    state = ""
    solutions = set()
    search(state,solutions,s)
    return solutions


s = "abcd"
perms = find_perms(s)
print(f"perms for {s}:\n{perms}\n Count: {len(perms)}")
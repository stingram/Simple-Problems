from typing import List

def _is_palindrome(s: str):
    for i in range(len(s)//2):
        if s[i] != s[-1-i]:
            return False
    return True

def _helper(s:str, curr_str: List[str], res: List[str]):
    if(s == ""):
        res.append(curr_str.copy())
        return
    
    if(len(s) == 1):
        curr_str.append(s)
        _helper("",curr_str,res)
        curr_str.pop()
        return
    else:
        for split_index in range(1,len(s)+1):
            if _is_palindrome(s[:split_index]):
                curr_str.append(s[:split_index])
                _helper(s[split_index:],curr_str,res)
                curr_str.pop()
        return

def partition(s: str) -> List[List[str]]:
    res = []
    _helper(s,[],res)
    return res

arr = "aab"

print(f"partitions: {partition(arr)}")
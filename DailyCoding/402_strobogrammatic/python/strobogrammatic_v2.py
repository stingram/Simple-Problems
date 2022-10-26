from typing import List



def helper(N: int, is_odd: bool) -> List[str]:
    if N == 0:
        return [""]
    if N == 1:
        return ["1", "8", "0"]
    arr = helper(N-1)
    ones, eights, zeros, s, n = [],[],[],[],[]
    for a in arr:
        if N > 2:
            ones.append("1"+a)
            eights.append("8"+a)
            zeros.append("0"+a)
            s.append("6"+a)
            n.append("9"+a)
        else:
            ones.append("1"+a+"1")
            eights.append("8"+a+"8")
            zeros.append("0"+a+"0")
            s.append("6"+a+"9")
            n.append("9"+a+"6")
    return ones + eights +zeros+s+n
    
def find_strobogrammatic_numbers(N: int) -> List[str]:
    res = helper(N, N % 2)
    return res

N = 2
print(f"Res: {find_strobogrammatic_numbers(N)}.")
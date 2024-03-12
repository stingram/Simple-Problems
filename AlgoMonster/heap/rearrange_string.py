# Reorganize String
# Given a string s, check if the letters can be rearranged 
# such that so consecutive letters appear


from collections import Counter
from heapq import heappop, heappush

def rearrange(s: str) -> str:
    c = Counter()
    for char in s:
        c[char] += 1

    # build heap from counter
    heap = []
    for char,count in c.items():
        heappush(heap,(-count,char))

    # check biggest
    biggest = -heap[0][0]
    N = len(s)
    if (N % 2 == 0 and biggest > len(s)//2) or (biggest > len(s) // 2 +1):
        return ""

    new_s = [""]*N

    # set index
    index = 0
    while heap:
        count, char = heappop(heap)
        for _ in range(-count):
            s[index] = char
            index +=2
            if index > N - 1:
                index = 1

    return "".join([c for c in new_s])
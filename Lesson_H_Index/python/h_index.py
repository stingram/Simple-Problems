
# SPACE
#  O(n)

# TIME
# O(n) 

def h_index(papers):
    n = len(papers)

    # build histogram
    freq = [0]*(n+1)
    for p in papers:
        if p >= n:
            freq[n] += 1 # H_index can't exceed number of publications
        else:
            freq[p] += 1

    print(freq)

    total = 0
    i= n
    while i>= 0:
        total += freq[i]
        if total >= i:
            return i
        i -= 1

    return 0 

print(h_index([5, 3, 3, 1, 0]))
# 3

print(h_index([5, 3, 3, 1, 4, 4, 4]))
# 4

print(h_index([0, 3, 3, 1, 4, 4, 4]))
# 3
# an ugly is one whose only prime factors are [2,3,5]
# return the Nth ugly number
from heap import heappop, heappush

def ugly_number(n: int):
    if n == 1:
        return n
    
    factors =[2,3,5]
    # start with 1 numbers
    seen = set([1])
    heap = [1]

    count = 1
    while count < n:
        val = heappop(heap)
        for factor in factors:
            new_val = factor*val
            if new_val not in seen:
                seen.add(new_val)
                heappush(heap, new_val)
                count += 1 

    return heap[0]
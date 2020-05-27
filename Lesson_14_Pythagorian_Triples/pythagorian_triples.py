from typing import List

def findPythagoreanTriplets(in_list :List[int]):
    c = set([i*i for i in in_list])
 
    for a in in_list:
        for b in in_list:
            if a*a + b*b in c:
                return True
    return False


print(findPythagoreanTriplets([3, 5, 12, 5, 13]))
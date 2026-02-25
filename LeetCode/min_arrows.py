from typing import List
import random

def min_arrows(points: List[List[int]]) -> int:
    """
    Given balloon intervals [start, end],
    return the minimum number of arrows required to burst all balloons.
    """
    res = 0
    if not points:
        return res
    n = len(points)
    # sort by end times
    points.sort(key= lambda p: p[1])
    # This is the first interval that ends so we at least must pop here
    pop = points[0][1]
    res += 1
    # look at rest of intervals
    for i in range(1,n):
        # get start of this current point
        start = points[i][0]
        # only if there's no overlap do we new arrow and a new pop point
        if start > pop:
            res += 1
            # since we need a new arrow, we should pop as far down as we can
            pop = points[i][1]
    return res

def _ref_min_arrows(points):
    if not points:
        return 0
    points = sorted(points, key=lambda x: x[1])
    arrows = 1
    curr_end = points[0][1]
    for s, e in points[1:]:
        if s > curr_end:
            arrows += 1
            curr_end = e
    return arrows

def run_tests():
    # deterministic tests
    cases = [
        ([[10,16],[2,8],[1,6],[7,12]], 2),
        ([[1,2],[3,4],[5,6],[7,8]], 4),
        ([[1,10],[2,3],[4,5],[6,7],[8,9]], 4),
        ([[1,2]], 1),
        ([], 0),
        ([[1,5],[2,6],[3,7],[4,8]], 1),
        ([[1,2],[2,3],[3,4]], 2),
        ([[-10,-1],[-5,0],[1,5]], 2),
    ]

    for points, want in cases:
        got = min_arrows(points)
        assert got == want, f"Failed: {points} → {got}, want {want}"

    # randomized tests
    for _ in range(200):
        n = random.randint(0, 50)
        points = []
        for _ in range(n):
            a = random.randint(-20, 20)
            b = random.randint(a, a + random.randint(0, 10))
            points.append([a, b])

        got = min_arrows(points)
        want = _ref_min_arrows(points)
        assert got == want, f"Random failed: {points} → {got}, want {want}"

    print("✅ All tests passed!")

if __name__ == "__main__":
    run_tests()
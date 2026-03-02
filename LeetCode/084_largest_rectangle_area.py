from typing import List
import random


def largest_rectangle_area(heights: List[int]) -> int:
    """
    Return the area of the largest rectangle in the histogram.
    """
    stack = []
    res = 0
    n = len(heights)
    
    for i, height in enumerate(heights):
        start = i
        # we check if previous heights are bigger than current,
        # if so we need to pop them, compute area and see how
        # far back left this height at i can go
        while stack and stack[-1][0] > height:
            
            h, j = stack.pop()
            w = i - j
            a = h * w
            res = max(res, a)
            start = j
        # we've cleared stack so it' monotonic so now can push
        stack.append((height, start))
    
    # now that we've checked "left" side of all heights, for 
    # whatever is left we need to check "right"
    while stack:
        h, j = stack.pop()
        w = n - j
        a = h * w
        res = max(res,a)
    return res

def _ref_largest_rectangle_area(heights):
    # O(n^2) brute force for small n
    best = 0
    n = len(heights)
    for i in range(n):
        mn = heights[i]
        for j in range(i, n):
            mn = min(mn, heights[j])
            best = max(best, mn * (j - i + 1))
    return best


def run_tests():
    # deterministic tests
    cases = [
        ([2,1,5,6,2,3], 10),
        ([2,4], 4),
        ([1,1,1], 3),
        ([0], 0),
        ([3,0,3], 3),
        ([6,5,4,3,2,1], 12),  # 4*3 or 3*4 etc.
        ([1,2,3,4,5], 9),     # 3*3
    ]

    for heights, want in cases:
        got = largest_rectangle_area(heights)
        assert got == want, f"Failed: {heights} → {got}, want {want}"

    # randomized tests (small for brute force)
    for _ in range(300):
        n = random.randint(1, 40)
        heights = [random.randint(0, 10) for _ in range(n)]
        got = largest_rectangle_area(heights)
        want = _ref_largest_rectangle_area(heights)
        assert got == want, f"Random failed: {heights} → {got}, want {want}"

    print("✅ All tests passed!")


if __name__ == "__main__":
    run_tests()
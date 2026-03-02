from typing import List
import random


def _is_feasible(nums, target, k):
    curr_sum = 0
    splits = 1
    for num in nums:
        if num > target:
            return False
        curr_sum += num
        if curr_sum > target:
            splits += 1
            curr_sum = num
            if splits > k:
                return False
    return True
            

def split_array_largest_sum(nums: List[int], k: int) -> int:
    """
    Return the minimum possible largest subarray sum
    when splitting nums into at most k contiguous parts.
    """
    # binary search
    L = max(nums)
    R = sum(nums)
    while L < R:
        M = (R + L) // 2
        if _is_feasible(nums, M, k):
            R = M
        else:
            L = M + 1
    return L

def _can_split(nums, k, max_allowed):
    """
    Greedy check:
    can we split nums into <= k subarrays
    where each subarray sum <= max_allowed?
    """
    curr = 0
    groups = 1
    for x in nums:
        if curr + x <= max_allowed:
            curr += x
        else:
            groups += 1
            curr = x
            if groups > k:
                return False
    return True

def _ref_split_array(nums, k):
    # brute force for small n
    from itertools import combinations
    n = len(nums)
    best = float('inf')

    for cuts in combinations(range(1, n), k - 1):
        parts = []
        prev = 0
        for c in cuts:
            parts.append(sum(nums[prev:c]))
            prev = c
        parts.append(sum(nums[prev:]))
        best = min(best, max(parts))
    return best

def run_tests():
    # deterministic tests
    cases = [
        ([7,2,5,10,8], 2, 18),
        ([1,2,3,4,5], 2, 9),
        ([1,4,4], 3, 4),
        ([1,1,1,1], 4, 1),
        ([10], 1, 10),
        ([0,0,0,0], 2, 0),
        ([5,5,5,5], 1, 20),
        ([5,5,5,5], 2, 10),
    ]

    for nums, k, want in cases:
        got = split_array_largest_sum(nums, k)
        assert got == want, f"Failed: {nums}, k={k} → {got}, want {want}"

    # randomized tests (small for brute force)
    for _ in range(200):
        n = random.randint(1, 8)
        nums = [random.randint(0, 9) for _ in range(n)]
        k = random.randint(1, n)
        got = split_array_largest_sum(nums, k)
        want = _ref_split_array(nums, k)
        assert got == want, f"Random failed: {nums}, k={k} → {got}, want {want}"

    print("✅ All tests passed!")

if __name__ == "__main__":
    run_tests()
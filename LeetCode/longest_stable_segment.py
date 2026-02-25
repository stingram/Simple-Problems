from typing import List, Tuple
import random
from collections import deque, defaultdict


def longest_stable_segment(nums: List[int], limit: int) -> int:
    """
    Return the length of the longest contiguous subarray where
    max(subarray) - min(subarray) <= limit.
    """
    # need two queues, the front of each is valid for our given window
    
    min_q = deque() # monotonically increasing
    max_q = deque() # monotonically decreasing

    res = 0
    # use sliding window method
    l = 0
    for r in range(len(nums)):
        # first we check that before we add nums[r]
        # to our deques that doing so won't violate 
        # invariant
        while len(min_q) > 0 and nums[r] < min_q[-1]:
            min_q.pop()
        while len(max_q) >0 and nums[r] > max_q[-1]:
            max_q.pop()
            
        # now we can append
        min_q.append(nums[r])
        max_q.append(nums[r])
        
        # now we check if window is valid by checking limit
        while max_q[0] - min_q[0] > limit:
            # we check if we need to remove nums[l] from front of our deques
            if nums[l] == max_q[0]:
                max_q.popleft()
            if nums[l] == min_q[0]:
                min_q.popleft()
            l += 1
        res = max(res, r-l+1)
        
    return res
    

# # problem 560
# def count_num_subarray_sum_k(nums: List[int], k: int) -> int:
#     """
#     Return the length of the longest contiguous subarray with sum exactly k.
#     Works with negative numbers.
#     """
#     res = 0
#     prefix_sum_counter = {}
    
#     # we can take nothing achieve 0, 1 time
#     prefix_sum_counter[0] = 1
    
#     # build prefix sum
#     curr_sum = 0
    
#     # go through nums
    
#     # check if prefix_sum_at_this_position - k is in prefix_sum_counter
#     # if it is, we add that prefix_sum_counter[prefix_sum_at_this_position - k] to result
#     # then we add 1 to prefix_sum_counter[prefix_sum_at_this_position] 
#     for num in nums:
#         curr_sum += num
#         res += prefix_sum_counter.get(curr_sum - k,0)
#         prefix_sum_counter[curr_sum] = 1 + prefix_sum_counter.get(curr_sum,0) 
#     return res

def longest_subarray_sum_k(nums: List[int], k: int) -> int:
    """
    Return the length of the longest contiguous subarray with sum exactly k.
    Works with negative numbers.
    """
    res = 0
    prefix_sum_first_seen_counter = {}
    
    # we can take nothing achieve 0, 1 time
    prefix_sum_first_seen_counter[0] = -1
    
    # build prefix sum
    curr_sum = 0
    
    # go through nums
    
    # check if curr_sum - k is in prefix_sum_first_seen_counter
    # if it is, we get our i value (position index) from
    # prefix_sum_first_seen_counter[curr_sum - k] and use current index to
    # get length = j - i, we update result if length is bigger then res
    # then we set prefix_sum_first_seen_counter[curr_sum] = current_index 
    for j,num in enumerate(nums):
        curr_sum += num
        i = prefix_sum_first_seen_counter.get(curr_sum - k)
        if i is not None and j-i > res:
            res = j - i
        if curr_sum not in prefix_sum_first_seen_counter:
            prefix_sum_first_seen_counter[curr_sum] = j
    return res
    

# ---------------------------
# Reference solutions (slow / for testing only)
# ---------------------------

def _ref_longest_stable_segment(nums: List[int], limit: int) -> int:
    best = 0
    for i in range(len(nums)):
        mn = mx = nums[i]
        for j in range(i, len(nums)):
            mn = min(mn, nums[j])
            mx = max(mx, nums[j])
            if mx - mn <= limit:
                best = max(best, j - i + 1)
            else:
                break
    return best

def _ref_min_semesters(n: int, prereqs: List[List[int]]) -> int:
    # BFS level-by-level; O(n+m). This is actually fast, but used as "reference".
    g = [[] for _ in range(n)]
    indeg = [0] * n
    for a, b in prereqs:
        g[b].append(a)
        indeg[a] += 1

    q = deque([i for i in range(n) if indeg[i] == 0])
    taken = 0
    sem = 0
    while q:
        sem += 1
        for _ in range(len(q)):
            u = q.popleft()
            taken += 1
            for v in g[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
    return sem if taken == n else -1

def _ref_longest_subarray_sum_k(nums: List[int], k: int) -> int:
    best = 0
    for i in range(len(nums)):
        s = 0
        for j in range(i, len(nums)):
            s += nums[j]
            if s == k:
                best = max(best, j - i + 1)
    return best

# ---------------------------
# Deterministic tests
# ---------------------------

def _run_deterministic_tests():
    # Problem 1
    cases1: List[Tuple[List[int], int, int]] = [
        ([8,2,4,7], 4, 2),
        ([10,1,2,4,7,2], 5, 4),
        ([4,2,2,2,4,4,2,2], 0, 3),
        ([1], 0, 1),
        ([1,2,3,4,5], 10, 5),
        ([5,4,3,2,1], 1, 2),
        ([-1,-1,-1], 0, 3),
        ([-10,0,10], 5, 1),
    ]
    for nums, limit, want in cases1:
        got = longest_stable_segment(nums, limit)
        assert got == want, f"P1 failed: nums={nums}, limit={limit}, got={got}, want={want}"

    # # Problem 2
    # cases2: List[Tuple[int, List[List[int]], int]] = [
    #     (4, [[1,0],[2,0],[3,1],[3,2]], 3),
    #     (2, [[0,1],[1,0]], -1),
    #     (1, [], 1),
    #     (3, [[1,0],[2,1]], 3),
    #     (3, [[2,0],[2,1]], 2),
    #     (5, [], 1),
    # ]
    # for n, prereqs, want in cases2:
    #     got = min_semesters(n, prereqs)
    #     assert got == want, f"P2 failed: n={n}, prereqs={prereqs}, got={got}, want={want}"

    # Problem 3
    cases3: List[Tuple[List[int], int, int]] = [
        ([1,-1,5,-2,3], 3, 4),
        ([-2,-1,2,1], 1, 2),
        ([1,2,3], 3, 2),
        ([3], 3, 1),
        ([0,0,0], 0, 3),
        ([1,-1,1,-1], 0, 4),
        ([5,-2,-1,2,-2,3], 3, 3),  # e.g., [-2,-1,2,-2,6?] depends; ref checks
    ]
    for nums, k, want in cases3:
        got = longest_subarray_sum_k(nums, k)
        assert got == want, f"P3 failed: nums={nums}, k={k}, got={got}, want={want}"

# ---------------------------
# Randomized tests (compare to brute force references)
# ---------------------------

def _run_random_tests(seed: int = 0):
    random.seed(seed)

    # P1 random: keep small for brute force
    for _ in range(200):
        n = random.randint(1, 40)
        nums = [random.randint(-10, 10) for _ in range(n)]
        limit = random.randint(0, 15)
        got = longest_stable_segment(nums, limit)
        want = _ref_longest_stable_segment(nums, limit)
        assert got == want, f"P1 random failed: nums={nums}, limit={limit}, got={got}, want={want}"

    # # P2 random DAG/cycle mix
    # for _ in range(200):
    #     n = random.randint(1, 40)
    #     m = random.randint(0, 80)
    #     prereqs = []
    #     for _ in range(m):
    #         a = random.randint(0, n-1)
    #         b = random.randint(0, n-1)
    #         if a != b:
    #             prereqs.append([a,b])
    #     got = min_semesters(n, prereqs)
    #     want = _ref_min_semesters(n, prereqs)
    #     assert got == want, f"P2 random failed: n={n}, prereqs={prereqs}, got={got}, want={want}"

    # P3 random: small for brute force
    for _ in range(200):
        n = random.randint(1, 50)
        nums = [random.randint(-5, 5) for _ in range(n)]
        k = random.randint(-10, 10)
        got = longest_subarray_sum_k(nums, k)
        want = _ref_longest_subarray_sum_k(nums, k)
        assert got == want, f"P3 random failed: nums={nums}, k={k}, got={got}, want={want}"

def run_all_tests():
    _run_deterministic_tests()
    _run_random_tests(seed=42)
    print("✅ All tests passed!")

if __name__ == "__main__":
    run_all_tests()
import unittest
from collections import defaultdict, Counter


def _is_smaller(w1, w2, s):
    # if w1[1] - w1[0] == w2[1] - w2[0]:
    #     for i in range(w1[1] - w1[0]):
    #         if s[w1[0] + i] != s[w2[0] + i]:
    #             return s[w1[0] + i] < s[w2[0] + i]
    #     return False
    # else:
    return w1[1] - w1[0] < w2[1] - w2[0]

def min_window(s: str, t: str) -> str:
    """
    Return the minimum-length substring of s that contains all chars in t
    with correct multiplicity, or "" if impossible.

    Target complexity: O(len(s) + len(t))

    Implement using sliding window + counting.
    """

    m, n = len(s), len(t)
    if m < n or m == 0 or n == 0:
        return ""
    check_counter = Counter(t)
    required = len(check_counter.keys())
    satisfied = 0
    l = 0
    window = (-m-1,0)
    window_counter = defaultdict(int)
    for r in range(m):
        if s[r] in check_counter:
            window_counter[s[r]] += 1
            if window_counter[s[r]] == check_counter[s[r]]:
                satisfied += 1
        while satisfied == required:
            if _is_smaller((l,r+1),window,s):
                window = (l,r+1)
            if s[l] in check_counter:
                window_counter[s[l]] -= 1
                if window_counter[s[l]] < check_counter[s[l]]:
                    satisfied -= 1
            l += 1
    
    return s[window[0]:window[1]]


class TestMinWindow(unittest.TestCase):
    def test_examples(self):
        self.assertEqual(min_window("ADOBECODEBANC", "ABC"), "BANC")
        self.assertEqual(min_window("a", "a"), "a")
        self.assertEqual(min_window("a", "aa"), "")

    def test_exact_match(self):
        self.assertEqual(min_window("abc", "abc"), "abc")

    def test_multiple_counts(self):
        self.assertEqual(min_window("aaabdabcefaecbef", "abc"), "abc")
        self.assertEqual(min_window("bba", "ab"), "ba")
        self.assertEqual(min_window("aaflslflsldkalskaaa", "aaa"), "aaa")

    def test_not_found(self):
        self.assertEqual(min_window("hello", "world"), "")

    def test_empty_inputs(self):
        self.assertEqual(min_window("", "a"), "")
        self.assertEqual(min_window("a", ""), "")
        self.assertEqual(min_window("", ""), "")

    def test_repeated_characters(self):
        self.assertEqual(min_window("caaab", "aab"), "aab")
        self.assertEqual(min_window("acaabb", "aab"), "aab")


if __name__ == "__main__":
    unittest.main()

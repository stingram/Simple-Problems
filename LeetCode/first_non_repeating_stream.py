from typing import List

from collections import deque
import random
import string


def first_non_repeating_stream(stream: List[str]) -> List[str]:
    """
    After each character in the stream, return the first
    non-repeating character seen so far, or '#' if none exists.
    """
    seen = {}
    q = deque([])
    res = []
    for s in stream:
        if s not in seen:
            seen[s] = 1
            q.append(s)
        else:
            seen[s] += 1
        while q:
            if seen[q[0]] > 1:
                q.popleft()
            else:
                break
        res.append(q[0] if q else '#')
    return res



def _ref_first_non_repeating(stream):
    res = []
    from collections import Counter
    cnt = Counter()
    for i, ch in enumerate(stream):
        cnt[ch] += 1
        found = '#'
        for j in range(i + 1):
            if cnt[stream[j]] == 1:
                found = stream[j]
                break
        res.append(found)
    return res

def run_tests():
    # deterministic tests
    cases = [
        (['a','b','a','c','c','b','d'], ['a','a','b','b','b','#','d']),
        (['a','a','a'], ['a','#','#']),
        (['a','b','c'], ['a','a','a']),
        ([], []),
        (['z'], ['z']),
    ]

    for stream, want in cases:
        got = first_non_repeating_stream(stream)
        assert got == want, f"Failed: {stream} → {got}, want {want}"

    # randomized tests
    for _ in range(200):
        n = random.randint(0, 50)
        stream = [random.choice(string.ascii_lowercase[:5]) for _ in range(n)]
        got = first_non_repeating_stream(stream)
        want = _ref_first_non_repeating(stream)
        assert got == want, f"Random failed: {stream}"

    print("✅ All tests passed!")

if __name__ == "__main__":
    run_tests()
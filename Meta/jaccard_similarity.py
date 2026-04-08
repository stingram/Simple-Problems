"""
Meta-style AI Coding Practice #3: Jaccard Similarity for Sparse Vectors

Problem
-------
You are given two sparse binary vectors represented as sorted lists of indices.

Example:
vector A = [1, 4, 7]
vector B = [2, 4, 7, 10]

This means:
dimension = implicit large space
value = 1 at listed indices
value = 0 elsewhere

Implement:

    def jaccard_similarity(a, b):

Definition
----------
Jaccard similarity:

    J(A,B) = |A ∩ B| / |A ∪ B|

Requirements
------------
- Input lists are sorted and contain unique indices.
- Must run in O(len(a) + len(b)) time.
- Must use O(1) extra space.
- If both vectors are empty → return 1.0
- If only one empty → return 0.0

What this tests
---------------
- sparse reasoning
- two-pointer technique
- edge cases
- algorithmic efficiency

Implement only jaccard_similarity.
"""


def jaccard_similarity(a, b):
    len_a = len(a)
    len_b = len(b)
    if len_a == 0 and len_b == 0:
        return 1.0
    if len_a == 0 or len_b == 0:
        return 0.0
    
    apointer, bpointer = 0, 0
    and_sum = 0.0
    or_sum = 0.0
    while apointer < len_a and bpointer < len_b:
        if a[apointer] == b[bpointer]:
            and_sum += 1
            or_sum += 1
            apointer += 1
            bpointer += 1
        elif a[apointer] < b[bpointer]:
            or_sum += 1
            apointer += 1
        else:
            or_sum += 1
            bpointer += 1
    # if a isn't done
    while apointer < len_a:
        or_sum += 1
        apointer += 1
    # if b isn't done
    while bpointer < len_b:
        or_sum += 1
        bpointer += 1
    return and_sum / or_sum


# =========================
# Tests
# =========================

def test_basic():
    a = [1,4,7]
    b = [2,4,7,10]
    assert abs(jaccard_similarity(a,b) - (2/5)) < 1e-8


def test_identical():
    a = [1,3,5]
    b = [1,3,5]
    assert jaccard_similarity(a,b) == 1.0


def test_disjoint():
    a = [1,2]
    b = [3,4]
    assert jaccard_similarity(a,b) == 0.0


def test_one_empty():
    assert jaccard_similarity([], [1,2]) == 0.0
    assert jaccard_similarity([1,2], []) == 0.0


def test_both_empty():
    assert jaccard_similarity([], []) == 1.0


def test_subset():
    a = [1,2]
    b = [1,2,3,4]
    assert abs(jaccard_similarity(a,b) - (2/4)) < 1e-8


def test_large_gap():
    a = [1,1000,2000]
    b = [1000,3000]
    assert abs(jaccard_similarity(a,b) - (1/4)) < 1e-8


def run_all_tests():
    test_basic()
    test_identical()
    test_disjoint()
    test_one_empty()
    test_both_empty()
    test_subset()
    test_large_gap()
    print("All tests passed!")


if __name__ == "__main__":
    run_all_tests()
"""
Meta-style AI Coding Practice #2: Average Precision for Binary Classification

Problem
-------
Implement average precision (AP) for a binary classifier.

You are given:
- y_true: list of 0/1 labels of length N
- y_score: list of prediction scores of length N, where larger score means more likely positive

Write:

    def average_precision(y_true, y_score):
        ...

Definition
----------
1. Sort examples by descending score.
2. Sweep through the sorted list from top to bottom.
3. Every time you encounter a true positive, compute precision at that rank:
       precision@i = (# true positives seen so far) / i
   where i is 1-indexed rank.
4. Average those precision values over the total number of true positives.

Formally:

    AP = (1 / P) * sum_{i : item at rank i is positive} precision@i

where P is total number of positives in y_true.

Requirements
------------
- If there are no positive labels, return 0.0.
- Break score ties stably using original order.
- Do not use sklearn.
- Code should be clean and easy to explain.

Examples
--------
y_true  = [1, 0, 1, 0]
y_score = [0.9, 0.8, 0.7, 0.1]

Sorted by score:
rank 1: label 1 -> precision = 1/1
rank 2: label 0
rank 3: label 1 -> precision = 2/3

AP = (1 + 2/3) / 2 = 0.833333...

What this tests
---------------
- metric implementation
- sorting and stable tie handling
- careful indexing
- debugging / verification

Complexity target
-----------------
- Time: O(N log N)
- Space: O(N)

Implement only average_precision().
Do not modify the tests.
"""


def average_precision(y_true, y_score):
    zipped = zip(y_score, y_true)
    sorted_pairs = sorted(zipped,key=lambda x: x[0],reverse=True)
    _, y_true_sorted_tup = zip(*sorted_pairs) 
    y_true_sorted = list(y_true_sorted_tup)
    sum_precision = 0.0
    num_pos = 0
    for i,true in enumerate(y_true_sorted):
        if true:
            num_pos += 1
            sum_precision += (num_pos/(i + 1))
    if num_pos == 0:
        return 0.0
    return sum_precision/num_pos


# =========================
# Tests
# =========================

def test_basic_case():
    y_true = [1, 0, 1, 0]
    y_score = [0.9, 0.8, 0.7, 0.1]
    ap = average_precision(y_true, y_score)
    expected = (1.0 + (2.0 / 3.0)) / 2.0
    assert abs(ap - expected) < 1e-8, f"got {ap}, expected {expected}"


def test_all_positive():
    y_true = [1, 1, 1]
    y_score = [0.2, 0.9, 0.5]
    ap = average_precision(y_true, y_score)
    # After sorting labels are all positive, so precision at every positive rank is 1.
    expected = 1.0
    assert abs(ap - expected) < 1e-8, f"got {ap}, expected {expected}"


def test_no_positive():
    y_true = [0, 0, 0]
    y_score = [0.1, 0.2, 0.3]
    ap = average_precision(y_true, y_score)
    assert ap == 0.0, f"got {ap}, expected 0.0"


def test_worst_ranking():
    y_true = [1, 1, 0, 0]
    y_score = [0.1, 0.2, 0.9, 0.8]
    ap = average_precision(y_true, y_score)
    # Sorted labels: [0, 0, 1, 1]
    # precision at positive ranks = 1/3 and 2/4
    expected = ((1.0 / 3.0) + (2.0 / 4.0)) / 2.0
    assert abs(ap - expected) < 1e-8, f"got {ap}, expected {expected}"


def test_stable_tie_breaking():
    y_true = [1, 0, 1]
    y_score = [0.5, 0.5, 0.5]
    ap = average_precision(y_true, y_score)
    # Stable sort preserves original order: labels remain [1,0,1]
    # precision at positive ranks = 1/1 and 2/3
    expected = (1.0 + (2.0 / 3.0)) / 2.0
    assert abs(ap - expected) < 1e-8, f"got {ap}, expected {expected}"


def test_single_positive():
    y_true = [0, 0, 1, 0]
    y_score = [0.9, 0.8, 0.7, 0.1]
    ap = average_precision(y_true, y_score)
    # Sorted labels: [0,0,1,0], only positive at rank 3 => precision = 1/3
    expected = 1.0 / 3.0
    assert abs(ap - expected) < 1e-8, f"got {ap}, expected {expected}"


def test_input_validation():
    try:
        average_precision([1, 0], [0.5])
        raise AssertionError("expected exception for mismatched lengths")
    except Exception:
        pass

    try:
        average_precision([], [])
        raise AssertionError("expected exception for empty input")
    except Exception:
        pass


def run_all_tests():
    test_basic_case()
    test_all_positive()
    test_no_positive()
    test_worst_ranking()
    test_stable_tie_breaking()
    test_single_positive()
    test_input_validation()
    print("All tests passed!")


if __name__ == "__main__":
    run_all_tests()
"""
Meta-style AI Coding Practice #6: Beam Search

Problem
-------
You are given:
- probs: a list of length T, where each element is a list of probabilities over a vocabulary

Example:
probs = [
    [0.6, 0.4],   # step 0
    [0.5, 0.5],   # step 1
]

Vocabulary indices: 0, 1

At each timestep you choose one token.

Goal:
Implement beam search to find the top-k most likely sequences.

Write:

    def beam_search(probs, beam_width):

Return:
- list of (sequence, score) sorted by descending score

Where:
- sequence = list of token indices
- score = product of probabilities along the sequence

Requirements
------------
- beam_width >= 1
- At each step:
    - expand all current beams
    - keep only top-k sequences
- Final result should contain k sequences
- Use plain Python or NumPy

What this tests
---------------
- combinatorial search
- pruning
- careful bookkeeping

Implement only beam_search().
"""

import heapq



def beam_search(probs, beam_width):
    assert beam_width >= 1
    beams = [([], 1.0)]
    for step_prob in probs:
        candidates = []
        for seq, score in beams:
            for token,prob in enumerate(step_prob):
                candidates.append((seq+[token],score*prob))
        candidates.sort(key = lambda x: x[1], reverse=True)
        beams = candidates[:beam_width]
    return beams


# =========================
# Tests
# =========================

def test_basic():
    probs = [
        [0.6, 0.4],
        [0.5, 0.5],
    ]

    result = beam_search(probs, beam_width=2)

    sequences = [seq for seq, _ in result]
    assert [0,0] in sequences
    assert [0,1] in sequences


def test_single_step():
    probs = [[0.1, 0.9, 0.0]]
    result = beam_search(probs, beam_width=2)

    assert result[0][0] == [1]
    assert result[1][0] == [0]


def test_beam_pruning():
    probs = [
        [0.9, 0.1],
        [0.1, 0.9],
        [0.9, 0.1],
    ]

    result = beam_search(probs, beam_width=2)

    # ensure only 2 sequences returned
    assert len(result) == 2


def test_sorted_output():
    probs = [
        [0.5, 0.5],
        [0.5, 0.5],
    ]

    result = beam_search(probs, beam_width=2)

    scores = [score for _, score in result]
    assert scores[0] >= scores[1]


def run_all_tests():
    test_basic()
    test_single_step()
    test_beam_pruning()
    test_sorted_output()
    print("All tests passed!")


if __name__ == "__main__":
    run_all_tests()
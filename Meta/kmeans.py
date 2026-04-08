"""
Meta-style AI Coding Practice #1: K-Means Clustering from Scratch

Problem
-------
Implement one iteration-based k-means clustering algorithm.

You are given:
- points: a 2D list or tensor-like structure of shape (N, D), where N is number of points
  and D is feature dimension
- k: number of clusters
- max_iters: maximum number of k-means iterations
- tol: tolerance for convergence based on centroid movement

Write:

    def kmeans(points, k, max_iters=100, tol=1e-4):
        ...

The function should:
1. Initialize centroids as the first k points.
2. Repeatedly:
   - assign each point to its nearest centroid using squared Euclidean distance
   - recompute each centroid as the mean of all points assigned to that cluster
3. Stop early if the maximum centroid movement is <= tol.
4. Return:
   - centroids: list of k centroids, shape (k, D)
   - assignments: list of length N, where assignments[i] is the cluster index of point i

Rules / Requirements
--------------------
- Do not use sklearn.
- You may use Python standard library and/or numpy if you want, but a pure Python solution is fine.
- If a cluster gets no assigned points during an iteration, keep its centroid unchanged.
- Distance metric: squared Euclidean distance.
- Break ties by choosing the smaller cluster index.
- Code should be clean and easy to explain.

What this tests
---------------
- basic ML algorithm implementation
- loops / arrays / indexing
- handling empty clusters
- convergence logic
- debugging / correctness

Complexity target
-----------------
A straightforward implementation is fine:
- Time: O(max_iters * N * k * D)
- Space: O(N + kD)

You only need to implement kmeans().
Do not modify the tests.
"""

from math import isclose
import numpy as np

def kmeans(points, k, max_iters=100, tol=1e-4):
    # arbitrarily assign first k points as centroids
    points = np.array(points)
    centroids = points[:k]
    labels = np.zeros(len(points),dtype=np.int32)
    for i in range(max_iters):
        # calculate distances
        for j,point in enumerate(points):
            min_dist = float('inf')
            for c,centroid in enumerate(centroids):
                d = _squared_dist(point,centroid)
                # label point
                if d < min_dist:
                    labels[j] = c
                    min_dist = d
        # move centroids
        new_centroids = np.zeros_like(centroids)
        # print(f'{labels=}')
        for c in range(k):
            new_centroid =  np.mean(points[np.where(labels==c)],axis=0)
            if ~np.isnan(new_centroid).any():
                new_centroids[c] = new_centroid
            else:
                new_centroids[c] = centroids[c]

        # check if we can exit
        diff = centroids - new_centroids
        diff_sq = diff ** 2
        diff_sq_sum = np.sum(diff_sq,axis=1)
        norm = np.sqrt(diff_sq_sum)
        if np.all(norm <= tol):
            centroids = new_centroids.copy()
            return centroids, labels
        centroids = new_centroids.copy()
    return centroids, labels
        

# =========================
# Helpers for testing
# =========================

def _squared_dist(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))


def _sort_centroids_rows(rows):
    return sorted([tuple(round(x, 6) for x in row) for row in rows])


def _assert_centroids_close(actual, expected, atol=1e-5):
    a_sorted = _sort_centroids_rows(actual)
    e_sorted = _sort_centroids_rows(expected)
    assert len(a_sorted) == len(e_sorted), f"centroid count mismatch: {len(a_sorted)} vs {len(e_sorted)}"
    for a_row, e_row in zip(a_sorted, e_sorted):
        assert len(a_row) == len(e_row), f"centroid dim mismatch: {a_row} vs {e_row}"
        for a, e in zip(a_row, e_row):
            assert abs(a - e) <= atol, f"centroid mismatch: got {a_row}, expected {e_row}"


def _check_assignments_consistent(points, centroids, assignments):
    assert len(points) == len(assignments), "assignments length must equal number of points"
    k = len(centroids)
    for i, p in enumerate(points):
        c = assignments[i]
        assert 0 <= c < k, f"invalid cluster id {c} for point {i}"
        # print(f'{c=},{centroids=}')
        d_assigned = _squared_dist(p, centroids[c])
        for j in range(k):
            d_j = _squared_dist(p, centroids[j])
            # assigned cluster should be no farther than any other cluster
            assert d_assigned <= d_j + 1e-8, (
                f"point {i} assigned to non-nearest centroid: "
                f"assigned dist={d_assigned}, better dist={d_j}, assigned={c}, better={j}"
            )


# =========================
# Tests
# =========================

def test_two_well_separated_clusters():
    points = [
        [0.0, 0.0],
        [0.0, 2.0],
        [2.0, 0.0],
        [8.0, 8.0],
        [8.0, 10.0],
        [10.0, 8.0],
    ]
    centroids, assignments = kmeans(points, k=2, max_iters=20, tol=1e-8)

    expected = [
        [2.0 / 3.0, 2.0 / 3.0],
        [26.0 / 3.0, 26.0 / 3.0],
    ]
    _assert_centroids_close(centroids, expected, atol=1e-5)
    _check_assignments_consistent(points, centroids, assignments)


def test_single_cluster_mean():
    points = [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ]
    centroids, assignments = kmeans(points, k=1, max_iters=10, tol=1e-8)

    expected = [[3.0, 4.0]]
    _assert_centroids_close(centroids, expected, atol=1e-5)
    assert (assignments == [0, 0, 0]).all(), f"unexpected assignments: {assignments}"


def test_empty_cluster_stays_unchanged():
    # First two points are identical, so with k=3 one cluster can become empty.
    points = [
        [0.0, 0.0],
        [0.0, 0.0],
        [10.0, 10.0],
        [10.0, 10.0],
    ]
    centroids, assignments = kmeans(points, k=3, max_iters=10, tol=1e-8)

    assert len(centroids) == 3, f"expected 3 centroids, got {len(centroids)}"
    assert len(assignments) == len(points), "bad assignment length"
    _check_assignments_consistent(points, centroids, assignments)

    # At least one centroid should remain at an existing point if it receives no assignments.
    valid_positions = {(0.0, 0.0), (10.0, 10.0)}
    centroid_positions = {tuple(round(x, 6) for x in c) for c in centroids}
    assert any(c in valid_positions for c in centroid_positions), (
        f"expected at least one centroid to remain unchanged at an original position, got {centroid_positions}"
    )


def test_tie_break_smaller_index():
    # Point [1,0] is equally distant to centroid 0 ([0,0]) and centroid 1 ([2,0]) initially.
    # Tie should go to smaller index.
    points = [
        [0.0, 0.0],   # centroid 0 initial
        [2.0, 0.0],   # centroid 1 initial
        [1.0, 0.0],   # tie
    ]
    centroids, assignments = kmeans(points, k=2, max_iters=1, tol=0.0)

    assert assignments[2] == 0, f"expected tie to break to cluster 0, got {assignments[2]}"


def test_higher_dimensional_case():
    points = [
        [1.0, 1.0, 1.0],
        [1.0, 2.0, 1.0],
        [9.0, 9.0, 9.0],
        [8.0, 9.0, 9.0],
    ]
    centroids, assignments = kmeans(points, k=2, max_iters=20, tol=1e-8)

    expected = [
        [1.0, 1.5, 1.0],
        [8.5, 9.0, 9.0],
    ]
    _assert_centroids_close(centroids, expected, atol=1e-5)
    _check_assignments_consistent(points, centroids, assignments)


def test_invalid_k():
    points = [[1.0, 2.0], [3.0, 4.0]]

    try:
        kmeans(points, k=0)
        raise AssertionError("expected exception for k=0")
    except Exception:
        pass

    try:
        kmeans(points, k=3)
        raise AssertionError("expected exception for k > N")
    except Exception:
        pass


def run_all_tests():
    test_two_well_separated_clusters()
    test_single_cluster_mean()
    test_empty_cluster_stays_unchanged()
    test_tie_break_smaller_index()
    test_higher_dimensional_case()
    test_invalid_k()
    print("All tests passed!")


if __name__ == "__main__":
    run_all_tests()
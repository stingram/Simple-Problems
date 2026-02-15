#!/usr/bin/env python3
"""
hp_ml_leetcode_drills.py

Run:
  python hp_ml_leetcode_drills.py
or:
  python -m unittest hp_ml_leetcode_drills.py -v

This file contains 5 "LeetCode related to ML" problems with #TODO stubs and
small, fast unit tests.

Problems:
  1) Top-K most similar items (NumPy)
  2) Top-p (nucleus) filtering (Torch)
  3) Sliding window chunk ranges with overlap
  4) Session/user aggregation: last-N items + time-decayed sum
  5) LRU cache (O(1) get/put)

Notes:
  - Keep implementations simple & readable first; optimize second.
  - Tests are intentionally small.
"""

from __future__ import annotations

import math
import unittest
from collections import defaultdict, deque, OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
except ImportError as e:
    raise RuntimeError("This script requires PyTorch installed.") from e


# ============================================================
# Problem 1: Top-K most similar items (NumPy)
# ============================================================

def topk_dot_indices(q: np.ndarray, X: np.ndarray, k: int) -> np.ndarray:
    """
    Return indices of top-k items by dot product similarity.

    Inputs:
      q: (D,)
      X: (N, D)
      k: int

    Output:
      idx: (k,) indices sorted by descending score

    #TODO:
      - Implement efficiently (avoid full sort if you can).
      - Handle k <= 0 and k > N gracefully.
    """
    # TODO: implement
    if k <= 0:
        return np.array([], dtype=q.dtype)
    scores = np.matmul(X,q)
    k = min(k, scores.shape[0])
    idx = np.argpartition(scores,-k)[-k:] # biggest scores will be at the end, then select them
    # now we have to sort them
    idx = idx[np.argsort(scores[idx])[::-1]]
    return idx


# ============================================================
# Problem 2: Top-p (nucleus) filtering (Torch)
# ============================================================

def top_p_filtering(logits: torch.Tensor, p: float = 0.9) -> torch.Tensor:
    """
    Nucleus (top-p) filtering.

    Inputs:
      logits: (V,) or (B, V)
      p: float in (0,1]

    Output:
      filtered_logits: same shape as logits, where tokens outside the nucleus are set to -inf.

    Definition:
      - Sort logits descending
      - Convert to probs (softmax)
      - Keep the smallest set of tokens whose cumulative probability >= p
      - Mask out everything after that set (but keep at least 1 token)

    #TODO:
      - Implement with torch ops.
      - Do NOT loop over V.
      - Must work for both 1D (V,) and 2D (B,V) logits.
    """
    squeeze = False
    if logits.ndim == 1:
        logits2 = logits[None, :]
        squeeze = True
    elif logits.ndim == 2:
        logits2 = logits
    else:
        raise ValueError
    
    sorted_logits, sorted_idx = torch.sort(logits2, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cum = torch.cumsum(probs,dim=-1)
    
    # mask tokens after reaching p; keep at least 1
    mask = cum > p
    mask[:,0] = False
    mask[:,1:] = mask[:,:-1].clone()
    mask[:,0] = False
    
    # scatter back to original positions
    mask_orig = torch.zeros_like(mask).scatter(-1, sorted_idx, mask)
    
    
    filtered = logits2.clone()
    print(f'{filtered=}, {sorted_idx=}, {mask=}, {mask_orig=}')
    filtered[mask_orig] = float("-inf")
    return filtered[0] if squeeze else filtered


# ============================================================
# Problem 3: Chunk ranges with overlap (Sliding window)
# ============================================================

def chunk_ranges(L: int, W: int, overlap: int = 0) -> List[Tuple[int, int]]:
    """
    Produce a list of (start, end) ranges that cover [0, L) with window size W and overlap.

    Example:
      L=10, W=4, overlap=1 -> step=3 -> [(0,4),(3,7),(6,10)]

    Constraints:
      - 0 <= overlap < W
      - W > 0
      - L >= 0

    #TODO:
      - Implement without off-by-one errors.
      - Ensure last chunk ends exactly at L.
      - For L=0 return [].
    """
    assert 0 <= overlap and overlap < W and W > 0 and L >= 0
    if L == 0:
        return []
    stride = W - overlap
    res = []
    last = 0
    for i in range(0,L-stride,stride):
        res.append((i,i+W))
        last = i + W
    if last < L:
        res.append((last,L))
    return res


# ============================================================
# Problem 4: Session/user aggregation
# ============================================================

EventLastN = Tuple[str, str, int]     # (user_id, item_id, timestamp)
EventDecay = Tuple[str, float, int]   # (user_id, value, timestamp)

def last_n_items(events: Iterable[EventLastN], N: int) -> Dict[str, List[str]]:
    """
    For each user, return their last N item_ids in chronological order (assuming events are time-sorted).

    #TODO:
      - Use an efficient bounded structure (deque(maxlen=N)) per user.
      - Return {user: list_of_items}.
    """
    res: Dict[str, deque] = defaultdict(lambda: deque(maxlen=N))
    for user_id, item_id, timestamp in events:
        res[user_id].append(item_id)
    return {user: list(d) for user,d in res.items()}


def time_decay_sum(events: Iterable[EventDecay], t_now: int, tau: float) -> Dict[str, float]:
    r"""
    For each user, compute:
        S_u = sum_j value_{u,j} * exp(-(t_now - t_{u,j})/tau)

    #TODO:
      - Single pass over events.
      - Avoid recomputing 1/tau inside the loop.
      - Return a normal dict (not defaultdict).
    """
    out =defaultdict(float)
    inv_tau = 1.0/tau
    for user_id, val, ts in events:
        w = math.exp(-(t_now-ts)*inv_tau)
        out[user_id] += val*w
    return dict(out)


# ============================================================
# Problem 5: LRU Cache
# ============================================================

class LRUCache:
    """
    LRU cache supporting O(1) get and put.

    API:
      - get(key) -> value or None
      - put(key, value) -> None

    #TODO:
      - Implement using OrderedDict (acceptable for interviews) OR a manual linked-list+dict.
      - Must evict least-recently-used item when capacity exceeded.
      - get/put should update recency.
    """

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self.capacity = capacity
        self.od = OrderedDict()

    def get(self, key):
        if key not in self.od:
            return None
        self.od.move_to_end(key)
        return self.od[key]

    def put(self, key, value) -> None:
        if key in self.od:
            self.od.move_to_end(key)
        self.od[key]=value
        if len(self.od) > self.capacity:
            self.od.popitem(last=False)


# ============================================================
# Reference implementations (tests only)
# ============================================================

def _topk_dot_indices_ref(q: np.ndarray, X: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return np.array([], dtype=np.int64)
    scores = X @ q
    k = min(k, scores.shape[0])
    idx = np.argpartition(scores, -k)[-k:]
    idx = idx[np.argsort(scores[idx])[::-1]]
    return idx.astype(np.int64)

def _top_p_filtering_ref(logits: torch.Tensor, p: float) -> torch.Tensor:
    if logits.ndim == 1:
        logits2 = logits[None, :]
        squeeze = True
    elif logits.ndim == 2:
        logits2 = logits
        squeeze = False
    else:
        raise ValueError("logits must be 1D or 2D")

    sorted_logits, sorted_idx = torch.sort(logits2, dim=-1, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cum = torch.cumsum(probs, dim=-1)

    mask_sorted = cum > p
    # keep at least 1
    mask_sorted[..., 0] = False
    # shift: remove tokens AFTER threshold (common convention)
    mask_sorted[..., 1:] = mask_sorted[..., :-1].clone()
    mask_sorted[..., 0] = False

    mask = torch.zeros_like(mask_sorted).scatter(-1, sorted_idx, mask_sorted)
    out = logits2.clone()
    out[mask] = float("-inf")
    return out[0] if squeeze else out

def _chunk_ranges_ref(L: int, W: int, overlap: int) -> List[Tuple[int,int]]:
    if L == 0:
        return []
    step = W - overlap
    out = []
    s = 0
    while s < L:
        e = min(s + W, L)
        out.append((s, e))
        if e == L:
            break
        s += step
    return out

def _last_n_items_ref(events: Iterable[EventLastN], N: int) -> Dict[str, List[str]]:
    buf: Dict[str, deque] = defaultdict(lambda: deque(maxlen=N))
    for u, item, ts in events:
        buf[u].append(item)
    return {u: list(d) for u, d in buf.items()}

def _time_decay_sum_ref(events: Iterable[EventDecay], t_now: int, tau: float) -> Dict[str, float]:
    out = defaultdict(float)
    inv_tau = 1.0 / tau
    for u, val, ts in events:
        out[u] += val * math.exp(-(t_now - ts) * inv_tau)
    return dict(out)


# ============================================================
# Unit tests
# ============================================================

class TestHPMLLeetCodeDrills(unittest.TestCase):
    def test_topk_dot_indices_basic(self):
        q = np.array([1.0, 0.0])
        X = np.array([
            [1.0, 0.0],  # score 1
            [2.0, 0.0],  # score 2
            [-1.0, 0.0], # score -1
            [0.5, 0.0],  # score 0.5
        ])
        got = topk_dot_indices(q, X, k=2)
        ref = _topk_dot_indices_ref(q, X, k=2)
        self.assertTrue(np.array_equal(got, ref))

    def test_topk_dot_indices_edge_k(self):
        q = np.array([1.0, 2.0])
        X = np.random.RandomState(0).randn(5, 2)
        got0 = topk_dot_indices(q, X, k=0)
        self.assertEqual(got0.shape[0], 0)

        got_big = topk_dot_indices(q, X, k=10)
        ref_big = _topk_dot_indices_ref(q, X, k=10)
        self.assertTrue(np.array_equal(got_big, ref_big))

    def test_top_p_filtering_1d(self):
        torch.manual_seed(0)
        logits = torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0])
        got = top_p_filtering(logits, p=0.8)
        ref = _top_p_filtering_ref(logits, p=0.8)
        self.assertTrue(torch.equal(torch.isinf(got), torch.isinf(ref)))
        mask = ~torch.isinf(ref)
        self.assertTrue(torch.allclose(got[mask], ref[mask], atol=1e-6, rtol=1e-6))

    def test_top_p_filtering_2d(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 7) * 2.0
        got = top_p_filtering(logits, p=0.9)
        ref = _top_p_filtering_ref(logits, p=0.9)
        self.assertTrue(torch.equal(torch.isinf(got), torch.isinf(ref)))
        mask = ~torch.isinf(ref)
        self.assertTrue(torch.allclose(got[mask], ref[mask], atol=1e-6, rtol=1e-6))

    def test_chunk_ranges(self):
        self.assertEqual(chunk_ranges(L=0, W=4, overlap=1), [])
        self.assertEqual(chunk_ranges(L=10, W=4, overlap=1), [(0,4),(3,7),(6,10)])
        self.assertEqual(chunk_ranges(L=10, W=4, overlap=0), [(0,4),(4,8),(8,10)])
        self.assertEqual(chunk_ranges(L=3, W=10, overlap=0), [(0,3)])

    def test_last_n_items(self):
        events: List[EventLastN] = [
            ("u1", "a", 1),
            ("u1", "b", 2),
            ("u2", "x", 2),
            ("u1", "c", 3),
            ("u2", "y", 4),
            ("u1", "d", 5),
        ]
        got = last_n_items(events, N=2)
        ref = _last_n_items_ref(events, N=2)
        self.assertEqual(got, ref)
        self.assertEqual(got["u1"], ["c", "d"])
        self.assertEqual(got["u2"], ["x", "y"])

    def test_time_decay_sum(self):
        events: List[EventDecay] = [
            ("u1", 1.0, 10),
            ("u1", 2.0, 12),
            ("u2", 5.0, 11),
        ]
        t_now = 13
        tau = 2.0
        got = time_decay_sum(events, t_now=t_now, tau=tau)
        ref = _time_decay_sum_ref(events, t_now=t_now, tau=tau)
        self.assertEqual(set(got.keys()), set(ref.keys()))
        for k in got:
            self.assertAlmostEqual(got[k], ref[k], places=10)

    def test_lru_cache(self):
        c = LRUCache(capacity=2)
        self.assertIsNone(c.get("a"))
        c.put("a", 1)          # cache: a
        c.put("b", 2)          # cache: a,b
        self.assertEqual(c.get("a"), 1)  # cache: b,a
        c.put("c", 3)          # evict b -> cache: a,c
        self.assertIsNone(c.get("b"))
        self.assertEqual(c.get("c"), 3)
        c.put("a", 10)         # update a, cache: c,a
        self.assertEqual(c.get("a"), 10)
        c.put("d", 4)          # evict c -> cache: a,d
        self.assertIsNone(c.get("c"))
        self.assertEqual(c.get("d"), 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)

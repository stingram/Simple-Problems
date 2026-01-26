"""
Distributed Paged KV Cache (Interview-Style)
============================================

Single-file skeleton + tests.

Goal:
- Implement a distributed, paged KV cache for inference.
- Each (seq_id, page_index) has a single owner rank (via KVCoordinator).
- Writes may originate from ANY rank; must be routed to owner.
- Reads may originate from ANY rank; must fetch remote pages if needed.
- Each rank has limited capacity (max_pages) and must evict via LRU, but:
  - MUST NOT evict the "last page" of an active sequence.

You will implement TODO sections:
  - KVCacheShard.allocate(...)
  - KVCacheShard.get_kv(...)
  - KVCacheShard.evict_if_needed(...)

You do NOT need to implement the network; Cluster is provided.

Run:
  python kv_cache_problem2.py

Once TODOs are implemented correctly, all tests should pass.
"""

from __future__ import annotations

import numpy as np
from collections import OrderedDict, defaultdict, namedtuple
from dataclasses import dataclass
from math import inf
from typing import Dict, Tuple, List, Optional, Set


# -----------------------------
# Simulated cluster primitives
# -----------------------------

class Cluster:
    """
    A very small blocking-message simulation.

    - send(src, dst, tag, payload): enqueue message in (dst, tag)
    - recv(dst, tag): dequeue oldest message for (dst, tag), asserts non-empty
    - has_messages(dst, tag): whether mailbox non-empty
    """
    def __init__(self, world_size: int):
        self.world_size = world_size
        self.mailboxes = defaultdict(list)  # key=(dst, tag) -> List[(src, payload)]

    def send(self, src: int, dst: int, tag: str, payload):
        self.mailboxes[(dst, tag)].append((src, payload))

    def recv(self, dst: int, tag: str):
        msgs = self.mailboxes[(dst, tag)]
        assert len(msgs) > 0, f"Deadlock or missing send: dst={dst}, tag={tag}"
        return msgs.pop(0)

    def has_messages(self, dst: int, tag: str) -> bool:
        return len(self.mailboxes.get((dst, tag), [])) > 0


class RankContext:
    def __init__(self, rank: int, world_size: int, cluster: Cluster):
        self.rank = rank
        self.world_size = world_size
        self.cluster = cluster


# -----------------------------
# KV Page + Coordinator
# -----------------------------

@dataclass
class KVPage:
    seq_id: int
    page_index: int
    K: np.ndarray  # [page_size, n_heads, head_dim]
    V: np.ndarray  # [page_size, n_heads, head_dim]


class KVCoordinator:
    """
    Global mapping: (seq_id, page_index) -> owner_rank

    NOTE: The mapping must be stable and deterministic.
    """
    def __init__(self, world_size: int):
        self.world_size = world_size
        self.owners: Dict[Tuple[int, int], int] = {}

    def assign_owner(self, seq_id: int, page_index: int) -> int:
        owner = (seq_id + page_index) % self.world_size
        self.owners[(seq_id, page_index)] = owner
        return owner

    def get_owner(self, seq_id: int, page_index: int) -> int:
        return self.owners[(seq_id, page_index)]

    def ensure_owner(self, seq_id: int, page_index: int) -> int:
        if (seq_id, page_index) not in self.owners:
            return self.assign_owner(seq_id, page_index)
        return self.get_owner(seq_id, page_index)


# -----------------------------
# Helpers: paging and routing
# -----------------------------

def page_of(pos: int, page_size: int) -> int:
    return pos // page_size

def offset_in_page(pos: int, page_size: int) -> int:
    return pos % page_size


# -----------------------------
# KV Cache Shard (per-rank)
# -----------------------------

class KVCacheShard:
    """
    Per-rank KV storage.

    Local storage contains BOTH:
      - "owned pages" (authoritative)
      - optional cached copies of remote pages (non-authoritative)

    For this interview problem:
      - Treat any page stored locally as readable.
      - Ownership must be respected for writes and authoritative responses.

    LRU applies to local pages stored in self.pages (owned or cached).
    """
    def __init__(
        self,
        ctx: RankContext,
        coordinator: KVCoordinator,
        page_size: int,
        max_pages: int,
        n_heads: int,
        head_dim: int,
    ):
        self.ctx = ctx
        self.coord = coordinator
        self.page_size = page_size
        self.max_pages = max_pages
        self.n_heads = n_heads
        self.head_dim = head_dim

        # Local storage: (seq_id, page_index) -> KVPage
        self.pages: Dict[Tuple[int, int], KVPage] = {}

        # LRU tracking: keys are (seq_id, page_index); most-recent at end
        self.lru: "OrderedDict[Tuple[int,int], None]" = OrderedDict()

        # Active sequences: their last page MUST NOT be evicted
        self.active_sequences: Set[int] = set()

        # For pinning: track last written position per active seq
        self._last_pos: Dict[int, int] = {}

    # -------------------------
    # Network handlers (given)
    # -------------------------

    def _handle_write_requests_once(self) -> bool:
        """
        Handle a single WRITE_PAGE message if present.
        Payload:
          (seq_id, page_index, positions_in_page, K_slice, V_slice)
        Where:
          positions_in_page: List[int] offsets within the page
          K_slice, V_slice: [len(positions_in_page), n_heads, head_dim]
        """
        if not self.ctx.cluster.has_messages(self.ctx.rank, "WRITE_PAGE"):
            return False
        src, payload = self.ctx.cluster.recv(self.ctx.rank, "WRITE_PAGE")
        seq_id, page_index, pos_in_page, K_slice, V_slice = payload

        key = (seq_id, page_index)
        if key not in self.pages:
            # Allocate empty page
            K = np.zeros((self.page_size, self.n_heads, self.head_dim), dtype=np.float32)
            V = np.zeros((self.page_size, self.n_heads, self.head_dim), dtype=np.float32)
            self.pages[key] = KVPage(seq_id, page_index, K, V)

        page = self.pages[key]
        for i, off in enumerate(pos_in_page):
            page.K[off] = K_slice[i]
            page.V[off] = V_slice[i]

        # LRU update on write
        self._touch(key)

        # Acknowledge
        self.ctx.cluster.send(self.ctx.rank, src, "WRITE_PAGE_ACK", (seq_id, page_index))
        return True

    def _handle_fetch_request_once(self) -> bool:
        """
        Handle a single FETCH_PAGE request if present.
        Payload:
          (seq_id, page_index)
        Respond with:
          (seq_id, page_index, K, V)
        """
        if not self.ctx.cluster.has_messages(self.ctx.rank, "FETCH_PAGE"):
            return False
        src, payload = self.ctx.cluster.recv(self.ctx.rank, "FETCH_PAGE")
        seq_id, page_index = payload
        key = (seq_id, page_index)
        assert key in self.pages, f"Owner rank missing page {key} (bug in writes?)"
        page = self.pages[key]

        # LRU update on read (owner-side)
        self._touch(key)

        self.ctx.cluster.send(
            self.ctx.rank,
            src,
            "FETCH_PAGE_RESP",
            (seq_id, page_index, page.K.copy(), page.V.copy()),
        )
        return True

    def handle_network_once(self) -> bool:
        """
        Process at most one network message (write or fetch).
        Returns True if processed a message, else False.
        """
        # Prefer writes to make sure pages exist before fetches
        if self._handle_write_requests_once():
            return True
        if self._handle_fetch_request_once():
            return True
        return False

    # -------------------------
    # Local helpers (given)
    # -------------------------

    def _touch(self, key: Tuple[int, int]) -> None:
        """Mark key as most-recent in LRU."""
        if key in self.lru:
            self.lru.move_to_end(key)
        else:
            self.lru[key] = None

    def _get_pinned_key_for_seq(self, seq_id: int) -> Optional[Tuple[int, int]]:
        """
        Returns the (seq_id, last_page_index) that must not be evicted if active.
        If seq is not active or has no last_pos known, returns None.
        """
        if seq_id not in self.active_sequences:
            return None
        if seq_id not in self._last_pos:
            return None
        last_page = page_of(self._last_pos[seq_id], self.page_size)
        return (seq_id, last_page)

    def set_inactive(self, seq_id: int) -> None:
        """Sequence finished; allows its last page to be evicted."""
        self.active_sequences.discard(seq_id)

    # -------------------------
    # TODO 1: Allocate / write
    # -------------------------
    def allocate(
        self,
        seq_id: int,
        token_positions: List[int],
        K_new: np.ndarray,
        V_new: np.ndarray,
    ):
        """
        Write new KV entries for the given token positions.

        token_positions: global token indices for this sequence
        K_new, V_new: [len(token_positions), n_heads, head_dim]

        Requirements:
        - For each token position:
            page_index = pos // page_size
            off       = pos % page_size
        - Ownership:
            owner = coordinator.ensure_owner(seq_id, page_index)
            * If owner == self.rank: write locally (allocate page if absent)
            * Else: send WRITE_PAGE to owner, and wait for WRITE_PAGE_ACK
        - Must support positions spanning multiple pages
        - Mark seq_id active, update last_pos[seq_id] to max written position
        - Update LRU for any local pages written
        - After writes, call evict_if_needed() locally AND owners should evict too
          (owners can evict as they receive writes; your implementation may
           explicitly call evict_if_needed() when writing locally)
        """
        # TODO: implement
        
        # Check new KV entries match expected dimensions
        assert K_new.shape == [len(token_positions), self.n_heads, self.head_dim]
        assert V_new.shape == [len(token_positions), self.n_heads, self.head_dim]
        
        # mark active and record last token position
        self.active_sequences.add(seq_id)
        self._last_pos[seq_id] = max(self._last_pos.get(seq_id, -inf), max(token_positions))
        
        # create dictionary of page writes
        # i is row
        page_writes: Dict[int, List[int]] = {}
        for i in range(len(token_positions)):
            pos = token_positions[i]
            p = pos // self.page_size
            off = pos % self.page_size
            page_writes[p].append((off, i))
        
        # for each page bucket, decide local vs remote owner and perform write
        # collect remote sends so we can send-first-then-wait
        RemoteSend = namedtuple('RemoteSend', ['owner_rank', 'page_index', 'pos_in_page_list', 'K_slice', 'V_slice'])
        remote_sends: List[RemoteSend] = []
        for page in page_writes.keys():
            owner = self.coord.ensure_owner(seq_id, page)
            offsets_and_rows = page_writes[p]
            pos_in_page_list = [off for (off, _) in offsets_and_rows]
            rows = [i for (_, i) in offsets_and_rows]
            
            # use rows to index slices for this page
            K_slice = K_new[rows, :, :]
            V_slice = V_new[rows, :, :]
            
            if owner == self.rank:
                key = (seq_id, page)
                if key not in self.pages:
                    # make new page
                    K_page = np.zeros((self.page_size,self.n_heads,self.head_dim))
                    V_page = np.zeros((self.page_size,self.n_heads,self.head_dim))
                    self.pages[key] = KVPage(seq_id,page,K_page,V_page)
                    
                for j in range(len(pos_in_page_list)):
                    off = pos_in_page_list[j]
                    self.pages[key].K[off, :, :] = K_slice[j, :, :]
                    self.pages[key].V[off, :, :] = V_slice[j, :, :]
                # make sure to update LRU
                self._touch(key)
                        
            else:
                remote_sends.append(RemoteSend(owner,page,pos_in_page_list,K_slice,V_slice))
                
        # now we can send
        for remote_send in remote_sends:
            self.ctx.cluster.send(
                src=self.rank,
                dst=remote_send.owner_rank,
                tag="WRITE_PAGE",
                payload=(seq_id,
                         remote_send.page_index,
                         remote_send.pos_in_page_list,
                         remote_send.K_slice.copy(),
                         remote_send.V_slice.copy())
            )
            
        
        raise NotImplementedError

    # -------------------------
    # TODO 2: Read / fetch
    # -------------------------
    def get_kv(
        self,
        seq_id: int,
        token_positions: List[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return K, V for the requested token positions
        Shape: [len(token_positions), n_heads, head_dim]

        Requirements:
        - Determine required pages per position.
        - For each required page:
            owner = coordinator.get_owner(seq_id, page_index)
            * If page exists locally: use it
            * Else if owner != self.rank: fetch from owner via FETCH_PAGE / RESP
              - Store fetched page locally as a cached copy (optional but expected)
              - Update LRU on access
        - Assemble outputs in the same order as token_positions
        - Update LRU for every page you touch locally (owned or cached)
        - If local storage exceeds max_pages after caching, evict_if_needed()
        """
        # TODO: implement
        raise NotImplementedError

    # -------------------------
    # TODO 3: Eviction
    # -------------------------
    def evict_if_needed(self):
        """
        Evict local pages until len(self.pages) <= max_pages

        Rules:
        - LRU eviction
        - Must NOT evict the last page of an active sequence (pin)
          Pin key for each active seq is: (seq_id, page_of(last_pos[seq_id]))
        - If everything is pinned and you're over capacity, you may raise AssertionError
          (tests avoid that case).
        - Eviction must remove from BOTH self.pages and self.lru
        """
        # TODO: implement
        raise NotImplementedError


# -----------------------------
# Test harness
# -----------------------------

def drive_network_until_quiescent(shards: List[KVCacheShard], max_steps: int = 100000) -> None:
    """
    Because our Cluster is synchronous blocking but our test code isn't truly parallel,
    we "drive" the network by repeatedly letting ranks process one message at a time.

    This simulates progress in a distributed program.
    """
    steps = 0
    while steps < max_steps:
        progressed = False
        for sh in shards:
            if sh.handle_network_once():
                progressed = True
        if not progressed:
            return
        steps += 1
    raise RuntimeError("Network did not quiesce (possible protocol deadlock).")


def make_world(
    world_size: int,
    page_size: int,
    max_pages: int,
    n_heads: int,
    head_dim: int,
):
    cluster = Cluster(world_size)
    coord = KVCoordinator(world_size)
    shards: List[KVCacheShard] = []
    for r in range(world_size):
        ctx = RankContext(r, world_size, cluster)
        shards.append(KVCacheShard(ctx, coord, page_size, max_pages, n_heads, head_dim))
    return cluster, coord, shards


def assert_allclose(a: np.ndarray, b: np.ndarray, atol=0, rtol=0):
    assert a.shape == b.shape, f"shape mismatch {a.shape} vs {b.shape}"
    if not np.allclose(a, b, atol=atol, rtol=rtol):
        diff = np.max(np.abs(a - b))
        raise AssertionError(f"arrays differ; max abs diff={diff}")


def test_basic_write_read_remote():
    """
    Write from a non-owner rank, read from a different rank, spans pages & owners.
    This forces:
      - write routing
      - remote fetch
      - correct assembly order
    """
    world_size = 2
    page_size = 4
    max_pages = 8
    n_heads = 2
    head_dim = 3

    _, coord, shards = make_world(world_size, page_size, max_pages, n_heads, head_dim)

    seq_id = 42
    positions = list(range(0, 10))  # spans pages 0,1,2

    # Create identifiable K/V
    # K[t,h,d] = 1000*t + 100*h + d
    K_new = np.zeros((len(positions), n_heads, head_dim), dtype=np.float32)
    V_new = np.zeros((len(positions), n_heads, head_dim), dtype=np.float32)
    for i, pos in enumerate(positions):
        for h in range(n_heads):
            for d in range(head_dim):
                K_new[i, h, d] = 1000 * pos + 100 * h + d
                V_new[i, h, d] = 2000 * pos + 100 * h + d

    # Writes originate from rank 1 (could be non-owner for some pages)
    shards[1].allocate(seq_id, positions, K_new, V_new)
    drive_network_until_quiescent(shards)

    # Sanity: owners have the authoritative pages
    for page_index in [0, 1, 2]:
        owner = coord.ensure_owner(seq_id, page_index)
        key = (seq_id, page_index)
        assert key in shards[owner].pages, f"owner {owner} missing page {key}"

    # Read from rank 0 in non-contiguous order (forces assembly correctness)
    read_pos = [3, 4, 5, 8]
    K_out, V_out = shards[0].get_kv(seq_id, read_pos)
    drive_network_until_quiescent(shards)

    # Expected values
    expK = np.zeros((len(read_pos), n_heads, head_dim), dtype=np.float32)
    expV = np.zeros((len(read_pos), n_heads, head_dim), dtype=np.float32)
    for i, pos in enumerate(read_pos):
        for h in range(n_heads):
            for d in range(head_dim):
                expK[i, h, d] = 1000 * pos + 100 * h + d
                expV[i, h, d] = 2000 * pos + 100 * h + d

    assert_allclose(K_out, expK)
    assert_allclose(V_out, expV)


def test_eviction_lru_and_pinning():
    """
    Force local cache pressure and verify:
      - LRU eviction happens
      - pinned "last page of active sequences" is not evicted
    """
    world_size = 2
    page_size = 4
    max_pages = 2   # VERY small to force eviction
    n_heads = 1
    head_dim = 2

    _, coord, shards = make_world(world_size, page_size, max_pages, n_heads, head_dim)

    # We'll do everything from rank 0 to create local cache copies via reads.
    r = 0
    sh = shards[r]

    # Sequence A: write tokens 0..7 => pages 0 and 1
    seqA = 10
    posA = list(range(0, 8))
    KA = np.random.randn(len(posA), n_heads, head_dim).astype(np.float32)
    VA = np.random.randn(len(posA), n_heads, head_dim).astype(np.float32)
    sh.allocate(seqA, posA, KA, VA)
    drive_network_until_quiescent(shards)

    # Sequence B: write tokens 0..7 => pages 0 and 1
    seqB = 11
    posB = list(range(0, 8))
    KB = np.random.randn(len(posB), n_heads, head_dim).astype(np.float32)
    VB = np.random.randn(len(posB), n_heads, head_dim).astype(np.float32)
    sh.allocate(seqB, posB, KB, VB)
    drive_network_until_quiescent(shards)

    # Both sequences active now; pinned pages are their last page (page 1)
    pinA = (seqA, 1)
    pinB = (seqB, 1)

    # Now perform reads on rank 0 to populate its local cache with multiple remote pages.
    # Read a token from page 0 of each seq first (to make those pages "recent" in LRU),
    # then read from other pages to exceed capacity.
    sh.get_kv(seqA, [0])  # touches page 0
    drive_network_until_quiescent(shards)
    sh.get_kv(seqB, [0])  # touches page 0
    drive_network_until_quiescent(shards)

    # Now fetch page 1 for A and B, which are pinned. This will exceed local max_pages=2.
    # Eviction must NOT evict pinned pages if they're local.
    sh.get_kv(seqA, [7])  # touches page 1 (pinned)
    drive_network_until_quiescent(shards)
    sh.get_kv(seqB, [7])  # touches page 1 (pinned)
    drive_network_until_quiescent(shards)

    # After these accesses, local cache must have <= max_pages
    assert len(sh.pages) <= max_pages, f"local pages={len(sh.pages)} exceeds max_pages={max_pages}"

    # If pinned pages exist locally, they must not have been evicted.
    # It's possible one pinned page is not local if owner==other rank and caching isn't implemented,
    # but expected solution caches. We'll assert that if present, it's present.
    if pinA in sh.pages:
        assert pinA in sh.pages, "Pinned page A was evicted locally (bug)"
    if pinB in sh.pages:
        assert pinB in sh.pages, "Pinned page B was evicted locally (bug)"


def test_read_order_and_cross_page():
    """
    Writes a sequence, then reads positions that cross page boundaries and are out of order.
    """
    world_size = 3
    page_size = 4
    max_pages = 10
    n_heads = 2
    head_dim = 2

    _, _, shards = make_world(world_size, page_size, max_pages, n_heads, head_dim)

    seq = 7
    positions = list(range(0, 13))  # pages 0..3

    K_new = np.zeros((len(positions), n_heads, head_dim), dtype=np.float32)
    V_new = np.zeros((len(positions), n_heads, head_dim), dtype=np.float32)
    for i, pos in enumerate(positions):
        K_new[i] = (pos + 1) * 1.0
        V_new[i] = (pos + 1) * 10.0

    # Write from rank 2
    shards[2].allocate(seq, positions, K_new, V_new)
    drive_network_until_quiescent(shards)

    # Read from rank 1
    read_pos = [4, 3, 8, 9, 0, 12]  # crosses pages and out-of-order
    K_out, V_out = shards[1].get_kv(seq, read_pos)
    drive_network_until_quiescent(shards)

    expK = np.stack([np.full((n_heads, head_dim), (p + 1) * 1.0, dtype=np.float32) for p in read_pos], axis=0)
    expV = np.stack([np.full((n_heads, head_dim), (p + 1) * 10.0, dtype=np.float32) for p in read_pos], axis=0)

    assert_allclose(K_out, expK)
    assert_allclose(V_out, expV)


def run_all_tests():
    print("Running tests (will FAIL until you implement TODOs)...")
    test_basic_write_read_remote()
    print("  ✓ test_basic_write_read_remote")
    test_eviction_lru_and_pinning()
    print("  ✓ test_eviction_lru_and_pinning")
    test_read_order_and_cross_page()
    print("  ✓ test_read_order_and_cross_page")
    print("All tests passed ✅")


if __name__ == "__main__":
    run_all_tests()

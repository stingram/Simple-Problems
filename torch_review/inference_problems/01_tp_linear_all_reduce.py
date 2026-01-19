import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional, List


# -----------------------------
# Single-process distributed harness
# -----------------------------

class BlockedRecv(Exception):
    """Raised when a recv() would block (message not yet available)."""
    def __init__(self, src: int, tag: str):
        super().__init__(f"Blocked waiting for src={src}, tag={tag}")
        self.src = src
        self.tag = tag


class Cluster:
    """
    Simulated message-passing cluster with per-(dst, src, tag) mailboxes.
    send() is non-blocking. recv() blocks by raising BlockedRecv if absent.

    A cooperative scheduler (run_all_ranks) makes progress by re-running ranks
    until all complete.
    """
    def __init__(self, world_size: int):
        self.world_size = world_size
        self.mailboxes: Dict[Tuple[int, int, str], List[np.ndarray]] = {}

    def send(self, src: int, dst: int, tag: str, payload: np.ndarray):
        key = (dst, src, tag)
        self.mailboxes.setdefault(key, []).append(payload)

    def recv(self, dst: int, src: int, tag: str) -> np.ndarray:
        key = (dst, src, tag)
        q = self.mailboxes.get(key, [])
        if not q:
            raise BlockedRecv(src=src, tag=tag)
        return q.pop(0)


@dataclass
class RankContext:
    rank: int
    world_size: int
    cluster: Cluster

    # You may find these helpers useful
    def send(self, dst: int, tag: str, payload: np.ndarray):
        self.cluster.send(src=self.rank, dst=dst, tag=tag, payload=payload)

    def recv(self, src: int, tag: str) -> np.ndarray:
        return self.cluster.recv(dst=self.rank, src=src, tag=tag)

    def next_rank(self) -> int:
        return (self.rank + 1) % self.world_size

    def prev_rank(self) -> int:
        return (self.rank - 1 + self.world_size) % self.world_size

    # -----------------------------
    # TODO collectives (your work)
    # -----------------------------
    def _send_chunk_id(self, rank:int, step: int):
        return (rank + 1 + step) % self.world_size

    def _send_chunk_id_ag(self, rank: int, step: int):
        return (rank - step) % self.world_size

    def all_reduce_sum_ring(self, x: np.ndarray, tag_base: str = "ar") -> np.ndarray:
        """
        Ring all-reduce (sum) over x.

        Requirements:
        - Same code on all ranks.
        - Works for any world_size >= 1.
        - Must not allocate O(world_size * x) extra memory (O(x) ok).
        - Deterministic.

        Hint (classic ring):
        1) Reduce-scatter phase: circulate chunks and reduce into owner chunk
        2) All-gather phase: circulate reduced chunks to everyone

        But to keep the interview scope sane, you may implement a simpler ring:
        - circulate full buffer world_size-1 steps, summing as you go
        This is O(p) bandwidth vs optimal, but still ring-structured and correct.

        Choose one approach and implement it.
        """
        # TODO: implement
        print(f'Doing allreduce...')
        # Do reduce scatter phase
        n = x.size
        assert n % self.world_size == 0
        # flatten
        xf = x.reshape(-1)
        block_size = n // self.world_size
        chunk_inds = [(i*block_size,(i+1)*block_size) for i in range(self.world_size)]
        
        print(f'{self.world_size=}')
        for i in range(self.world_size-1):
            send_chunk_id = self._send_chunk_id(self.rank, i)
            receive_chunk_id = self._send_chunk_id(self.next_rank(),i)
            buffer_to_send = xf[chunk_inds[send_chunk_id][0]:chunk_inds[send_chunk_id][1]].copy()
            print(
                f"rank={self.rank}, step={i}, "
                f"send_chunk={send_chunk_id}, "
                f"expect_recv_chunk={receive_chunk_id}"
                )
            # send data
            self.send(self.prev_rank(),f'{tag_base}:rs:{self.rank}:{i}:{send_chunk_id}',buffer_to_send)
            payload = self.recv(self.next_rank(),f'{tag_base}:rs:{self.next_rank()}:{i}:{receive_chunk_id}')
            xf[chunk_inds[receive_chunk_id][0]:chunk_inds[receive_chunk_id][1]] += payload         
        
        # Do all-gather phase
        for i in range(self.world_size-1):
            # send data
            send_chunk_id = self._send_chunk_id_ag(self.rank, i)
            receive_chunk_id = self._send_chunk_id_ag(self.prev_rank(),i)
            buffer_to_send = xf[chunk_inds[send_chunk_id][0]:chunk_inds[send_chunk_id][1]].copy()
            print(
                f"rank={self.rank}, step={i}, "
                f"send_chunk={send_chunk_id}, "
                f"expect_recv_chunk={receive_chunk_id}"
                )
            self.send(self.next_rank(),f'{tag_base}:ag:{self.rank}:{i}:{send_chunk_id}',buffer_to_send)
            payload = self.recv(self.prev_rank(),f'{tag_base}:ag:{self.prev_rank()}:{i}:{receive_chunk_id}')
            xf[chunk_inds[receive_chunk_id][0]:chunk_inds[receive_chunk_id][1]] = payload
        return x


    def all_gather_concat_ring(
        self,
        x: np.ndarray,
        axis: int,
        tag_base: str = "ag",
    ) -> np.ndarray:
        """
        Ring all-gather that concatenates shards along 'axis'.

        Each rank starts with its local shard x.
        Returns the full tensor which is concat of shards in rank order:
        [rank0_shard, rank1_shard, ..., rank(p-1)_shard] along 'axis'.

        Requirements:
        - Same code on all ranks.
        - Works for any world_size.
        - Use ring send/recv (neighbor communication).
        - Avoid deadlock: all ranks should send and recv in consistent order.

        Simple ring method:
        - Maintain a list of shards you have so far (start with your own).
        - For step s in 0..p-2:
            send the shard that originated from rank (rank - s) to next_rank
            recv a shard from prev_rank
            store it in correct position
        """
        # TODO: implement
        # Do reduce scatter phase
        block_size = x.shape[axis]
        chunk_inds = [(i*block_size,(i+1)*block_size) for i in range(self.world_size)]

        # allocate output
        shape = list(x.shape)
        shape[axis] *= self.world_size
        out = np.empty(tuple(shape),dtype=x.dtype)
        
        # write my own chunk first
        send_chunk_id = self.rank
        idx = [slice(None)]*len(x.shape)
        idx[axis] = slice(chunk_inds[send_chunk_id][0],chunk_inds[send_chunk_id][1])
        out[tuple(idx)] = x.copy()

        # Do all-gather phase
        for i in range(self.world_size-1):
            # send data
            send_chunk_id = self._send_chunk_id_ag(self.rank, i)
            receive_chunk_id = self._send_chunk_id_ag(self.prev_rank(),i)
            send_idx = [slice(None)]*len(x.shape)
            send_idx[axis] = slice(chunk_inds[send_chunk_id][0],chunk_inds[send_chunk_id][1])
            buffer_to_send = out[tuple(send_idx)].copy()
            print(
                f"rank={self.rank}, step={i}, "
                f"send_chunk={send_chunk_id}, "
                f"expect_recv_chunk={receive_chunk_id}"
                )
            self.send(self.next_rank(),f'{tag_base}:ag:{self.rank}:{i}:{send_chunk_id}',buffer_to_send)
            payload = self.recv(self.prev_rank(),f'{tag_base}:ag:{self.prev_rank()}:{i}:{receive_chunk_id}')
            receive_idx = [slice(None)]*len(x.shape)
            receive_idx[axis] = slice(chunk_inds[receive_chunk_id][0],chunk_inds[receive_chunk_id][1])
            out[tuple(receive_idx)] = payload

        return out


# -----------------------------
# TP Linear op to implement
# -----------------------------

class TPLinear:
    """
    Tensor-parallel linear layer.

    mode:
      - "col": column-parallel (W sharded on out_features)
      - "row": row-parallel (W sharded on in_features)

    Shapes:
      X: [B, in_features]
      W_full: [in_features, out_features]

    Column-parallel:
      W_shard: [in_features, out_features/world]
      Y_local = X @ W_shard -> [B, out_features/world]
      Y = all_gather_concat(Y_local, axis=1) -> [B, out_features]

    Row-parallel:
      X_shard: [B, in_features/world]
      W_shard: [in_features/world, out_features]
      Y_partial = X_shard @ W_shard -> [B, out_features]
      Y = all_reduce_sum(Y_partial)
    """
    def __init__(self, ctx: RankContext, mode: str, W_shard: np.ndarray, bias: Optional[np.ndarray] = None):
        assert mode in ("col", "row")
        self.ctx = ctx
        self.mode = mode
        self.W_shard = W_shard
        self.bias = bias  # bias is full [out_features] (for simplicity)

    def forward(self, X_full: np.ndarray) -> np.ndarray:
        """
        TODO:
        - implement both modes correctly.
        - in row mode: shard X on last dim by rank (contiguous shard)
        - in col mode: do not shard X; shard is already in W_shard
        - add bias at end (bias is full)
        """
        # TODO: implement
        if self.mode == "col":
            assert X_full.shape[-1] == self.W_shard.shape[0]
            Y_local = X_full @ self.W_shard
            Y = self.ctx.all_gather_concat_ring(Y_local, axis=-1)
        elif self.mode == "row":
            assert X_full.shape[-1] == self.W_shard.shape[0] * self.ctx.world_size 
            X_partial = shard_dim_contiguous(X_full,-1,self.ctx.rank,self.ctx.world_size)
            Y_partial = X_partial @ self.W_shard
            # print(f'{Y_partial=}')
            Y = self.ctx.all_reduce_sum_ring(Y_partial)
        else:
            raise NotImplementedError
        if self.bias is not None:
            Y += self.bias
        return Y


# -----------------------------
# Cooperative runner
# -----------------------------

def run_all_ranks(world_size: int, per_rank_fn):
    """
    per_rank_fn(ctx) should return a final np.ndarray (or any object) for that rank.

    This runs all ranks "concurrently" using cooperative scheduling:
    - If a rank tries to recv before a message exists, it raises BlockedRecv
    - The scheduler keeps cycling until all ranks complete.
    """
    cluster = Cluster(world_size=world_size)
    ctxs = [RankContext(rank=r, world_size=world_size, cluster=cluster) for r in range(world_size)]

    done = [False] * world_size
    results: List[Any] = [None] * world_size

    # Wrap each rank call so we can re-enter it after blocking.
    # We'll do this by storing a lambda that retries from scratch each time,
    # but that would repeat work. Instead: require per_rank_fn to be pure given ctx
    # and use deterministic message tags so retrying is safe.
    #
    # In a real interview harness, you'd structure code to avoid re-entry.
    # For practice, this is fine and keeps the simulation simple.
    progress = True
    safety_iters = 0
    while not all(done):
        safety_iters += 1
        if safety_iters > 200000:
            raise RuntimeError("Scheduler appears stuck (possible deadlock).")

        progress = False
        for r in range(world_size):
            if done[r]:
                continue
            try:
                results[r] = per_rank_fn(ctxs[r])
                done[r] = True
                progress = True
            except BlockedRecv:
                # This rank is waiting; try others.
                pass

        if not progress:
            # Nobody completed in this full pass; but maybe some sends happened
            # and now recvs can proceed. Keep looping.
            # If truly deadlocked, safety_iters will trip.
            pass

    return results


# -----------------------------
# Reference helpers (for tests)
# -----------------------------

def shard_dim_contiguous(x: np.ndarray, axis: int, rank: int, world: int) -> np.ndarray:
    n = x.shape[axis]
    assert n % world == 0, "Dimension must be divisible by world_size in this exercise"
    per = n // world
    slc = [slice(None)] * x.ndim
    slc[axis] = slice(rank * per, (rank + 1) * per)
    return x[tuple(slc)]


# -----------------------------
# Tests
# -----------------------------

def test_column_parallel(world_size: int, B: int, in_f: int, out_f: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((B, in_f), dtype=np.float32)
    W = rng.standard_normal((in_f, out_f), dtype=np.float32)
    bias = rng.standard_normal((out_f,), dtype=np.float32)

    assert out_f % world_size == 0

    def per_rank(ctx: RankContext):
        W_shard = shard_dim_contiguous(W, axis=1, rank=ctx.rank, world=ctx.world_size)
        layer = TPLinear(ctx=ctx, mode="col", W_shard=W_shard, bias=bias)
        Y = layer.forward(X)
        return Y

    Ys = run_all_ranks(world_size, per_rank)
    Y_ref = X @ W + bias

    for r, Y in enumerate(Ys):
        np.testing.assert_allclose(Y, Y_ref, rtol=1e-4, atol=1e-4), f"col-parallel mismatch on rank {r}"


def test_row_parallel(world_size: int, B: int, in_f: int, out_f: int, seed: int = 1):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((B, in_f), dtype=np.float32)
    W = rng.standard_normal((in_f, out_f), dtype=np.float32)
    bias = rng.standard_normal((out_f,), dtype=np.float32)

    assert in_f % world_size == 0

    def per_rank(ctx: RankContext):
        # Row-parallel shards input features of W (axis=0)
        W_shard = shard_dim_contiguous(W, axis=0, rank=ctx.rank, world=ctx.world_size)
        layer = TPLinear(ctx=ctx, mode="row", W_shard=W_shard, bias=bias)
        Y = layer.forward(X)
        return Y

    Ys = run_all_ranks(world_size, per_rank)
    Y_ref = X @ W + bias

    for r, Y in enumerate(Ys):
        np.testing.assert_allclose(Y, Y_ref, rtol=1e-4, atol=1e-4), f"row-parallel mismatch on rank {r}"


def run_tests():
    # A small but meaningful sweep
    configs = [
        (1, 4, 8, 12),
        (2, 4, 8, 12),
        (4, 2, 16, 32),
        (4, 8, 32, 16),
    ]
    for (world, B, in_f, out_f) in configs:
        # Make divisibility valid
        if out_f % world == 0:
            test_column_parallel(world, B, in_f, out_f)
        if in_f % world == 0:
            test_row_parallel(world, B, in_f, out_f)
    print("✅ All tests passed (if your TODOs are implemented correctly).")


if __name__ == "__main__":
    run_tests()

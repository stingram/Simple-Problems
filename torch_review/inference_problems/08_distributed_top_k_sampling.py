import numpy as np
from typing import Tuple, Dict


# =========================
# Infrastructure (DO NOT MODIFY)
# =========================

class Cluster:
    """
    Single-process simulation of distributed message passing.
    """
    def __init__(self, world_size: int):
        self.world_size = world_size
        self.mailboxes: Dict[Tuple[int, int, str], np.ndarray] = {}

    def send(self, src: int, dst: int, tag: str, payload: np.ndarray):
        key = (src, dst, tag)
        if key in self.mailboxes:
            raise RuntimeError(f"Duplicate send {key}")
        self.mailboxes[key] = payload

    def recv(self, src: int, dst: int, tag: str) -> np.ndarray:
        key = (src, dst, tag)
        if key not in self.mailboxes:
            raise RuntimeError(f"Missing message {key}")
        return self.mailboxes.pop(key)


class RankContext:
    def __init__(self, rank: int, world_size: int, cluster: Cluster):
        self.rank = rank
        self.world_size = world_size
        self.cluster = cluster


# =========================
# Your Task
# =========================

def sample_topk_sharded(
    logits_local: np.ndarray,
    k: int,
    temperature: float,
    ctx: RankContext,
    greedy: bool = False,
) -> int:
    """
    Args:
        logits_local: [vocab_shard] float32 logits for this rank
        k: global top-k (k <= vocab_shard * world_size)
        temperature: softmax temperature (>0)
        ctx: RankContext
        greedy: if True, return global argmax (ignores k/temperature)

    Returns:
        global_token_id: int in [0, vocab_size)
    """
    # TODO: Implement distributed top-k sampling without gathering full vocab.
    #
    # Requirements:
    #  1) Each rank computes local top-k candidates from logits_local.
    logits_local_top_k = logits_local[np.argmax(logits_local,-1)[:k]]
    
    #  2) Convert local indices -> global token ids using:
    #        vocab_shard = logits_local.shape[0]
    #        global_id = ctx.rank * vocab_shard + local_index
    vocab_
    #  3) Send only your local candidates to rank 0 (coordinator).
    #  4) Rank 0 merges candidates from all ranks, selects global top-k.
    #  5) If greedy=True: select global argmax token.
    #     Else: apply temperature, softmax over the global top-k, sample a token.
    #     Sampling MUST be deterministic across ranks: only rank 0 samples.
    #  6) Rank 0 broadcasts chosen token id to all ranks; all ranks return it.
    #
    # Communication contract (suggested):
    #  - Each rank sends a single payload to rank 0 with tag f"cand/{step}"
    #  - Rank 0 broadcasts token id with tag f"out/{step}"
    #
    # Note: This function may be called multiple times in a longer program.
    # If you need unique tags, you can derive them from a hash of logits_local
    # or add a monotonically increasing counter stored on ctx (but ctx has no state).
    # For this exercise/tests, a fixed tag is OK because calls are not concurrent.
    
    
    
    raise NotImplementedError


# =========================
# Test Harness (DO NOT MODIFY)
# =========================

def run_all_ranks(fn, inputs_per_rank, ctxs):
    """
    Calls fn on each rank sequentially. This simulates SPMD code where each rank
    runs the same function, but our single-process harness executes ranks in order.
    """
    results = [None] * len(ctxs)
    for r, ctx in enumerate(ctxs):
        results[r] = fn(inputs_per_rank[r], ctx)
    return results


def test_topk_sharded_greedy():
    np.random.seed(0)
    world_size = 4
    vocab_size = 64
    shard = vocab_size // world_size

    logits = np.random.randn(vocab_size).astype(np.float32)
    cluster = Cluster(world_size)
    ctxs = [RankContext(r, world_size, cluster) for r in range(world_size)]

    logits_per_rank = [logits[r*shard:(r+1)*shard] for r in range(world_size)]

    def wrapped(logits_local, ctx):
        return sample_topk_sharded(
            logits_local,
            k=5,
            temperature=1.0,
            ctx=ctx,
            greedy=True
        )

    results = run_all_ranks(wrapped, logits_per_rank, ctxs)

    expected = int(np.argmax(logits))
    assert all(r == expected for r in results), (results, expected)
    print("✅ test_topk_sharded_greedy passed")


def test_topk_sharded_sampling():
    np.random.seed(42)
    world_size = 2
    vocab_size = 32
    shard = vocab_size // world_size

    logits = np.random.randn(vocab_size).astype(np.float32)
    cluster = Cluster(world_size)
    ctxs = [RankContext(r, world_size, cluster) for r in range(world_size)]

    logits_per_rank = [logits[r*shard:(r+1)*shard] for r in range(world_size)]

    def wrapped(logits_local, ctx):
        return sample_topk_sharded(
            logits_local,
            k=4,
            temperature=0.7,
            ctx=ctx,
            greedy=False
        )

    results = run_all_ranks(wrapped, logits_per_rank, ctxs)

    # All ranks must agree
    assert len(set(results)) == 1, results

    token = results[0]
    assert 0 <= token < vocab_size, token
    print("✅ test_topk_sharded_sampling passed")


if __name__ == "__main__":
    test_topk_sharded_greedy()
    test_topk_sharded_sampling()
    print("All tests passed ✅")

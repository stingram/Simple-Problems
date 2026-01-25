import numpy as np
from typing import Tuple, Dict


# =========================
# Infrastructure (DO NOT MODIFY)
# =========================
class BlockedRecv(Exception):
    """
    Raised when a rank tries to recv a message that hasn't been sent yet.
    The cooperative runner will catch this and retry later.
    """
    def __init__(self, src: int, dst: int, tag: str):
        super().__init__(f"Blocked recv waiting for (src={src}, dst={dst}, tag={tag})")
        self.src = src
        self.dst = dst
        self.tag = tag


class Cluster:
    """
    Single-process simulation of distributed message passing.
    Supports non-blocking progress via BlockedRecv + cooperative scheduling.
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
            # IMPORTANT: block instead of fail-fast
            raise BlockedRecv(src=src, dst=dst, tag=tag)
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
    local_index = np.argsort(logits_local,-1)[-k:]
    print(f'{k=},{logits_local=}')
    print(f'{local_index=}')
    logits_local_top_k = logits_local[local_index]
    
    #  2) Convert local indices -> global token ids using:
    #        vocab_shard = logits_local.shape[0]
    #        global_id = ctx.rank * vocab_shard + local_index
    vocab_shard = logits_local.shape[0]
    global_id = ctx.rank * vocab_shard + local_index
    temp_buffer = np.stack((logits_local_top_k,global_id))
    
    #  3) Send only your local candidates to rank 0 (coordinator).
    coord_rank = ctx.world_size - 1
    if ctx.rank != coord_rank:
        ctx.cluster.send(ctx.rank,coord_rank,f'cand/{ctx.rank}',temp_buffer.copy())
    #  4) Rank 0 merges candidates from all ranks, selects global top-k.
    else:
        global_buffer = np.zeros((2,k*ctx.world_size))
        for i in range(ctx.world_size):
            if i != coord_rank:
                buffer = ctx.cluster.recv(i,coord_rank,f'cand/{i}')
            else:
                buffer = temp_buffer.copy()
            print(f'{i=},{vocab_shard=},{buffer.shape=}')
            global_buffer[:,i*k:(i+1)*k] = buffer.copy()
    # print(f'{global_buffer=}')
    #  5) If greedy=True: select global argmax token.
    #     Else: apply temperature, softmax over the global top-k, sample a token.
    #     Sampling MUST be deterministic across ranks: only rank 0 samples.
    if ctx.rank == coord_rank:
        logits_ids = global_buffer[:,np.argsort(global_buffer[0])[-k:]].copy()
        if greedy:
            logit_to_send = logits_ids[1,-1]
            print(f'Greedy: {logit_to_send.copy()}')
        else:
            if temperature > 0:
                scaled_logits = logits_ids[0,:] / temperature
                max_logit = scaled_logits[-1]
                e_x = np.exp(scaled_logits - max_logit)
                probs = e_x / e_x.sum(axis=-1)
                logit_to_send = np.array(np.random.choice(len(probs),p=probs))
        for i in range(ctx.world_size-1):
            ctx.cluster.send(coord_rank,i,f"out/{i}",logit_to_send.copy())
            print(f'sent to {i}: {logit_to_send.copy()}')
    else:
        logit_to_send = ctx.cluster.recv(coord_rank,ctx.rank,f"out/{ctx.rank}") 
    print(f"{logit_to_send=}")
    return int(logit_to_send)
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
# Cooperative Test Harness (DO NOT MODIFY)
# =========================

def run_all_ranks(fn, inputs_per_rank, ctxs, max_rounds: int = 10_000):
    """
    Cooperative SPMD runner.

    Each rank runs the same code, but may temporarily block on recv() (BlockedRecv).
    The runner cycles through ranks until all finish or a deadlock is detected.
    """
    world_size = len(ctxs)
    results = [None] * world_size
    done = [False] * world_size

    for _ in range(max_rounds):
        progress = False
        for r, ctx in enumerate(ctxs):
            if done[r]:
                continue
            try:
                results[r] = fn(inputs_per_rank[r], ctx)
                done[r] = True
                progress = True
            except BlockedRecv:
                # This rank can't proceed yet; try other ranks.
                pass

        if all(done):
            return results

        if not progress:
            # No rank completed a step this round, so we are stuck
            # (e.g., circular waits / mismatched tags).
            raise RuntimeError("Deadlock detected: no rank made progress in this round.")

    raise RuntimeError("Too many rounds: likely deadlock or logic error.")


# =========================
# Unit Tests (DO NOT MODIFY)
# =========================

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
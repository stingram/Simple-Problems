import json, os, math, textwrap, random, pathlib, pandas as pd
from datetime import datetime

out_dir = os.curdir
os.makedirs(out_dir, exist_ok=True)

def write_trace(filename, events):
    path = os.path.join(out_dir, filename)
    with open(path, "w") as f:
        json.dump({"traceEvents": events, "displayTimeUnit": "ms"}, f)
    return path

def meta_events_for_ranks(nranks, cpu_tid=0, gpu_tid=1):
    ev = []
    # process names
    for r in range(nranks):
        ev.append({"name":"process_name","ph":"M","pid":r,"tid":0,"args":{"name":f"rank{r}"}})
        ev.append({"name":"thread_name","ph":"M","pid":r,"tid":cpu_tid,"args":{"name":"CPU"}})
        ev.append({"name":"thread_name","ph":"M","pid":r,"tid":gpu_tid,"args":{"name":"GPU"}})
        ev.append({"name":"thread_name","ph":"M","pid":r,"tid":2,"args":{"name":"Comm(active)"}})
        ev.append({"name":"thread_name","ph":"M","pid":r,"tid":3,"args":{"name":"Pipeline"}})
    return ev

def X(name, ts, dur, pid, tid, cat=""):
    e = {"name":name,"ph":"X","ts":ts,"dur":dur,"pid":pid,"tid":tid}
    if cat: e["cat"]=cat
    return e

def make_pattern_1(nranks=4):
    # Long collective on all ranks, but one rank arrives late. All end together.
    ev = meta_events_for_ranks(nranks)
    base = 0
    iter_dur = 200_000  # us
    ar_end = base + 180_000
    late_rank = 2
    # compute before collective
    for r in range(nranks):
        ev.append(X("compute_backward", base+20_000, 60_000, r, 1, "compute"))
    # collective bars (logical)
    for r in range(nranks):
        start = base+90_000
        if r == late_rank:
            start = base+150_000  # late due to CPU compile
        ev.append(X("all_reduce(logical)", start, ar_end-start, r, 1, "collective"))
        # show the "active transfer" window aligned to late start + expected duration
        active_start = base+150_000  # when last arrives
        active_dur = 25_000          # expected transfer
        ev.append(X("all_reduce(active)", active_start, active_dur, r, 2, "collective"))
    # CPU compilation on late rank
    ev.append(X("CPU: compile / graph specialize", base+95_000, 55_000, late_rank, 0, "cpu"))
    # next iter marker
    for r in range(nranks):
        ev.append(X("iter_next", base+180_000, 10_000, r, 0, "marker"))
    return ev

def make_pattern_2(nranks=4):
    # Long collective, staggered end times (contention/topology)
    ev = meta_events_for_ranks(nranks)
    base = 0
    start = 100_000
    durs = [35_000, 55_000, 80_000, 50_000]  # different
    for r in range(nranks):
        ev.append(X("compute_backward", 20_000, 70_000, r, 1, "compute"))
        ev.append(X("all_reduce(active)", start, durs[r], r, 2, "collective"))
        ev.append(X("all_reduce(logical)", start, durs[r], r, 1, "collective"))
    return ev

def make_pattern_3(nranks=4):
    # Many tiny collectives dominate
    ev = meta_events_for_ranks(nranks)
    base = 0
    ts = 20_000
    for step in range(30):
        for r in range(nranks):
            ev.append(X("tiny_all_gather", ts, 800, r, 1, "collective"))
        ts += 1_200  # 0.4ms gap
        for r in range(nranks):
            ev.append(X("small_gemm", ts, 600, r, 1, "compute"))
        ts += 900
    return ev

def make_pattern_4(nranks=1):
    # Kernel duration scales with size (compute-bound GEMMs)
    ev = meta_events_for_ranks(nranks)
    base = 0
    ts = 10_000
    sizes = [("GEMM M=256", 1_000), ("GEMM M=512", 2_000), ("GEMM M=1024", 4_000), ("GEMM M=2048", 8_000)]
    for name, dur in sizes:
        ev.append(X(name, ts, dur, 0, 1, "compute"))
        ts += dur + 1_000
    return ev

def make_pattern_5(nranks=1):
    # Kernel looks long but math says it shouldn't: launch/dependency stall
    ev = meta_events_for_ranks(nranks)
    base = 0
    # small GEMM expected 200us but observed "kernel" 2000us because waiting on event
    ev.append(X("GPU idle (waiting)", 10_000, 1_500, 0, 1, "idle"))
    ev.append(X("small_gemm", 11_500, 200, 0, 1, "compute"))
    # But framework labels a larger op spanning both
    ev.append(X("matmul_op (framework span)", 10_000, 1_700, 0, 0, "framework"))
    return ev

def make_pattern_6(nranks=1):
    # Many small kernels with whitespace -> runtime/orchestration bound
    ev = meta_events_for_ranks(nranks)
    ts = 10_000
    for i in range(20):
        ev.append(X("CPU: dispatch / scheduling", ts, 600, 0, 0, "cpu"))
        ev.append(X("tiny_kernel", ts+650, 150, 0, 1, "compute"))
        ts += 1_500  # big gap
    return ev

def make_pattern_7(nranks=1):
    # Attention time grows with seq_len (KV bandwidth bound)
    ev = meta_events_for_ranks(nranks)
    ts = 10_000
    seqs = [256, 512, 1024, 2048]
    for s in seqs:
        dur = int(200 + s*2.0)  # linear
        ev.append(X(f"attn_kv_read seq={s}", ts, dur, 0, 1, "memory"))
        ts += dur + 500
        ev.append(X("mlp_gemm (constant)", ts, 600, 0, 1, "compute"))
        ts += 1_000
    return ev

def make_pattern_8(nranks=1):
    # Memory ops serialize compute; no overlap
    ev = meta_events_for_ranks(nranks)
    ts = 10_000
    for i in range(6):
        ev.append(X("KV cache load", ts, 1_200, 0, 1, "memory"))
        ts += 1_250
        ev.append(X("attention_compute", ts, 900, 0, 1, "compute"))
        ts += 950
    # show hypothetical desired overlap on another tid (for contrast)
    ts2 = 10_000
    for i in range(6):
        ev.append(X("desired_overlap: attn_compute", ts2+400, 900, 0, 2, "compute"))
        ts2 += 2_200
    return ev

def make_pattern_9(nranks=4):
    # Pipeline bubbles across stages
    ev = meta_events_for_ranks(nranks)
    # Use pid as stage as well; tid=3 as pipeline lane
    base = 0
    microbatches = 4
    stage_compute = [40_000, 55_000, 45_000, 50_000]  # stage imbalance
    gap = 5_000
    # schedule: simple pipeline fill/drain
    for stage in range(nranks):
        t = base + stage*15_000
        for mb in range(microbatches):
            ev.append(X(f"stage{stage}: microbatch{mb}", t, stage_compute[stage], stage, 3, "pipeline"))
            t += stage_compute[stage] + gap
    # show idle bubbles as explicit events (optional)
    for stage in range(nranks):
        ev.append(X("bubble/idle", base, stage*15_000, stage, 3, "idle"))
    return ev

def make_pattern_10(nranks=4):
    # One rank consistently late (tail), gating step time
    ev = meta_events_for_ranks(nranks)
    base = 0
    for it in range(3):
        it_base = base + it*200_000
        tail = 3
        for r in range(nranks):
            dur = 80_000
            if r == tail:
                dur = 110_000  # slower compute
            ev.append(X("compute_step", it_base+20_000, dur, r, 1, "compute"))
            # barrier at end
            barrier_end = it_base + 150_000
            start = it_base + 20_000 + dur
            if start > barrier_end:
                barrier_end = start + 20_000
            ev.append(X("barrier_wait", start, barrier_end-start, r, 1, "sync"))
        # mark next iter start at barrier_end
        for r in range(nranks):
            ev.append(X("iter_next", it_base+170_000, 10_000, r, 0, "marker"))
    return ev

pattern_makers = [
    ("01_collective_waiting_last_rank.json", make_pattern_1),
    ("02_collective_staggered_ends.json", make_pattern_2),
    ("03_many_tiny_collectives.json", make_pattern_3),
    ("04_compute_scales_with_size.json", make_pattern_4),
    ("05_fake_long_compute_due_to_wait.json", make_pattern_5),
    ("06_runtime_orchestration_whitespace.json", make_pattern_6),
    ("07_kv_bandwidth_scales_with_seqlen.json", make_pattern_7),
    ("08_memory_serializes_compute.json", make_pattern_8),
    ("09_pipeline_bubbles.json", make_pattern_9),
    ("10_tail_rank_gates_progress.json", make_pattern_10),
]

paths = []
for fname, fn in pattern_makers:
    events = fn()
    paths.append(write_trace(fname, events))

# write a small README
readme = """\
How to view these traces
------------------------
1) Open https://ui.perfetto.dev in Chrome (recommended) OR open chrome://tracing
2) Load any .json file from this folder.
3) Use WASD / mousewheel to zoom and pan (Perfetto has on-screen help).
4) Each 'rank' is a separate process. Threads are labeled:
   - CPU
   - GPU
   - Comm(active) (often shows the "true active transfer window" to contrast waiting)
   - Pipeline (for pipeline bubble demo)

What each file demonstrates
---------------------------
01_collective_waiting_last_rank.json
  Long all-reduce bars on most ranks are mostly WAITING; true active transfer starts when last rank arrives.

02_collective_staggered_ends.json
  All-reduce durations differ across ranks -> contention/topology/imbalance.

03_many_tiny_collectives.json
  Many small collectives dominate timeline -> latency/launch overhead.

04_compute_scales_with_size.json
  GEMM time scales with problem size -> compute-bound.

05_fake_long_compute_due_to_wait.json
  Framework span looks long, but GPU does little compute; most time is waiting/dependency.

06_runtime_orchestration_whitespace.json
  Big whitespace between tiny kernels -> CPU/runtime orchestration bottleneck.

07_kv_bandwidth_scales_with_seqlen.json
  Attention/KV load time grows with seq_len (linear) -> memory bandwidth bound.

08_memory_serializes_compute.json
  KV loads serialize attention compute; 'desired_overlap' track shows what overlap would look like.

09_pipeline_bubbles.json
  Pipeline stages show bubbles/idle (plus imbalance).

10_tail_rank_gates_progress.json
  One rank (tail) consistently slower; others wait at barrier -> tail-latency gates throughput.
"""
readme_path = os.path.join(out_dir, "README.txt")
with open(readme_path, "w") as f:
    f.write(readme)

#!/usr/bin/env python3
import json, os

OUT_DIR = "trace_patterns_fixed"
os.makedirs(OUT_DIR, exist_ok=True)

def write_trace(filename, events):
    path = os.path.join(OUT_DIR, filename)
    with open(path, "w") as f:
        json.dump({"traceEvents": events, "displayTimeUnit": "ms"}, f)
    return path

def meta_events_for_ranks(nranks, cpu_tid=0, gpu_tid=1):
    ev = []
    for r in range(nranks):
        ev.append({"name":"process_name","ph":"M","pid":r,"tid":0,"args":{"name":f"rank{r}"}})
        ev.append({"name":"thread_name","ph":"M","pid":r,"tid":cpu_tid,"args":{"name":"CPU"}})
        ev.append({"name":"thread_name","ph":"M","pid":r,"tid":gpu_tid,"args":{"name":"GPU"}})
        ev.append({"name":"thread_name","ph":"M","pid":r,"tid":2,"args":{"name":"Comm(active)"}})
        ev.append({"name":"thread_name","ph":"M","pid":r,"tid":3,"args":{"name":"Pipeline"}})
    return ev

def X(name, ts, dur, pid, tid, cat=""):
    e = {"name":"op", "ph":"X", "ts":ts, "dur":dur, "pid":pid, "tid":tid}
    e["name"] = name
    if cat:
        e["cat"] = cat
    return e

def fix_01(nranks=4):
    ev = meta_events_for_ranks(nranks)
    base = 0
    ar_start = base + 150_000
    ar_dur = 25_000
    for r in range(nranks):
        ev.append(X("CPU: warmup compile / cache hit", base + 5_000, 15_000, r, 0, "cpu"))
    for r in range(nranks):
        ev.append(X("compute_backward", base + 20_000, 120_000, r, 1, "compute"))
        ev.append(X("all_reduce(logical)", ar_start, ar_dur, r, 1, "collective"))
        ev.append(X("all_reduce(active)", ar_start, ar_dur, r, 2, "collective"))
        ev.append(X("iter_next", ar_start + ar_dur, 10_000, r, 0, "marker"))
    return ev

def fix_02(nranks=4):
    ev = meta_events_for_ranks(nranks)
    start = 100_000
    dur = 40_000
    for r in range(nranks):
        ev.append(X("compute_backward", 20_000, 70_000, r, 1, "compute"))
        ev.append(X("all_reduce(active)", start, dur, r, 2, "collective"))
        ev.append(X("all_reduce(logical)", start, dur, r, 1, "collective"))
    return ev

def fix_03(nranks=4):
    ev = meta_events_for_ranks(nranks)
    ts = 20_000
    for _ in range(6):
        for r in range(nranks):
            ev.append(X("fused_all_gather", ts, 4_000, r, 1, "collective"))
            ev.append(X("fused_all_gather(active)", ts, 4_000, r, 2, "collective"))
        for r in range(nranks):
            ev.append(X("gemm_overlapped", ts + 1_000, 5_000, r, 1, "compute"))
        ts += 10_000
    return ev

def fix_04(nranks=1):
    ev = meta_events_for_ranks(nranks)
    ts = 10_000
    sizes = [
        ("GEMM M=256 (opt)", 700),
        ("GEMM M=512 (opt)", 1_300),
        ("GEMM M=1024 (opt)", 2_500),
        ("GEMM M=2048 (opt)", 5_000),
    ]
    for name, dur in sizes:
        ev.append(X(name, ts, dur, 0, 1, "compute"))
        ts += dur + 1_000
    return ev

def fix_05(nranks=1):
    ev = meta_events_for_ranks(nranks)
    ev.append(X("small_gemm", 10_000, 220, 0, 1, "compute"))
    ev.append(X("matmul_op (framework span)", 10_000, 240, 0, 0, "framework"))
    return ev

def fix_06(nranks=1):
    ev = meta_events_for_ranks(nranks)
    ts = 10_000
    ev.append(X("CPU: one-time setup", ts, 1_000, 0, 0, "cpu"))
    ts += 1_200
    for _ in range(25):
        ev.append(X("graph_replay_step", ts, 50, 0, 0, "cpu"))
        ev.append(X("fused_kernel", ts, 600, 0, 1, "compute"))
        ts += 700
    return ev

def fix_07(nranks=1):
    ev = meta_events_for_ranks(nranks)
    ts = 10_000
    for s in [256, 512, 1024, 2048]:
        dur = int(180 + s * 1.1)
        ev.append(X(f"attn_kv_read seq={s} (opt)", ts, dur, 0, 1, "memory"))
        ts += dur + 500
        ev.append(X("mlp_gemm (constant)", ts, 600, 0, 1, "compute"))
        ts += 1_000
    return ev

def fix_08(nranks=1):
    ev = meta_events_for_ranks(nranks)
    ts = 10_000
    for _ in range(6):
        ev.append(X("KV prefetch (async)", ts, 1_200, 0, 2, "memory"))
        ev.append(X("attention_compute", ts + 300, 900, 0, 1, "compute"))
        ts += 1_400
    return ev

def fix_09(nranks=4):
    ev = meta_events_for_ranks(nranks)
    base = 0
    microbatches = 8
    stage_compute = [45_000] * 4
    gap = 2_000
    for stage in range(nranks):
        t = base + stage * 8_000
        for mb in range(microbatches):
            ev.append(X(f"stage{stage}: microbatch{mb}", t, stage_compute[stage], stage, 3, "pipeline"))
            t += stage_compute[stage] + gap
        ev.append(X("bubble/idle (reduced)", base, stage * 8_000, stage, 3, "idle"))
    return ev

def fix_10(nranks=4):
    ev = meta_events_for_ranks(nranks)
    base = 0
    for it in range(3):
        it_base = base + it * 180_000
        for r in range(nranks):
            ev.append(X("compute_step", it_base + 20_000, 90_000, r, 1, "compute"))
            ev.append(X("barrier_wait", it_base + 110_000, 10_000, r, 1, "sync"))
        for r in range(nranks):
            ev.append(X("iter_next", it_base + 125_000, 10_000, r, 0, "marker"))
    return ev

MAKERS = [
    ("01_fixed_collective_no_wait.json", fix_01),
    ("02_fixed_collective_uniform.json", fix_02),
    ("03_fixed_fused_collectives.json", fix_03),
    ("04_fixed_more_efficient_gemm.json", fix_04),
    ("05_fixed_no_dependency_wait.json", fix_05),
    ("06_fixed_cudagraph_replay_dense.json", fix_06),
    ("07_fixed_kv_cache_optimized.json", fix_07),
    ("08_fixed_overlap_memory_compute.json", fix_08),
    ("09_fixed_pipeline_reduced_bubbles.json", fix_09),
    ("10_fixed_tail_rank_removed.json", fix_10),
]

def main():
    paths = []
    for fname, fn in MAKERS:
        paths.append(write_trace(fname, fn()))
    print("Wrote traces:")
    for p in paths:
        print("  ", p)

if __name__ == "__main__":
    main()

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

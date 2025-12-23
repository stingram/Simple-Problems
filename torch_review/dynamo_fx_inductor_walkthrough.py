import os
import dis
import time
from pathlib import Path

import torch
from torch.fx.passes.graph_drawer import FxGraphDrawer


# ---------------------------------------
# Logging knobs (can also be set in shell)
# ---------------------------------------
os.environ.setdefault(
    "TORCH_LOGS",
    "dynamo,graph_breaks,guards,recompiles,inductor"
)
os.environ.setdefault("TORCHDYNAMO_VERBOSE", "1")
os.environ.setdefault("TORCH_COMPILE_DEBUG", "1")

import importlib
dynamo = importlib.import_module("torch._dynamo")


# ---------------------------------------
# Phase 1: Pure Python
# ---------------------------------------
def py_only(xs):
    s = 0.0
    for x in xs:
        if x > 0:
            s += x * x
        else:
            s += (-x)
    return s


# ---------------------------------------
# Phase 2: Torch eager
# ---------------------------------------
def eager_torch_with_graph_break(x, w):
    y = torch.relu(x @ w)

    # Python control flow based on tensor value → graph break
    if y.sum().item() > 0:
        y = y + 1

    return y.sum()


def graphable_torch(x, w):
    return torch.relu(x @ w).sum()


# ---------------------------------------
# Helpers
# ---------------------------------------
def show_bytecode(fn):
    print("\n=== Python bytecode ===")
    dis.dis(fn)


@torch.no_grad()
def run_and_time(label, fn, *args, iters=2):
    print(f"\n--- {label} ---")
    for i in range(iters):
        torch.cuda.synchronize()
        t0 = time.time()
        out = fn(*args)
        torch.cuda.synchronize()
        print(f"iter {i}: out={out.item():.4f}, time_ms={(time.time() - t0)*1000:.2f}")


def export_fx(fn, *example_args, name="fx_graph"):
    # gm, guards = torch._dynamo.export(fn)(*example_args)
    gm, guards = dynamo.export(fn, *example_args)

    print("\n=== FX Graph ===\n")
    print(gm.code)
    print("\n=== Guards ===\n")
    print(guards)

    out_dir = Path("fx_graphs")
    out_dir.mkdir(exist_ok=True)

    drawer = FxGraphDrawer(gm, name)
    svg_path = out_dir / f"{name}.svg"
    drawer.get_dot_graph().write_svg(str(svg_path))

    print(f"\n[FX graph written to {svg_path}]\n")


# ---------------------------------------
# Main
# ---------------------------------------
def main():
    device = "cuda"
    print("Using device:", torch.cuda.get_device_name(0))

    x = torch.randn(1024, 1024, device=device)
    w = torch.randn(1024, 1024, device=device)

    print("\nPHASE 1: Pure Python")
    show_bytecode(py_only)
    print("py_only result:", py_only([1.0, -2.0, 3.0]))

    print("\nPHASE 2: Torch eager")
    show_bytecode(graphable_torch)
    run_and_time("eager", graphable_torch, x, w)

    print("\nPHASE 3: torch.compile (graphable)")
    export_fx(graphable_torch, x, w, name="graphable")

    compiled = torch.compile(graphable_torch)
    run_and_time("compiled first/second", compiled, x, w)

    print("\nTrigger guard failure (shape change)")
    x2 = torch.randn(512, 1024, device=device)
    run_and_time("compiled new shape", compiled, x2, w)

    print("\nPHASE 4: Graph breaks (.item())")
    export_fx(eager_torch_with_graph_break, x, w, name="graph_breaks")

    compiled_breaky = torch.compile(eager_torch_with_graph_break)
    run_and_time("compiled with graph breaks", compiled_breaky, x, w)


if __name__ == "__main__":
    main()

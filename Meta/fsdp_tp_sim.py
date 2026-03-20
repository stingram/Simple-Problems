
"""
fsdp_tp_sim.py

A single-process PyTorch simulator for understanding a 2D TP + FSDP layout.

Important note:
----------------
This is a pedagogical simulator, not a real distributed implementation.
To make the geometry easy to inspect, it uses:

  - TP (tensor parallel) = split each linear layer along output columns
  - FSDP (storage sharding) = split each TP slice along input rows

Real FSDP in PyTorch usually shards a flattened parameter buffer, so the exact
physical layout is different. But the simulator captures the core idea:

  TP  => who computes which output slice
  FSDP => who persistently stores which bytes of that TP slice

Forward:
    sharded storage -> all-gather TP slice -> TP compute -> free gathered slice

Backward:
    sharded storage -> all-gather TP slice -> compute dW + partial dX
    -> TP reduce over partial dX -> FSDP reduce-scatter dW -> local optimizer shard

Defaults are intentionally tiny.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


# -----------------------------
# Configuration
# -----------------------------
@dataclass
class SimConfig:
    batch_size: int = 1
    seq_len: int = 2
    num_heads: int = 2
    head_dim: int = 2
    num_gpus: int = 16
    tp: int = 4
    fsdp: int = 4
    mlp_dim: int = 8
    seed: int = 0
    print_values: bool = True
    plot: bool = True
    dtype: torch.dtype = torch.float32

    @property
    def d_model(self) -> int:
        return self.num_heads * self.head_dim

    def validate(self) -> None:
        assert self.num_gpus == self.tp * self.fsdp, (
            f"Expected num_gpus == tp * fsdp for this 2D simulator, "
            f"got {self.num_gpus=}, {self.tp=}, {self.fsdp=}"
        )
        assert self.d_model % self.tp == 0, "d_model must be divisible by tp"
        assert self.d_model % self.fsdp == 0, "d_model must be divisible by fsdp"
        assert self.mlp_dim % self.tp == 0, "mlp_dim must be divisible by tp"
        assert self.mlp_dim % self.fsdp == 0, (
            "For this simple simulator, mlp_dim should also be divisible by fsdp "
            "if you want to inspect row-wise FSDP sharding cleanly."
        )


# -----------------------------
# Utility / pretty-print helpers
# -----------------------------
def make_toy_tensor(shape: Tuple[int, ...], start: int = 1, dtype=torch.float32) -> torch.Tensor:
    n = 1
    for s in shape:
        n *= s
    t = torch.arange(start, start + n, dtype=dtype)
    return t.reshape(shape)


def print_header(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def print_tensor(name: str, x: torch.Tensor, max_elements: int = 128) -> None:
    flat = x.reshape(-1)
    if flat.numel() > max_elements:
        print(f"{name}: shape={tuple(x.shape)}, dtype={x.dtype}, values skipped ({flat.numel()} elements)")
    else:
        print(f"{name}: shape={tuple(x.shape)}")
        print(x)


def gpu_id(fsdp_row: int, tp_col: int, tp: int) -> int:
    return fsdp_row * tp + tp_col


def split_columns(W: torch.Tensor, tp: int) -> List[torch.Tensor]:
    # Column-parallel split for TP
    return list(torch.chunk(W, tp, dim=1))


def split_rows(W_col: torch.Tensor, fsdp: int) -> List[torch.Tensor]:
    # Simple educational FSDP split: split each TP slice along input rows
    return list(torch.chunk(W_col, fsdp, dim=0))


def concatenate_rows(row_shards: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat(row_shards, dim=0)


def format_layer_grid_labels(layer_name: str, row_shards: Dict[int, Dict[int, torch.Tensor]], tp: int, fsdp: int) -> List[List[str]]:
    labels: List[List[str]] = []
    for fr in range(fsdp):
        row: List[str] = []
        for tc in range(tp):
            gid = gpu_id(fr, tc, tp)
            shard = row_shards[fr][tc]
            row.append(f"GPU{gid}\n{layer_name}\nshape={tuple(shard.shape)}")
        labels.append(row)
    return labels


def plot_gpu_grid(title: str, labels: List[List[str]]) -> None:
    rows = len(labels)
    cols = len(labels[0])
    fig, ax = plt.subplots(figsize=(1.9 * cols, 1.5 * rows))
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.invert_yaxis()
    ax.set_xticks([i + 0.5 for i in range(cols)], labels=[f"TP col {i}" for i in range(cols)])
    ax.set_yticks([i + 0.5 for i in range(rows)], labels=[f"FSDP row {i}" for i in range(rows)])
    ax.set_title(title)

    for r in range(rows):
        for c in range(cols):
            rect = plt.Rectangle((c, r), 1, 1, fill=False)
            ax.add_patch(rect)
            ax.text(c + 0.5, r + 0.5, labels[r][c], ha="center", va="center", fontsize=9)

    plt.tight_layout()
    plt.show()


# -----------------------------
# 2D TP + FSDP sharded linear layer simulator
# -----------------------------
class ShardedLinear2D:
    def __init__(self, name: str, W_full: torch.Tensor, cfg: SimConfig):
        """
        W_full shape: [d_in, d_out]
        TP split: output columns
        FSDP split: rows of each TP slice
        """
        self.name = name
        self.cfg = cfg
        self.W_full = W_full.to(cfg.dtype)

        d_in, d_out = W_full.shape
        assert d_out % cfg.tp == 0, f"{name}: output dim must be divisible by tp"
        assert d_in % cfg.fsdp == 0, f"{name}: input dim must be divisible by fsdp"

        self.tp_slices = split_columns(self.W_full, cfg.tp)

        # persistent storage: row_shards[fsdp_row][tp_col]
        self.row_shards: Dict[int, Dict[int, torch.Tensor]] = {fr: {} for fr in range(cfg.fsdp)}
        for tc, col_slice in enumerate(self.tp_slices):
            shards = split_rows(col_slice, cfg.fsdp)
            for fr, shard in enumerate(shards):
                self.row_shards[fr][tc] = shard.contiguous()

    def describe_permanent_layout(self) -> None:
        print_header(f"{self.name}: permanent / persistent placement")
        for fr in range(self.cfg.fsdp):
            for tc in range(self.cfg.tp):
                gid = gpu_id(fr, tc, self.cfg.tp)
                shard = self.row_shards[fr][tc]
                print(f"GPU{gid:02d} stores shard with shape {tuple(shard.shape)}")
                if self.cfg.print_values:
                    print(shard)

        if self.cfg.plot:
            labels = format_layer_grid_labels(self.name, self.row_shards, self.cfg.tp, self.cfg.fsdp)
            plot_gpu_grid(f"{self.name}: permanent placement", labels)

    def all_gather_tp_slices(self) -> Dict[int, torch.Tensor]:
        """
        Gather across FSDP rows for each TP column.
        Returns:
            gathered[tp_col] = full TP slice with shape [d_in, d_out/tp]
        """
        print_header(f"{self.name}: FSDP all-gather of TP slices")
        gathered: Dict[int, torch.Tensor] = {}
        for tc in range(self.cfg.tp):
            parts = [self.row_shards[fr][tc] for fr in range(self.cfg.fsdp)]
            full_slice = concatenate_rows(parts).contiguous()
            gathered[tc] = full_slice
            print(f"TP column {tc}: gathered full slice with shape {tuple(full_slice.shape)}")
            if self.cfg.print_values:
                print(full_slice)

        return gathered

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """
        X shape: [N, d_in]
        Returns:
            Y_full: [N, d_out]
            cache: gathered TP slices for pedagogical backward
        """
        gathered = self.all_gather_tp_slices()

        print_header(f"{self.name}: TP forward compute")
        Y_parts = []
        for tc in range(self.cfg.tp):
            W_tc = gathered[tc]
            Y_tc = X @ W_tc
            Y_parts.append(Y_tc)
            print(f"TP column {tc}: X @ W_{tc} -> shape {tuple(Y_tc.shape)}")
            if self.cfg.print_values:
                print(Y_tc)

        Y_full = torch.cat(Y_parts, dim=1).contiguous()
        print_tensor(f"{self.name}: concatenated TP output", Y_full)

        print_header(f"{self.name}: free gathered TP slices (return to shard-only storage)")
        print("In a real FSDP system, the temporary gathered parameters would now be freed.")
        return Y_full, gathered

    def backward_manual(self, X: torch.Tensor, dY: torch.Tensor, gathered: Optional[Dict[int, torch.Tensor]] = None) -> Tuple[torch.Tensor, Dict[int, Dict[int, torch.Tensor]]]:
        """
        Manual backward for y = XW.
        X shape:  [N, d_in]
        dY shape: [N, d_out]

        Returns:
            dX: [N, d_in]
            grad_row_shards[fsdp_row][tp_col]: shard of dW matching persistent storage
        """
        if gathered is None:
            gathered = self.all_gather_tp_slices()

        print_header(f"{self.name}: manual backward")
        dY_parts = list(torch.chunk(dY, self.cfg.tp, dim=1))

        full_dW_per_tp: Dict[int, torch.Tensor] = {}
        partial_dX_per_tp: Dict[int, torch.Tensor] = {}

        for tc in range(self.cfg.tp):
            W_tc = gathered[tc]
            dY_tc = dY_parts[tc]

            # dW_tc = X^T @ dY_tc
            dW_tc = X.transpose(0, 1) @ dY_tc
            full_dW_per_tp[tc] = dW_tc

            # partial dX from this TP slice
            partial_dX = dY_tc @ W_tc.transpose(0, 1)
            partial_dX_per_tp[tc] = partial_dX

            print(f"TP column {tc}: dW_{tc} shape {tuple(dW_tc.shape)}")
            if self.cfg.print_values:
                print(dW_tc)
            print(f"TP column {tc}: partial dX_{tc} shape {tuple(partial_dX.shape)}")
            if self.cfg.print_values:
                print(partial_dX)

        # TP reduction for dX
        dX = sum(partial_dX_per_tp.values())
        print_tensor(f"{self.name}: TP-reduced dX", dX)

        # FSDP reduce-scatter for dW
        print_header(f"{self.name}: FSDP reduce-scatter of dW back to persistent shard owners")
        grad_row_shards: Dict[int, Dict[int, torch.Tensor]] = {fr: {} for fr in range(self.cfg.fsdp)}
        for tc in range(self.cfg.tp):
            dW_tc = full_dW_per_tp[tc]
            row_chunks = split_rows(dW_tc, self.cfg.fsdp)
            for fr in range(self.cfg.fsdp):
                grad_row_shards[fr][tc] = row_chunks[fr].contiguous()
                gid = gpu_id(fr, tc, self.cfg.tp)
                print(f"GPU{gid:02d} receives grad shard for TP col {tc}, FSDP row {fr}, shape={tuple(row_chunks[fr].shape)}")
                if self.cfg.print_values:
                    print(row_chunks[fr])

        return dX, grad_row_shards

    def reconstruct_from_shards(self) -> torch.Tensor:
        cols = []
        for tc in range(self.cfg.tp):
            col = concatenate_rows([self.row_shards[fr][tc] for fr in range(self.cfg.fsdp)])
            cols.append(col)
        return torch.cat(cols, dim=1)

    def apply_sgd_step(self, grad_row_shards: Dict[int, Dict[int, torch.Tensor]], lr: float = 1e-2) -> None:
        print_header(f"{self.name}: local shard-wise SGD step")
        for fr in range(self.cfg.fsdp):
            for tc in range(self.cfg.tp):
                gid = gpu_id(fr, tc, self.cfg.tp)
                self.row_shards[fr][tc] = self.row_shards[fr][tc] - lr * grad_row_shards[fr][tc]
                print(f"GPU{gid:02d} updates its local shard in place.")

        self.W_full = self.reconstruct_from_shards()


# -----------------------------
# Tiny transformer block using the sharded linear simulator
# -----------------------------
class TinyTransformer2DSim:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        cfg.validate()
        torch.manual_seed(cfg.seed)

        d_model = cfg.d_model
        mlp_dim = cfg.mlp_dim

        # Small deterministic weights
        self.Wq = ShardedLinear2D("Wq", make_toy_tensor((d_model, d_model), start=1, dtype=cfg.dtype), cfg)
        self.Wk = ShardedLinear2D("Wk", make_toy_tensor((d_model, d_model), start=101, dtype=cfg.dtype), cfg)
        self.Wv = ShardedLinear2D("Wv", make_toy_tensor((d_model, d_model), start=201, dtype=cfg.dtype), cfg)
        self.Wo = ShardedLinear2D("Wo", make_toy_tensor((d_model, d_model), start=301, dtype=cfg.dtype), cfg)
        self.Wup = ShardedLinear2D("Wup", make_toy_tensor((d_model, mlp_dim), start=401, dtype=cfg.dtype), cfg)
        self.Wdown = ShardedLinear2D("Wdown", make_toy_tensor((mlp_dim, d_model), start=501, dtype=cfg.dtype), cfg)

    def make_input(self) -> torch.Tensor:
        # [B, S, d_model]
        X = make_toy_tensor(
            (self.cfg.batch_size, self.cfg.seq_len, self.cfg.d_model),
            start=1,
            dtype=self.cfg.dtype,
        )
        return X

    def show_permanent_layouts(self) -> None:
        for layer in [self.Wq, self.Wk, self.Wv, self.Wo, self.Wup, self.Wdown]:
            layer.describe_permanent_layout()

    def forward(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, S, D = X.shape
        N = B * S
        X2 = X.reshape(N, D)

        print_header("INPUT")
        print_tensor("X", X)
        print_tensor("X_flat", X2)

        Q, cache_q = self.Wq.forward(X2)
        K, cache_k = self.Wk.forward(X2)
        V, cache_v = self.Wv.forward(X2)

        # reshape to heads
        H = self.cfg.num_heads
        Dh = self.cfg.head_dim
        Qh = Q.reshape(B, S, H, Dh).transpose(1, 2)  # [B, H, S, Dh]
        Kh = K.reshape(B, S, H, Dh).transpose(1, 2)
        Vh = V.reshape(B, S, H, Dh).transpose(1, 2)

        print_header("ATTENTION: reshape into heads")
        print_tensor("Qh", Qh)
        print_tensor("Kh", Kh)
        print_tensor("Vh", Vh)

        scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / math.sqrt(Dh)  # [B,H,S,S]
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, Vh)  # [B,H,S,Dh]

        print_header("ATTENTION internals")
        print_tensor("scores", scores)
        print_tensor("attn", attn)
        print_tensor("context", context)

        context2 = context.transpose(1, 2).reshape(N, D)
        O, cache_o = self.Wo.forward(context2)

        U, cache_up = self.Wup.forward(O)
        G = F.gelu(U)
        Y, cache_down = self.Wdown.forward(G)

        return {
            "X": X,
            "X2": X2,
            "Q": Q,
            "K": K,
            "V": V,
            "Qh": Qh,
            "Kh": Kh,
            "Vh": Vh,
            "scores": scores,
            "attn": attn,
            "context": context,
            "context2": context2,
            "O": O,
            "U": U,
            "G": G,
            "Y": Y,
            "cache_q": cache_q,
            "cache_k": cache_k,
            "cache_v": cache_v,
            "cache_o": cache_o,
            "cache_up": cache_up,
            "cache_down": cache_down,
        }

    def demo_manual_backward_on_mlp_only(self, forward_out: Dict[str, torch.Tensor]) -> None:
        """
        Full manual backward through attention is possible but verbose.
        To keep the simulator readable, we do a manual backward demo for the
        final two linear layers (Wup, Wdown), where TP+FSDP behavior is the same.
        """
        print_header("MANUAL BACKWARD DEMO (MLP only)")
        O = forward_out["O"]
        U = forward_out["U"]
        G = forward_out["G"]
        Y = forward_out["Y"]

        # Toy upstream gradient for output Y
        dY = torch.ones_like(Y)
        print_tensor("dY", dY)

        # Backward through Wdown: Y = G @ Wdown
        dG, grad_down = self.Wdown.backward_manual(G, dY, gathered=forward_out["cache_down"])

        # Backward through GELU (autograd used only for local scalar derivative convenience)
        G_req = U.detach().clone().requires_grad_(True)
        Y_gelu = F.gelu(G_req)
        dU = torch.autograd.grad(Y_gelu, G_req, grad_outputs=dG)[0]
        print_tensor("dU = dG * GELU'(U)", dU)

        # Backward through Wup: U = O @ Wup
        dO, grad_up = self.Wup.backward_manual(O, dU, gathered=forward_out["cache_up"])

        print_tensor("dO", dO)

        # Optional local SGD step demo
        self.Wdown.apply_sgd_step(grad_down, lr=1e-4)
        self.Wup.apply_sgd_step(grad_up, lr=1e-4)

    def reference_dense_forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Dense reference using reconstructed weights to sanity-check the forward.
        """
        B, S, D = X.shape
        N = B * S
        X2 = X.reshape(N, D)

        Wq = self.Wq.reconstruct_from_shards()
        Wk = self.Wk.reconstruct_from_shards()
        Wv = self.Wv.reconstruct_from_shards()
        Wo = self.Wo.reconstruct_from_shards()
        Wup = self.Wup.reconstruct_from_shards()
        Wdown = self.Wdown.reconstruct_from_shards()

        Q = X2 @ Wq
        K = X2 @ Wk
        V = X2 @ Wv

        H = self.cfg.num_heads
        Dh = self.cfg.head_dim
        Qh = Q.reshape(B, S, H, Dh).transpose(1, 2)
        Kh = K.reshape(B, S, H, Dh).transpose(1, 2)
        Vh = V.reshape(B, S, H, Dh).transpose(1, 2)

        scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / math.sqrt(Dh)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, Vh)
        context2 = context.transpose(1, 2).reshape(N, D)

        O = context2 @ Wo
        U = O @ Wup
        G = F.gelu(U)
        Y = G @ Wdown
        return Y

    def run(self) -> None:
        X = self.make_input()

        if self.cfg.plot:
            self.show_permanent_layouts()

        out = self.forward(X)
        Y_sim = out["Y"]
        Y_ref = self.reference_dense_forward(X)

        print_header("REFERENCE CHECK")
        print_tensor("Y_sim", Y_sim)
        print_tensor("Y_ref", Y_ref)
        print(f"max|Y_sim - Y_ref| = {(Y_sim - Y_ref).abs().max().item():.6f}")

        self.demo_manual_backward_on_mlp_only(out)


def main() -> None:
    cfg = SimConfig(
        batch_size=1,
        seq_len=2,
        num_heads=2,
        head_dim=2,
        num_gpus=16,
        tp=4,
        fsdp=4,
        mlp_dim=8,
        seed=0,
        print_values=True,
        plot=True,
    )
    sim = TinyTransformer2DSim(cfg)
    sim.run()


if __name__ == "__main__":
    main()

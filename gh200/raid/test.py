#!/usr/bin/env python3
"""
RS erasure coding benchmark on the weights of a single transformer layer
shaped like a 7B-parameter model (e.g. Llama-2-7B).

No real model is loaded — tensors are randomly initialised on CUDA.

Architecture defaults (override with CLI flags):
  hidden_size    = 4096
  intermediate   = 11008
  num_heads      = 32
  head_dim       = 128
  dtype          = float16

Weight tensors per layer:
  q/k/v_proj   : [hidden, num_heads * head_dim]
  o_proj       : [num_heads * head_dim, hidden]
  gate/up_proj : [intermediate, hidden]
  down_proj    : [hidden, intermediate]
  ln_input/post: [hidden]
"""

import os, sys, random, argparse
import numpy as np
import torch
from torch.utils.cpp_extension import load

os.environ.setdefault("CC",  "g++")
os.environ.setdefault("CXX", "g++")


# ── Helpers ───────────────────────────────────────────────────────────────
def tensor_memory_size(t: torch.Tensor) -> float:
    return t.numel() * t.element_size() / (1024 ** 3)

def set_seed(seed: int = 0):
    np.random.seed(seed); random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ── Load CUDA extension ───────────────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
rs_kernel = load(
    name="rs_kernel16_opt_v2",
    sources=[os.path.join(current_dir, "rs_new.cu")],
    extra_cuda_cflags=["--expt-relaxed-constexpr", "--use_fast_math", "-lineinfo"],
    extra_cflags=["-std=c++17"],
    verbose=False,
)

GF_SIZE  = 1 << 16
GF_ORDER = GF_SIZE - 1


# ── GF(2^16) tables ───────────────────────────────────────────────────────
def init_gf_tables():
    primitive_polynomial = 0x1100B
    gf_log = np.zeros(GF_SIZE,      dtype=np.uint16)
    gf_exp = np.zeros(2 * GF_ORDER, dtype=np.uint16)
    x = 1
    for i in range(GF_ORDER):
        gf_exp[i] = x
        gf_log[x] = i
        x <<= 1
        if x & GF_SIZE:
            x ^= primitive_polynomial
    gf_exp[GF_ORDER:] = gf_exp[:GF_ORDER]
    return gf_exp, gf_log

_gf_exp_np, _gf_log_np = init_gf_tables()


# ── Host-side GF arithmetic (matrix inversion only) ──────────────────────
def _gf_mul_np(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    s = int(_gf_log_np[a]) + int(_gf_log_np[b])
    if s >= GF_ORDER:
        s -= GF_ORDER
    return int(_gf_exp_np[s])

def _gf_inv_np(a: int) -> int:
    d = -int(_gf_log_np[a]) % GF_ORDER
    return int(_gf_exp_np[d])


def compute_inv_vandermonde(missing_data: list[int],
                            parity_used:  list[int]) -> np.ndarray:
    """
    Build and invert the m×m Vandermonde submatrix in GF(2^16) on the host.
    Returns flat (m*m,) uint16 array in column-major order matching the
    kernel's  s_invV[col * m + row]  indexing.
    """
    m = len(missing_data)
    assert m == len(parity_used)
    if m == 0:
        return np.zeros(0, dtype=np.uint16)

    V   = [[0]*m for _ in range(m)]
    inv = [[1 if r == c else 0 for c in range(m)] for r in range(m)]

    for r, pr in enumerate(parity_used):
        for c, mc in enumerate(missing_data):
            e = (pr + 1) * mc % GF_ORDER
            V[r][c] = int(_gf_exp_np[e]) if (e != 0 or mc == 0) else 0

    # Gauss-Jordan in GF(2^16)
    for i in range(m):
        pivot = next((r for r in range(i, m) if V[r][i] != 0), -1)
        assert pivot != -1, "Vandermonde matrix is singular."
        if pivot != i:
            V[i],   V[pivot]   = V[pivot],   V[i]
            inv[i], inv[pivot] = inv[pivot], inv[i]

        piv_inv = _gf_inv_np(V[i][i])
        for c in range(m):
            V[i][c]   = _gf_mul_np(V[i][c],   piv_inv)
            inv[i][c] = _gf_mul_np(inv[i][c], piv_inv)

        for r in range(m):
            if r == i or V[r][i] == 0:
                continue
            f = V[r][i]
            for c in range(m):
                V[r][c]   ^= _gf_mul_np(f, V[i][c])
                inv[r][c] ^= _gf_mul_np(f, inv[i][c])

    # Column-major layout: inv_flat[col * m + row] = inv[row][col]
    inv_flat = np.zeros(m * m, dtype=np.uint16)
    for r in range(m):
        for c in range(m):
            inv_flat[c * m + r] = inv[r][c]
    return inv_flat


# ── Shard sizing ──────────────────────────────────────────────────────────
def shard_size_and_pad(total_symbols: int, num_data_shards: int):
    pad      = (-total_symbols) % num_data_shards
    shard_sz = (total_symbols + pad) // num_data_shards
    return shard_sz, pad


# ── Layer weight construction ─────────────────────────────────────────────
def make_layer_weights(hidden: int, intermediate: int,
                       num_heads: int, head_dim: int,
                       device: torch.device) -> dict[str, torch.Tensor]:
    proj_dim = num_heads * head_dim
    return {
        "q_proj":    torch.randn(hidden, proj_dim,     device=device, dtype=torch.float16),
        "k_proj":    torch.randn(hidden, proj_dim,     device=device, dtype=torch.float16),
        "v_proj":    torch.randn(hidden, proj_dim,     device=device, dtype=torch.float16),
        "o_proj":    torch.randn(proj_dim, hidden,     device=device, dtype=torch.float16),
        "gate_proj": torch.randn(intermediate, hidden, device=device, dtype=torch.float16),
        "up_proj":   torch.randn(intermediate, hidden, device=device, dtype=torch.float16),
        "down_proj": torch.randn(hidden, intermediate, device=device, dtype=torch.float16),
        "ln_input":  torch.randn(hidden,               device=device, dtype=torch.float16),
        "ln_post":   torch.randn(hidden,               device=device, dtype=torch.float16),
    }


# ── Flatten / unflatten weights ───────────────────────────────────────────
def flatten_weights(weights: dict[str, torch.Tensor]):
    """Zero-copy reinterpret fp16 → uint16, then concatenate all tensors."""
    parts  = []
    shapes = {}
    offset = 0
    for name, w in weights.items():
        n = w.numel()
        shapes[name] = (offset, n, w.shape)
        parts.append(w.contiguous().view(torch.uint16).view(-1))
        offset += n
    return torch.cat(parts), shapes


def unflatten_weights(flat_u16: torch.Tensor,
                      shapes: dict) -> dict[str, torch.Tensor]:
    out = {}
    for name, (offset, n, shape) in shapes.items():
        out[name] = flat_u16[offset:offset + n].view(torch.float16).view(shape)
    return out


# ── Encode ────────────────────────────────────────────────────────────────
def encode_weights(flat_u16: torch.Tensor,
                   num_data_shards: int,
                   num_parity_shards: int,
                   gf_exp: torch.Tensor,
                   gf_log: torch.Tensor):
    total    = flat_u16.numel()
    shard_sz, _ = shard_size_and_pad(total, num_data_shards)
    total_shards = num_data_shards + num_parity_shards

    data_padded = torch.zeros(num_data_shards * shard_sz,
                              dtype=torch.uint16, device=flat_u16.device)
    data_padded[:total].copy_(flat_u16)

    shards = torch.empty((total_shards, shard_sz),
                         dtype=torch.uint16, device=flat_u16.device)

    rs_kernel.encode_rs_u16(data_padded, shards,
                            num_data_shards, num_parity_shards,
                            shard_sz, gf_exp, gf_log)
    return shards, shard_sz, total


# ── Decode ────────────────────────────────────────────────────────────────
def decode_weights(shards: torch.Tensor,
                   num_data_shards: int,
                   num_parity_shards: int,
                   shard_sz: int,
                   total: int,
                   missing_shards: list[int],
                   gf_exp: torch.Tensor,
                   gf_log: torch.Tensor) -> torch.Tensor:

    missing_data = sorted([i for i in missing_shards if i < num_data_shards])
    missing_par  = sorted([i - num_data_shards for i in missing_shards
                           if num_data_shards <= i < num_data_shards + num_parity_shards])
    avail_par    = [p for p in range(num_parity_shards) if p not in missing_par]

    if len(missing_data) > len(avail_par):
        raise RuntimeError("Not enough parity shards to reconstruct.")
    if len(missing_data) > 16:
        raise RuntimeError("At most 16 missing data shards supported.")

    if missing_data:
        parity_used    = avail_par[:len(missing_data)]
        inv_V_np       = compute_inv_vandermonde(missing_data, parity_used)
        inv_V          = torch.from_numpy(inv_V_np).to(device=shards.device,
                                                        dtype=torch.uint16)
        missing_data_t = torch.tensor(missing_data, dtype=torch.int32,
                                      device=shards.device)
        parity_used_t  = torch.tensor(parity_used,  dtype=torch.int32,
                                      device=shards.device)

        rs_kernel.reconstruct_rs_u16_inplace(
            shards, shard_sz, num_data_shards, num_parity_shards,
            missing_data_t, parity_used_t,
            inv_V,
            gf_exp, gf_log,
        )

    return shards[:num_data_shards].contiguous().view(-1)[:total]


# ── Main benchmark ────────────────────────────────────────────────────────
def main(args):
    set_seed(0)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")

    device = torch.device("cuda")

    weights = make_layer_weights(
        hidden=args.hidden_size,
        intermediate=args.intermediate_size,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        device=device,
    )

    total_params = sum(w.numel() for w in weights.values())
    total_gb     = sum(tensor_memory_size(w) for w in weights.values())

    print("=" * 60)
    print("7B-style single-layer RS coding benchmark")
    print("=" * 60)
    print(f"  hidden_size     : {args.hidden_size}")
    print(f"  intermediate    : {args.intermediate_size}")
    print(f"  num_heads       : {args.num_heads}")
    print(f"  head_dim        : {args.head_dim}")
    print(f"  dtype           : float16")
    print(f"  total params    : {total_params:,}")
    print(f"  total weight mem: {total_gb:.4f} GB")
    print(f"  data shards     : {args.num_data_shards}")
    print(f"  parity shards   : {args.num_parity_shards}")
    print(f"  missing shards  : {args.missing_shards}")
    print()

    # GF tables — built once, reused for all encode/decode calls
    gf_exp = torch.from_numpy(_gf_exp_np).to(device=device, dtype=torch.uint16)
    gf_log = torch.from_numpy(_gf_log_np).to(device=device, dtype=torch.uint16)

    # Flatten weights once (preprocessing, not timed)
    flat_u16, shapes = flatten_weights(weights)

    REPS = 5

    # ── Encode benchmark ──────────────────────────────────────────────────
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(REPS):
        shards, shard_sz, total = encode_weights(
            flat_u16, args.num_data_shards, args.num_parity_shards,
            gf_exp, gf_log)
    t1.record()
    torch.cuda.synchronize()
    enc_ms = t0.elapsed_time(t1) / REPS

    # Simulate shard loss
    shards_damaged = shards.clone()
    for idx in args.missing_shards:
        if 0 <= idx < shards_damaged.size(0):
            shards_damaged[idx].zero_()
        else:
            raise ValueError(f"Shard index {idx} out of range "
                             f"0..{shards_damaged.size(0)-1}")

    # ── Decode benchmark ──────────────────────────────────────────────────
    torch.cuda.synchronize()
    t2 = torch.cuda.Event(enable_timing=True)
    t3 = torch.cuda.Event(enable_timing=True)
    t2.record()
    for _ in range(REPS):
        flat_rec = decode_weights(
            shards_damaged.clone(),
            args.num_data_shards, args.num_parity_shards,
            shard_sz, total, args.missing_shards,
            gf_exp, gf_log)
    t3.record()
    torch.cuda.synchronize()
    dec_ms = t2.elapsed_time(t3) / REPS

    # Throughput (weight data only, 2 bytes per uint16)
    weight_gb = total * 2 / (1024 ** 3)
    print(f"  Encode : {enc_ms:.3f} ms   ({weight_gb / (enc_ms/1e3):.2f} GB/s)")
    print(f"  Decode : {dec_ms:.3f} ms   ({weight_gb / (dec_ms/1e3):.2f} GB/s)")
    print()

    # Correctness (bitwise exact since fp16 ↔ uint16 is lossless)
    rec_weights = unflatten_weights(flat_rec, shapes)
    all_ok = all(
        torch.equal(weights[k].view(torch.uint16), rec_weights[k].view(torch.uint16))
        for k in weights
    )
    print(f"  Exact reconstruction: {'PASS ✓' if all_ok else 'FAIL ✗'}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RS coding benchmark on a single 7B-style transformer layer.")
    # Architecture
    parser.add_argument("--hidden_size",       type=int, default=4096)
    parser.add_argument("--intermediate_size", type=int, default=11008)
    parser.add_argument("--num_heads",         type=int, default=32)
    parser.add_argument("--head_dim",          type=int, default=128)
    # RS parameters
    parser.add_argument("--num_data_shards",   type=int, default=4)
    parser.add_argument("--num_parity_shards", type=int, default=1)
    parser.add_argument("--missing_shards",    type=int, nargs="+", default=[0],
                        help="Shard indices to erase (0-based)")
    args = parser.parse_args()

    if len(args.missing_shards) > args.num_parity_shards:
        print("Error: more missing shards than parity shards — unrecoverable.")
        sys.exit(1)

    main(args)
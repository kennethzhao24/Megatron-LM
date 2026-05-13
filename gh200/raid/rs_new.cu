#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

namespace py = pybind11;

using gf_t = uint16_t;
using u64  = unsigned long long;

static constexpr int GF_SIZE  = (1 << 16);
static constexpr int GF_ORDER = GF_SIZE - 1;
static constexpr int VEC_W    = 8;   // uint16 elements per uint4 (128-bit) load
static constexpr int MAX_M    = 16;  // max missing data shards

// ============================================================
// GF(2^16) arithmetic — route table lookups through read-only
// texture cache (__ldg) to avoid thrashing the data L1 cache.
// ============================================================
__device__ __forceinline__ gf_t gf_mul(gf_t a, gf_t b,
                                       const gf_t* __restrict__ gf_exp,
                                       const gf_t* __restrict__ gf_log) {
    if (a == 0 || b == 0) return 0;
    int s = (int)__ldg(&gf_log[a]) + (int)__ldg(&gf_log[b]);
    if (s >= GF_ORDER) s -= GF_ORDER;
    return __ldg(&gf_exp[s]);
}

__device__ __forceinline__ gf_t gf_div(gf_t a, gf_t b,
                                       const gf_t* __restrict__ gf_exp,
                                       const gf_t* __restrict__ gf_log) {
    if (a == 0) return 0;
    if (b == 0) return 0;
    int d = (int)__ldg(&gf_log[a]) - (int)__ldg(&gf_log[b]);
    if (d < 0) d += GF_ORDER;
    return __ldg(&gf_exp[d]);
}

// ============================================================
// Pack / Unpack FP16 <-> U16  (unchanged, already fast)
// ============================================================
__global__ void pack_fp16_pair_to_u16_kernel(const __half* __restrict__ k,
                                             const __half* __restrict__ v,
                                             gf_t* __restrict__ out,
                                             u64 N_k, u64 N_v) {
    u64 idx = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (idx < N_k) {
        out[idx] = reinterpret_cast<const uint16_t*>(k)[idx];
    }
    if (idx < N_v) {
        out[N_k + idx] = reinterpret_cast<const uint16_t*>(v)[idx];
    }
}

__global__ void u16_to_fp16_split_kernel(const gf_t* __restrict__ in,
                                         __half* __restrict__ k_out,
                                         __half* __restrict__ v_out,
                                         u64 N_k, u64 N_v) {
    u64 idx = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (idx < N_k) {
        reinterpret_cast<uint16_t*>(k_out)[idx] = in[idx];
    }
    if (idx < N_v) {
        reinterpret_cast<uint16_t*>(v_out)[idx] = in[N_k + idx];
    }
}

// ============================================================
// RS ENCODE  — optimised
//
// Key changes vs original:
//  1. 128-bit (uint4) vectorised loads/stores: 8 uint16 per txn.
//  2. Coefficient gf_exp[(p+1)*k % GF_ORDER] hoisted out of the
//     per-element lane loop — one table lookup per (p,k) pair
//     instead of 8 × 2 = 16.
//  3. GF table lookups use __ldg (read-only / texture cache).
//  4. Accumulator kept in registers; single vectorised store at end.
//  5. Scalar tail handles shard_size % 8 != 0 remainder.
// ============================================================
__global__ void encode_rs_u16_kernel(
        const gf_t* __restrict__ data_padded,   // [data_shards * shard_size]
        gf_t*       __restrict__ shards,         // [(data+parity) * shard_size]
        int   data_shards,
        int   parity_shards,
        u64   shard_size,
        const gf_t* __restrict__ gf_exp,
        const gf_t* __restrict__ gf_log)
{
    u64 tid    = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    u64 stride = (u64)blockDim.x * gridDim.x;

    // ----- Copy data shards verbatim (vectorised) -----
    u64 total_data = (u64)data_shards * shard_size;
    u64 vec_total  = total_data / VEC_W;

    for (u64 vi = tid; vi < vec_total; vi += stride) {
        reinterpret_cast<uint4*>(shards)[vi] =
            reinterpret_cast<const uint4*>(data_padded)[vi];
    }
    // scalar tail
    for (u64 i = tid + vec_total * VEC_W; i < total_data; i += stride) {
        shards[i] = data_padded[i];
    }

    // ----- Compute parity shards (vectorised, coef hoisted) -----
    u64 vec_sz = shard_size / VEC_W;   // number of uint4 chunks per shard

    for (u64 vi = tid; vi < vec_sz; vi += stride) {
        u64 base_pos = vi * VEC_W;

        for (int p = 0; p < parity_shards; ++p) {
            uint16_t acc[VEC_W] = {};

            for (int k = 0; k < data_shards; ++k) {
                // Hoist: coefficient is constant for all 8 lanes
                int   e    = (int)((u64)(p + 1) * (u64)k % (u64)GF_ORDER);
                gf_t  coef = __ldg(&gf_exp[e]);

                // 128-bit coalesced load for 8 consecutive symbols
                uint4 chunk;
                chunk = *reinterpret_cast<const uint4*>(
                            data_padded + (u64)k * shard_size + base_pos);
                const uint16_t* x = reinterpret_cast<const uint16_t*>(&chunk);

                #pragma unroll
                for (int lane = 0; lane < VEC_W; ++lane) {
                    if (x[lane]) {
                        acc[lane] ^= gf_mul(x[lane], coef, gf_exp, gf_log);
                    }
                }
            }

            // 128-bit store
            uint4 out_vec;
            *reinterpret_cast<uint4*>(&out_vec) =
                *reinterpret_cast<uint4*>(acc);
            *reinterpret_cast<uint4*>(
                shards + ((u64)data_shards + (u64)p) * shard_size + base_pos)
                = out_vec;
        }
    }

    // scalar tail for shard_size % VEC_W != 0
    u64 tail_start = vec_sz * VEC_W;
    for (u64 pos = tail_start + tid; pos < shard_size; pos += stride) {
        for (int p = 0; p < parity_shards; ++p) {
            gf_t acc = 0;
            for (int k = 0; k < data_shards; ++k) {
                int  e    = (int)((u64)(p + 1) * (u64)k % (u64)GF_ORDER);
                gf_t coef = __ldg(&gf_exp[e]);
                gf_t x    = data_padded[(u64)k * shard_size + pos];
                if (x) acc ^= gf_mul(x, coef, gf_exp, gf_log);
            }
            shards[((u64)data_shards + (u64)p) * shard_size + pos] = acc;
        }
    }
}

// ============================================================
// RS RECONSTRUCT  — optimised
//
// Key changes vs original:
//  1. Vandermonde inverse is precomputed on the HOST (Python side)
//     and passed in as inv_V [m*m, uint16].  The per-block serial
//     Gauss-Jordan is completely removed — all threads are active
//     from the start.
//  2. inv_V loaded into shared memory once per block (m*m ≤ 256 B).
//  3. 128-bit vectorised loads/stores for the mat-vec multiply.
//  4. __ldg for all GF table accesses.
// ============================================================
__global__ void reconstruct_rs_u16_kernel_inplace(
        gf_t* __restrict__ shards,
        u64   shard_size,
        int   data_shards,
        int   parity_shards,
        const int*  __restrict__ missing_data,   // [m]
        int   m,
        const int*  __restrict__ parity_used,    // [m]
        const gf_t* __restrict__ inv_V,          // [m*m] row-major, precomputed on host
        const gf_t* __restrict__ gf_exp,
        const gf_t* __restrict__ gf_log)
{
    if (m == 0) return;

    __shared__ int  s_missing[MAX_M];
    __shared__ int  s_pused[MAX_M];
    __shared__ gf_t s_invV[MAX_M * MAX_M];   // ≤ 512 bytes

    // Load metadata into shared memory — only first m threads needed
    if (threadIdx.x < m) {
        s_missing[threadIdx.x] = missing_data[threadIdx.x];
        s_pused[threadIdx.x]   = parity_used[threadIdx.x];
    }
    // Load precomputed inverse — m*m entries, spread across threads
    int inv_size = m * m;
    for (int i = threadIdx.x; i < inv_size; i += blockDim.x) {
        s_invV[i] = __ldg(&inv_V[i]);
    }
    __syncthreads();

    // ----- Vectorised main loop -----
    u64 tid    = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    u64 stride = (u64)blockDim.x * gridDim.x;
    u64 vec_sz = shard_size / VEC_W;

    for (u64 vi = tid; vi < vec_sz; vi += stride) {
        u64 base_pos = vi * VEC_W;

        // Build RHS b[r] for each parity equation, vectorised over 8 lanes
        uint16_t b[MAX_M][VEC_W];

        for (int r = 0; r < m; ++r) {
            int pr = s_pused[r];
            // Load parity shard symbols
            uint4 par_chunk = *reinterpret_cast<const uint4*>(
                shards + ((u64)data_shards + (u64)pr) * shard_size + base_pos);
            const uint16_t* pv = reinterpret_cast<const uint16_t*>(&par_chunk);
            #pragma unroll
            for (int lane = 0; lane < VEC_W; ++lane) b[r][lane] = pv[lane];

            // Subtract contributions of known (non-missing) data shards
            int base = pr + 1;
            for (int k = 0; k < data_shards; ++k) {
                bool is_missing = false;
                #pragma unroll
                for (int mi = 0; mi < MAX_M; ++mi) {
                    if (mi >= m) break;
                    if (k == s_missing[mi]) { is_missing = true; break; }
                }
                if (is_missing) continue;

                int  e    = (int)((u64)base * (u64)k % (u64)GF_ORDER);
                gf_t coef = __ldg(&gf_exp[e]);

                uint4 d_chunk = *reinterpret_cast<const uint4*>(
                    shards + (u64)k * shard_size + base_pos);
                const uint16_t* dv = reinterpret_cast<const uint16_t*>(&d_chunk);

                #pragma unroll
                for (int lane = 0; lane < VEC_W; ++lane) {
                    if (dv[lane]) b[r][lane] ^= gf_mul(dv[lane], coef, gf_exp, gf_log);
                }
            }
        }

        // Solve: x = inv_V * b  (mat-vec in GF, write recovered shards)
        for (int c = 0; c < m; ++c) {
            uint16_t sum[VEC_W] = {};
            for (int r = 0; r < m; ++r) {
                gf_t coef = s_invV[c * m + r];
                if (!coef) continue;
                #pragma unroll
                for (int lane = 0; lane < VEC_W; ++lane) {
                    if (b[r][lane]) sum[lane] ^= gf_mul(coef, b[r][lane], gf_exp, gf_log);
                }
            }
            uint4 out_vec = *reinterpret_cast<uint4*>(sum);
            *reinterpret_cast<uint4*>(
                shards + (u64)s_missing[c] * shard_size + base_pos) = out_vec;
        }
    }

    // ----- Scalar tail -----
    u64 tail_start = vec_sz * VEC_W;
    for (u64 pos = tail_start + tid; pos < shard_size; pos += stride) {
        gf_t b[MAX_M];
        for (int r = 0; r < m; ++r) {
            int  pr  = s_pused[r];
            gf_t acc = shards[((u64)data_shards + (u64)pr) * shard_size + pos];
            int  base = pr + 1;
            for (int k = 0; k < data_shards; ++k) {
                bool is_missing = false;
                for (int mi = 0; mi < m; ++mi)
                    if (k == s_missing[mi]) { is_missing = true; break; }
                if (is_missing) continue;
                int  e    = (int)((u64)base * (u64)k % (u64)GF_ORDER);
                gf_t coef = __ldg(&gf_exp[e]);
                gf_t x    = shards[(u64)k * shard_size + pos];
                if (x) acc ^= gf_mul(x, coef, gf_exp, gf_log);
            }
            b[r] = acc;
        }
        for (int c = 0; c < m; ++c) {
            gf_t sum = 0;
            for (int r = 0; r < m; ++r) {
                gf_t coef = s_invV[c * m + r];
                if (coef && b[r]) sum ^= gf_mul(coef, b[r], gf_exp, gf_log);
            }
            shards[(u64)s_missing[c] * shard_size + pos] = sum;
        }
    }
}

// ============================================================
// C++ / Python bindings
// ============================================================
void pack_fp16_pair_to_u16(torch::Tensor k, torch::Tensor v, torch::Tensor out) {
    TORCH_CHECK(k.is_cuda() && v.is_cuda() && out.is_cuda(), "All tensors must be on CUDA");
    TORCH_CHECK(k.dtype() == torch::kFloat16 && v.dtype() == torch::kFloat16, "k/v must be float16");
    TORCH_CHECK(out.dtype() == torch::kUInt16, "out must be uint16");

    const u64 N_k = (u64)k.numel();
    const u64 N_v = (u64)v.numel();
    int threads = 256;
    int blocks  = (int)((std::max(N_k, N_v) + threads - 1ULL) / threads);
    if (blocks < 1) blocks = 1;

    pack_fp16_pair_to_u16_kernel<<<blocks, threads>>>(
        reinterpret_cast<const __half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(v.data_ptr<at::Half>()),
        out.data_ptr<gf_t>(), N_k, N_v);
}

void u16_to_fp16_split(torch::Tensor in, torch::Tensor k_out, torch::Tensor v_out,
                       int64_t N_k, int64_t N_v) {
    TORCH_CHECK(in.is_cuda() && k_out.is_cuda() && v_out.is_cuda(), "All tensors must be on CUDA");
    TORCH_CHECK(in.dtype() == torch::kUInt16, "in must be uint16");
    TORCH_CHECK(k_out.dtype() == torch::kFloat16 && v_out.dtype() == torch::kFloat16, "outputs must be float16");

    u64 n_k = (u64)N_k, n_v = (u64)N_v;
    int threads = 256;
    int blocks  = (int)((std::max(n_k, n_v) + threads - 1ULL) / threads);
    if (blocks < 1) blocks = 1;

    u16_to_fp16_split_kernel<<<blocks, threads>>>(
        in.data_ptr<gf_t>(),
        reinterpret_cast<__half*>(k_out.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(v_out.data_ptr<at::Half>()),
        n_k, n_v);
}

void encode_rs_u16(torch::Tensor data_padded, torch::Tensor shards,
                   int data_shards, int parity_shards, int shard_size,
                   torch::Tensor gf_exp, torch::Tensor gf_log) {
    TORCH_CHECK(data_padded.is_cuda() && shards.is_cuda(), "CUDA tensors expected");
    TORCH_CHECK(data_padded.dtype() == torch::kUInt16 && shards.dtype() == torch::kUInt16, "uint16 expected");
    TORCH_CHECK(gf_exp.dtype() == torch::kUInt16 && gf_log.dtype() == torch::kUInt16, "gf tables must be uint16");

    u64 sz      = (u64)shard_size;
    int threads = 256;
    // Grid sized to cover vec_sz chunks; tail handled inside kernel
    u64 vec_sz  = sz / VEC_W;
    int blocks  = (int)((vec_sz + threads - 1ULL) / threads);
    if (blocks < 1) blocks = 1;

    encode_rs_u16_kernel<<<blocks, threads>>>(
        data_padded.data_ptr<gf_t>(),
        shards.data_ptr<gf_t>(),
        data_shards, parity_shards, sz,
        gf_exp.data_ptr<gf_t>(),
        gf_log.data_ptr<gf_t>());
}

// New signature: accepts precomputed inv_V tensor instead of doing
// Gauss-Jordan inside the kernel.
void reconstruct_rs_u16_inplace(torch::Tensor shards,
                                int shard_size, int data_shards, int parity_shards,
                                torch::Tensor missing_data, torch::Tensor parity_used,
                                torch::Tensor inv_V,          // [m*m] uint16, row-major
                                torch::Tensor gf_exp, torch::Tensor gf_log) {
    TORCH_CHECK(shards.is_cuda() && gf_exp.is_cuda() && gf_log.is_cuda(), "CUDA tensors expected");
    TORCH_CHECK(shards.dtype() == torch::kUInt16, "shards must be uint16");
    TORCH_CHECK(missing_data.dtype() == torch::kInt32 && parity_used.dtype() == torch::kInt32, "indices must be int32");
    TORCH_CHECK(inv_V.is_cuda() && inv_V.dtype() == torch::kUInt16, "inv_V must be uint16 CUDA tensor");

    int m = (int)missing_data.numel();
    TORCH_CHECK(m <= MAX_M, "m (missing data shards) must be <= 16");

    u64 sz      = (u64)shard_size;
    int threads = 256;
    u64 vec_sz  = sz / VEC_W;
    int blocks  = (int)((vec_sz + threads - 1ULL) / threads);
    if (blocks < 1) blocks = 1;

    reconstruct_rs_u16_kernel_inplace<<<blocks, threads>>>(
        shards.data_ptr<gf_t>(),
        sz, data_shards, parity_shards,
        missing_data.data_ptr<int>(), m,
        parity_used.data_ptr<int>(),
        inv_V.data_ptr<gf_t>(),
        gf_exp.data_ptr<gf_t>(),
        gf_log.data_ptr<gf_t>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pack_fp16_pair_to_u16",       &pack_fp16_pair_to_u16,
          "Pack K/V FP16 -> one uint16 stream");
    m.def("u16_to_fp16_split",           &u16_to_fp16_split,
          "Split uint16 stream -> K/V FP16",
          py::arg("in"), py::arg("k_out"), py::arg("v_out"),
          py::arg("N_k"), py::arg("N_v"));
    m.def("encode_rs_u16",               &encode_rs_u16,
          "RS encode over GF(2^16) — vectorised");
    m.def("reconstruct_rs_u16_inplace",  &reconstruct_rs_u16_inplace,
          "RS reconstruct missing DATA shards in-place (inv_V precomputed on host)",
          py::arg("shards"), py::arg("shard_size"), py::arg("data_shards"),
          py::arg("parity_shards"), py::arg("missing_data"), py::arg("parity_used"),
          py::arg("inv_V"), py::arg("gf_exp"), py::arg("gf_log"));
}

#!/bin/bash

set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1

# Compiler env vars needed for compiling the C++ dataset helpers inside the container
export CC="${CC:-/usr/bin/cc}"
if [[ -z "${CXX:-}" || "${CXX}" == "CC" ]]; then
  export CXX="/usr/bin/g++"
fi

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_MEGATRON_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE="${APPTAINER_IMAGE:-/u/yzhao25/Sys-RL/slime_containers/slime-base.sif}"

# Use a dedicated checkpoint root so latency runs do not resume from or overwrite
# normal training checkpoints.
RUN_NAME="${RUN_NAME:-llama3_8b_bf16_4gpu_ckpt_latency}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/work/nvme/bfgy/yzhao25}"
HOST_CHECKPOINT_PATH="${HOST_CHECKPOINT_PATH:-${CHECKPOINT_PATH:-${CHECKPOINT_ROOT}/${RUN_NAME}}}"
CONTAINER_CHECKPOINT_PATH="${CONTAINER_CHECKPOINT_PATH:-/mnt/checkpoints/${RUN_NAME}}"
TENSORBOARD_LOGS_PATH="${TENSORBOARD_LOGS_PATH:-${HOST_MEGATRON_DIR}/tensorboard_logs/${RUN_NAME}}"
DATA_CACHE_PATH="${DATA_CACHE_PATH:-${HOST_MEGATRON_DIR}/benchmark_cache_llama3_8b_bf16}"

mkdir -p "$HOST_CHECKPOINT_PATH" "$TENSORBOARD_LOGS_PATH" "$DATA_CACHE_PATH"

# exclusively for GH200
CACHE_ROOT="/tmp/${USER:-$(id -un)}-cache"
mkdir -p "$CACHE_ROOT/triton" "$CACHE_ROOT/torchinductor" "$CACHE_ROOT/xdg"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torchinductor"

# Distributed setup
GPUS_PER_NODE=4
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6001}

# Match the main GH200 script.
TP_SIZE=2
PP_SIZE=2
CP_SIZE=1

# Short sample-based run for explicit checkpoint latency measurement. The default
# writes one full checkpoint, which avoids needing quota for old + new checkpoints.
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
LATENCY_ITERS="${LATENCY_ITERS:-4}"
SAVE_INTERVAL="${SAVE_INTERVAL:-4}"
SAVE_RETAIN_INTERVAL="${SAVE_RETAIN_INTERVAL:-1000000000}"
TRAIN_SAMPLES="$((GLOBAL_BATCH_SIZE * LATENCY_ITERS))"

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

MODEL_ARGS=(
    --use-mcore-models
    --num-layers 32
    --hidden-size 4096
    --ffn-hidden-size 14336
    --num-attention-heads 32
    --group-query-attention
    --num-query-groups 8
    --kv-channels 128
    --seq-length 8192
    --max-position-embeddings 8192
    --position-embedding-type rope
    --rotary-base 1000000
    --rotary-percent 1.0
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --swiglu
    --init-method-std 0.0134
    --attention-backend fused
    --apply-layernorm-1p
    --untie-embeddings-and-output-weights
    --disable-bias-linear
)

TRAINING_ARGS=(
    --micro-batch-size "$MICRO_BATCH_SIZE"
    --global-batch-size "$GLOBAL_BATCH_SIZE"
    --train-samples "$TRAIN_SAMPLES"
    --lr-decay-samples "$TRAIN_SAMPLES"
    --lr-warmup-samples 0
    --lr 0.00015
    --min-lr 0.00001
    --lr-decay-style cosine
    --clip-grad 1.0
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --bf16
    --grad-reduce-in-bf16
    --fp8-format hybrid
    --fp8-amax-history-len 1024
    --fp8-amax-compute-algo max
    --fp8-param-gather
    --cross-entropy-loss-fusion
    --calculate-per-token-loss
    --manual-gc
    --empty-unused-memory-level 1
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size $TP_SIZE
    --pipeline-model-parallel-size $PP_SIZE
    --context-parallel-size $CP_SIZE
    --sequence-parallel
)

DDP_ARGS=(
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)

DATA_ARGS=(
    --mock-data
    --tokenizer-type NullTokenizer
    --vocab-size 128256
    --data-cache-path "$DATA_CACHE_PATH"
    --split 99,1,0
    --no-create-attention-mask-in-dataloader
    --no-mmap-bin-files
    --num-workers 1
)

LOGGING_ARGS=(
    --log-interval 1
    --eval-iters 1
    --eval-interval 100000
    --save-interval "$SAVE_INTERVAL"
    --save-retain-interval "$SAVE_RETAIN_INTERVAL"
    --log-throughput
    --log-timers-to-tensorboard
    --ckpt-format torch_dist
    --ckpt-assume-constant-structure
    --distributed-timeout-minutes 60
    --save "$CONTAINER_CHECKPOINT_PATH"
    --tensorboard-dir "$TENSORBOARD_LOGS_PATH"
)

# Optional resume/load measurement:
#   LOAD_CHECKPOINT_PATH=/mnt/checkpoints/llama3_8b_bf16_4gpu_ckpt_latency bash gh200/train_llama3_8b_4gpu_ckpt_latency.sh
if [[ -n "${LOAD_CHECKPOINT_PATH:-}" ]]; then
    LOGGING_ARGS+=(--load "$LOAD_CHECKPOINT_PATH")
fi

if [[ "${ASYNC_SAVE:-0}" == "1" ]]; then
    LOGGING_ARGS+=(--async-save)
fi

echo "Checkpoint latency run:"
echo "  host checkpoint path:      ${HOST_CHECKPOINT_PATH}"
echo "  container checkpoint path: ${CONTAINER_CHECKPOINT_PATH}"
echo "  train samples:   ${TRAIN_SAMPLES} (${LATENCY_ITERS} iterations at global batch ${GLOBAL_BATCH_SIZE})"
echo "  save interval:   ${SAVE_INTERVAL}"
echo "  retain interval: ${SAVE_RETAIN_INTERVAL} (after a successful later save, deletes older checkpoints)"
echo "  optimizer state: saved"
echo "  async save:      ${ASYNC_SAVE:-0}"

apptainer exec --nv \
    --bind "${HOST_CHECKPOINT_PATH}:${CONTAINER_CHECKPOINT_PATH}" \
    --bind "${TENSORBOARD_LOGS_PATH}:${TENSORBOARD_LOGS_PATH}" \
    --bind "${DATA_CACHE_PATH}:${DATA_CACHE_PATH}" \
    "${IMAGE}" \
    torchrun "${DISTRIBUTED_ARGS[@]}" \
        /root/Megatron-LM/pretrain_gpt.py \
        "${MODEL_ARGS[@]}" \
        "${TRAINING_ARGS[@]}" \
        "${MODEL_PARALLEL_ARGS[@]}" \
        "${DDP_ARGS[@]}" \
        "${DATA_ARGS[@]}" \
        "${LOGGING_ARGS[@]}"

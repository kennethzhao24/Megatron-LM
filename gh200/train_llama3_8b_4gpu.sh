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
IMAGE="${APPTAINER_IMAGE:-/u/yzhao25/slime_containers/slime-base.sif}"

CHECKPOINT_PATH="${HOST_MEGATRON_DIR}/checkpoints/llama3_8b_bf16_4gpu"
TENSORBOARD_LOGS_PATH="${HOST_MEGATRON_DIR}/tensorboard_logs/llama3_8b_bf16_4gpu"
DATA_CACHE_PATH="${HOST_MEGATRON_DIR}/benchmark_cache_llama3_8b_bf16"

mkdir -p "$CHECKPOINT_PATH" "$TENSORBOARD_LOGS_PATH" "$DATA_CACHE_PATH"

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

# Parallelism: TP=2, PP=1, CP=1 → 2 data-parallel replicas
# TP=2 enables sequence-parallel and halves per-GPU activation memory
TP_SIZE=2
PP_SIZE=1
CP_SIZE=1

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
    --micro-batch-size 2
    --global-batch-size 64
    --train-samples 100000
    --lr-decay-samples 95000
    --lr-warmup-samples 5000
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
    # --recompute-granularity full
    # --recompute-method uniform
    # --recompute-num-layers 16
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
    --eval-iters 32
    --eval-interval 100
    --save-interval 1000
    --log-throughput
    --ckpt-format torch_dist
    --distributed-timeout-minutes 60
    --save "$CHECKPOINT_PATH"
    --load "$CHECKPOINT_PATH"
    --tensorboard-dir "$TENSORBOARD_LOGS_PATH"
)

apptainer exec --nv \
    --bind "${CHECKPOINT_PATH}:${CHECKPOINT_PATH}" \
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

#!/bin/bash
# GLM-5.3-Flash NVFP4 Spark (local-inference-lab). 8-bit KV cache,
# MTP-3, GPUs split (DCP=2). Aligned to the official glm53-spark-tp2
# preset (lil-docker-builds runtime/presets.yaml + profiles/glm53-flash).
set -euo pipefail

# ============================================================
# Image
# ============================================================
IMAGE="localhost/vllm-glm53f-exl3-sm120-turbo:r5"
PODNAME="vllm"
VLLM_PORT=8000

# ============================================================
# Paths
# ============================================================
# Normal locations. Export HF_CACHE / LOCAL_MODELS to override.
HF_CACHE="${HF_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}}"
LOCAL_MODELS="${LOCAL_MODELS:-$HOME/local_models}"

DIR=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
ROOTFS_CACHE="${DIR}/cache-rootfs"

mkdir -p "${HF_CACHE}" "${ROOTFS_CACHE}/triton" "${ROOTFS_CACHE}/inductor" "${ROOTFS_CACHE}/b12x"
mkdir -p "${DIR}/container-tmp"

# ============================================================
# Model
# ============================================================
MODELNAME="GLM-5.3-Flash"
MODEL="${LOCAL_MODELS}"/GLM-5.3-Flash-NVFP4-Spark

# ============================================================
# Settings (official glm53-spark-tp2 preset values)
# ============================================================
TP_SIZE=2
DCP_SIZE=2
GPU_UTIL=0.985
CONTEXT_SIZE=327680
MAX_NUM_SEQS=4
MAX_NUM_BATCHED_TOKENS=3072
KV_CACHE_MEMORY_BYTES=4190109696   # 3996 MiB per GPU, pinned
KV_OFFLOADING_SIZE=64             # GiB native host KV tier; empty/0 disables
KV_CACHE_DTYPE=fp8
BLOCK_SIZE=256
CP_KV_INTERLEAVE=4
MTP_TOKENS=3
REASONING_EFFORT=high
HEALTH_START_PERIOD=600
# Native offload pins/registers the GPU cache allocations, so PyTorch's
# remappable VMM segments must be off and the native-L2 keys stable
# across restarts (the same contract as the DeepSeek launcher).
ALLOC_CONF="expandable_segments:True,large_segment_size_mb:12"
VLLM_OFFLOAD=()
if [[ -n "${KV_OFFLOADING_SIZE}" && "${KV_OFFLOADING_SIZE}" != "0" ]]
then
    ALLOC_CONF="expandable_segments:False"
    VLLM_OFFLOAD=(
        -e PYTHONHASHSEED=0
        --kv-offloading-size "${KV_OFFLOADING_SIZE}"
        --kv-offloading-backend native
    )
fi

# Ready-made graph sizes. Spec decoding on: (K+1) x seqs. Off: 4 x seqs
if (( MTP_TOKENS > 0 ))
then
    GRAPH_CAP=$(( MAX_NUM_SEQS * (MTP_TOKENS + 1) ))
else
    GRAPH_CAP=$(( MAX_NUM_SEQS * 4 ))
fi
(( GRAPH_CAP < 6 )) && GRAPH_CAP=6

# The prepared sizes must cover every batch width the server can reach
CAPTURE_SIZES=()
s=1
while (( s <= GRAPH_CAP ))
do
    CAPTURE_SIZES+=("$s")
    s=$(( s < 4 ? s*2 : s+4 ))
done
IFS=,
SIZES_CSV="${CAPTURE_SIZES[*]}"
unset IFS

# ============================================================
# Build the server options
# ============================================================
VLLM_ENV=()
VLLM_BACKEND=()
VLLM_SPEC=()
VLLM_EXTRA=()

# The GPUs cannot talk to each other directly. Direct links are off
VLLM_ENV+=(
    -e VLLM_ENGINE_READY_TIMEOUT_S=${HEALTH_START_PERIOD}
    -e VLLM_ENGINE_ITERATION_TIMEOUT_S=120
    -e OMP_NUM_THREADS=1
    -e HF_HUB_OFFLINE=1
    -e TRANSFORMERS_OFFLINE=1
    -e SAFETENSORS_FAST_GPU=1
    -e VLLM_USE_V2_MODEL_RUNNER=1
    -e VLLM_ENABLE_PCIE_ALLREDUCE=0
    -e NCCL_P2P_DISABLE=1
    -e NCCL_SOCKET_IFNAME=lo
    -e GLOO_SOCKET_IFNAME=lo
    -e NCCL_MIN_NCHANNELS=2
    -e NCCL_MAX_NCHANNELS=2
    -e NCCL_BUFFSIZE=1048576
    -e NCCL_NET_PLUGIN=none
    -e NCCL_TUNER_PLUGIN=none
    -e CUTE_DSL_ARCH=sm_120a
    -e PYTORCH_CUDA_ALLOC_CONF=${ALLOC_CONF}
    -e CUBLAS_WORKSPACE_CONFIG=:4096:1
    -e VLLM_GLM53_MTP_DRAFT_HEAD=nvfp4
    -e VLLM_MXFP8_LM_HEAD=0
    -e VLLM_MTP_NVFP4_LM_HEAD=0
    -e VLLM_LM_HEAD_A16=1
    -e VLLM_B12X_MOE_FP4_FORCE_A16=0
    -e VLLM_DISABLED_KERNELS=MarlinFP8ScaledMMLinearKernel
    -e VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS=65536
    -e VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE=2048
    -e VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE=auto
    -e B12X_MHC_PDL=1
    -e VLLM_GLM53_L2_PREFETCH=1
    -e VLLM_GLM53_L2_PREFETCH_PERSIST_MB=0
    -e VLLM_GLM53_DFLASH_ATTN=1
    -e VLLM_CAUSAL_CONV1D_UPDATE_HOIST=1
    -e VLLM_GLM53_KDA_GATE_SIDE_STREAM=1
    -e VLLM_USE_FLASHINFER_SAMPLER=1
)

# Use the b12x fast code for attention and for the expert layers
VLLM_BACKEND=(
    --attention-backend B12X
    --moe-backend b12x
    --linear-backend b12x
)

# Speculative decoding. The draft part is inside the model file
if [[ "${MTP_TOKENS}" != "0" ]]
then
    VLLM_SPEC=(
        --speculative-config \
        "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_TOKENS},\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"standard\",\"moe_backend\":\"b12x\",\"attention_backend\":\"B12X\"}"
    )
fi

VLLM_EXTRA+=(
    --disable-custom-all-reduce
    --prefill-compute-share 0.4
    --prefill-schedule-interval 1
    --max-parallel-prefills 1
    --no-enable-flashinfer-autotune
)

# ============================================================
# Startup summary
# ============================================================
{
  printf 'launch %s as %s\n' "${MODEL}" "${MODELNAME}"
  printf '  image        %s\n' "${IMAGE}"
  printf '  parallel     tp=%s dcp=%s\n' "${TP_SIZE}" "${DCP_SIZE}"
  printf '  quant        nvfp4 (modelopt_mixed), kv=%s, block=%s\n' "${KV_CACHE_DTYPE}" "${BLOCK_SIZE}"
  printf '  context      %s, kv-pool=%s bytes/gpu, mem-fraction=%s\n' "${CONTEXT_SIZE}" "${KV_CACHE_MEMORY_BYTES}" "${GPU_UTIL}"
  printf '  batching     max-seqs=%s, batched-tokens=%s, graph-cap=%s\n' "${MAX_NUM_SEQS}" "${MAX_NUM_BATCHED_TOKENS}" "${GRAPH_CAP}"
  printf '  speculation  mtp (K=%s, probabilistic)\n' "${MTP_TOKENS}"
  printf '  offloading   %s\n' "$([ -n "${KV_OFFLOADING_SIZE}" ] && [ "${KV_OFFLOADING_SIZE}" != "0" ] && echo "native ${KV_OFFLOADING_SIZE} GiB" || echo off)"
} >&2

# ============================================================
# Container
# ============================================================

podman run --replace --detach --restart=always \
    --entrypoint /bin/bash \
    --health-cmd="curl -f http://localhost:${VLLM_PORT}/health || exit 1" \
    --health-start-period="${HEALTH_START_PERIOD}s" \
    --health-interval=30s \
    --health-on-failure=kill \
    --health-retries=3 \
    --name "${PODNAME}" \
    --device nvidia.com/gpu=all \
    --ipc=host \
    --network=host \
    -v "${LOCAL_MODELS}":/workspace/local_models:ro \
    -v "${ROOTFS_CACHE}":/cache:rw \
    -v "${HF_CACHE}":/root/.cache/huggingface:ro \
    -v "${DIR}/container-tmp":/container-tmp \
    -v "${DIR}/chat_template.multimodal.jinja":/opt/glm53f/chat_template.multimodal.jinja:ro \
    -e TMPDIR=/container-tmp \
    -e TRITON_CACHE_DIR=/cache/triton \
    -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
    -e B12X_COMPILE_CACHE_DIR=/cache/b12x \
    "${VLLM_ENV[@]}" \
    "${VLLM_OFFLOAD[@]}" \
    "${IMAGE}" \
        -lc 'unset MAX_NUM_BATCHED_TOKENS MAX_CUDAGRAPH_CAPTURE_SIZE CUDAGRAPH_CAPTURE_SIZES PREFILL_SCHEDULE_INTERVAL FAIRNESS_ENGINE PREFILL_COMPUTE_SHARE VLLM_PCIE_ALLREDUCE_BACKEND VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE VLLM_PCIE_TWOSHOT_ALLREDUCE_MAX_SIZE MAX_MODEL_LEN
exec /opt/venv/bin/vllm serve "$@"' -- \
            "${MODEL}" \
            `# Networking` \
            --host 0.0.0.0 \
            --port "${VLLM_PORT}" \
            `# Model identity` \
            --served-model-name "${MODELNAME}" \
            `# Quantization` \
            `# nvfp4: modelopt_mixed, no --trust-remote-code` \
            --quantization modelopt_mixed \
            --load-format safetensors \
            --dtype bfloat16 \
            --kv-cache-dtype "${KV_CACHE_DTYPE}" \
            --block-size "${BLOCK_SIZE}" \
            --mamba-cache-mode align \
            --recurrent-checkpoint-policy request_boundaries \
            --enable-prefix-caching \
            --enable-chunked-prefill \
            `# Parallelism` \
            --tensor-parallel-size "${TP_SIZE}" \
            --decode-context-parallel-size "${DCP_SIZE}" \
            --cp-kv-cache-interleave-size "${CP_KV_INTERLEAVE}" \
            --dcp-kv-cache-interleave-size "${CP_KV_INTERLEAVE}" \
            `# Resource limits` \
            --max-model-len "${CONTEXT_SIZE}" \
            --max-num-seqs "${MAX_NUM_SEQS}" \
            --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
            --kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}" \
            --gpu-memory-utilization "${GPU_UTIL}" \
            --compilation-config "{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"cudagraph_capture_sizes\":[${SIZES_CSV}]}" \
            `# Tokenizer / tools / reasoning` \
            --reasoning-parser glm45 \
            --tool-call-parser glm47 \
            --enable-auto-tool-choice \
            --chat-template /opt/glm53f/chat_template.multimodal.jinja \
            --default-chat-template-kwargs.reasoning_effort="${REASONING_EFFORT}" \
            --default-chat-template-kwargs.clear_thinking=false \
            `# b12x KDA prefill auto-engages on karmic` \
            `# the old kda_prefill_backend key fails the karmic resolver` \
            `# Serving statistics` \
            --enable-request-id-headers \
            --enable-force-include-usage \
            --enable-per-request-metrics \
            --enable-prompt-tokens-details \
            "${VLLM_BACKEND[@]}" \
            "${VLLM_EXTRA[@]}" \
            "${VLLM_SPEC[@]}" \
            `# SAMPLER` \
            --override-generation-config '{"temperature": 1, "top_p": 0.95}' \
            "$@"

printf 'Started %s on http://127.0.0.1:%s/v1 - podman logs -f %s\n' \
    "${PODNAME}" "${VLLM_PORT}" "${PODNAME}"
